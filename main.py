import arxiv
import requests
from urllib.parse import quote

def _get_pdf_url_patch(links) -> str:
    """
    Finds the PDF link among a result's links and returns its URL.
    Should only be called once for a given `Result`, in its constructor.
    After construction, the URL should be available in `Result.pdf_url`.
    """
    pdf_urls = [link.href for link in links if "pdf" in link.href]
    if len(pdf_urls) == 0:
        return None
    return pdf_urls[0]

arxiv.Result._get_pdf_url = _get_pdf_url_patch

import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pyzotero import zotero
from recommender import rerank_paper
from construct_email import render_email, send_email
from tqdm import trange,tqdm
from loguru import logger
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from paper import ArxivPaper
from llm import set_global_llm
import feedparser

def get_zotero_corpus(id:str,key:str) -> list[dict]:
    zot = zotero.Zotero(id, 'user', key)
    collections = zot.everything(zot.collections())
    collections = {c['key']:c for c in collections}
    corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    corpus = [c for c in corpus if c['data']['abstractNote'] != '']
    def get_collection_path(col_key:str) -> str:
        if p := collections[col_key]['data']['parentCollection']:
            return get_collection_path(p) + '/' + collections[col_key]['data']['name']
        else:
            return collections[col_key]['data']['name']
    for c in corpus:
        paths = [get_collection_path(col) for col in c['data']['collections']]
        c['paths'] = paths
    return corpus

def filter_corpus(corpus:list[dict], pattern:str) -> list[dict]:
    _,filename = mkstemp()
    with open(filename,'w') as file:
        file.write(pattern)
    matcher = parse_gitignore(filename,base_dir='./')
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c['paths']]
        if not any(match_results):
            new_corpus.append(c)
    os.remove(filename)
    return new_corpus

ARXIV_API_ERROR = False  # 是否在本次运行中遇到过 arxiv.HTTPError

def get_arxiv_paper(query: str, debug: bool = False) -> list[ArxivPaper]:
    global ARXIV_API_ERROR
    client = arxiv.Client(num_retries=10, delay_seconds=10)

    # RSS 拉取：用 requests 先取文本，再交给 feedparser 解析，避免 feedparser 直接联网时
    # 在 GHA 上偶发拿到 HTML 错误页导致没有 title
    feed_url = f"https://rss.arxiv.org/atom/{quote(query, safe='')}"
    headers = {
        # 建议换成你自己的邮箱（更稳定、更礼貌）
        "User-Agent": "zotero-arxiv-daily/1.0 (mailto:tingting.wang@imperial.ac.uk)"
    }

    try:
        r = requests.get(feed_url, headers=headers, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        ARXIV_API_ERROR = True
        logger.error(f"ArXiv RSS request failed: {e} url={feed_url}")
        return []

    feed = feedparser.parse(r.text)

    feed_title = feed.feed.get("title", "")
    if not feed_title:
        # 解析失败或返回的不是 Atom XML 时，这里通常为空
        logger.error(
            "ArXiv RSS parsed but missing title. "
            f"http_status={r.status_code} url={feed_url} "
            f"bozo={getattr(feed, 'bozo', None)} "
            f"bozo_exception={getattr(feed, 'bozo_exception', None)}"
        )
        return []

    if "Feed error for query" in feed_title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")
    
    if not getattr(feed, "entries", None):
        logger.warning(f"ArXiv RSS has no entries. title={feed_title} url={feed_url}")
        return []

    if not debug:
        papers: list[ArxivPaper] = []
        all_paper_ids = [
            i.id.removeprefix("oai:arXiv.org:")
            for i in feed.entries
            if i.arxiv_announce_type == 'new'
        ]

        bar = tqdm(total=len(all_paper_ids), desc="Retrieving Arxiv papers")

        for i in range(0, len(all_paper_ids), 20):
            batch_ids = all_paper_ids[i:i + 20]
            search = arxiv.Search(id_list=batch_ids)

            try:
                # 这里可能会抛 arxiv.HTTPError(429/503)
                results = list(client.results(search))
            except arxiv.HTTPError as e:
                ARXIV_API_ERROR = True
                logger.error(
                    f"Arxiv API failed when fetching batch {i // 20} "
                    f"(ids={batch_ids}): {e}"
                )
                # 如果一篇都还没拿到，就直接返回空列表，
                # 让上层当作“今天没有新论文 / API 出错”处理
                if not papers:
                    bar.close()
                    return []
                # 如果已经拿到部分结果，就停止继续请求，返回已有的
                break

            batch = [ArxivPaper(p) for p in results]
            bar.update(len(batch))
            papers.extend(batch)

        bar.close()

    else:
        logger.debug("Retrieve 5 arxiv papers regardless of the date (debug mode).")
        search = arxiv.Search(
            query="cat:cs.AI",
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        papers: list[ArxivPaper] = []

        try:
            for result in client.results(search):
                papers.append(ArxivPaper(result))
                if len(papers) == 5:
                    break
        except arxiv.HTTPError as e:
            ARXIV_API_ERROR = True
            logger.error(f"Arxiv API failed in debug mode: {e}")
            papers = []

    return papers


parser = argparse.ArgumentParser(description='Recommender system for academic papers')

def add_argument(*args, **kwargs):
    def get_env(key:str,default=None):
        # handle environment variables generated at Workflow runtime
        # Unset environment variables are passed as '', we should treat them as None
        v = os.environ.get(key)
        if v == '' or v is None:
            return default
        return v
    parser.add_argument(*args, **kwargs)
    arg_full_name = kwargs.get('dest',args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)
    if env_value is not None:
        #convert env_value to the specified type
        if kwargs.get('type') == bool:
            env_value = env_value.lower() in ['true','1']
        else:
            env_value = kwargs.get('type')(env_value)
        parser.set_defaults(**{arg_full_name:env_value})


if __name__ == '__main__':
    
    add_argument('--zotero_id', type=str, help='Zotero user ID')
    add_argument('--zotero_key', type=str, help='Zotero API key')
    add_argument('--zotero_ignore',type=str,help='Zotero collection to ignore, using gitignore-style pattern.')
    add_argument('--send_empty', type=bool, help='If get no arxiv paper, send empty email',default=False)
    add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend',default=100)
    add_argument('--arxiv_query', type=str, help='Arxiv search query')
    add_argument('--smtp_server', type=str, help='SMTP server')
    add_argument('--smtp_port', type=int, help='SMTP port')
    add_argument('--sender', type=str, help='Sender email address')
    add_argument('--receiver', type=str, help='Receiver email address')
    add_argument('--sender_password', type=str, help='Sender email password')
    add_argument(
        "--use_llm_api",
        type=bool,
        help="Use OpenAI API to generate TLDR",
        default=False,
    )
    add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
        default=None,
    )
    add_argument(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
        default="https://api.openai.com/v1",
    )
    add_argument(
        "--model_name",
        type=str,
        help="LLM Model Name",
        default="gpt-4o",
    )
    add_argument(
        "--language",
        type=str,
        help="Language of TLDR",
        default="English",
    )
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    assert (
        not args.use_llm_api or args.openai_api_key is not None
    )  # If use_llm_api is True, openai_api_key must be provided
    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    if args.zotero_ignore:
        logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
        corpus = filter_corpus(corpus, args.zotero_ignore)
        logger.info(f"Remaining {len(corpus)} papers after filtering.")
    logger.info("Retrieving Arxiv papers...")
    papers = get_arxiv_paper(args.arxiv_query, args.debug)
    
    if len(papers) == 0:
        if ARXIV_API_ERROR:
            logger.warning(
                "No papers were retrieved from arXiv because the API returned "
                "errors (429/503). Treating this as 'no new papers' for today."
            )
        else:
            logger.info(
                "No new papers found. Maybe yesterday was a holiday and nobody "
                "submitted anything :). If this is not the case, please check "
                "the ARXIV_QUERY."
            )
    
        if not args.send_empty:
            sys.exit(0)
            
    else:
        logger.info("Reranking papers...")
        papers = rerank_paper(papers, corpus)
        if args.max_paper_num != -1:
            papers = papers[:args.max_paper_num]
        if args.use_llm_api:
            logger.info("Using OpenAI API as global LLM.")
            set_global_llm(api_key=args.openai_api_key, base_url=args.openai_api_base, model=args.model_name, lang=args.language)
        else:
            logger.info("Using Local LLM as global LLM.")
            set_global_llm(lang=args.language)

    html = render_email(papers)
    logger.info("Sending email...")
    send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html)
    logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")

