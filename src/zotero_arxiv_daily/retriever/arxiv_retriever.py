from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf
from tempfile import TemporaryDirectory
import feedparser
from urllib.request import urlopen
import shutil
from tqdm import tqdm
import os
from loguru import logger
@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")
    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=10,delay_seconds=10)
        query = '+'.join(self.config.source.arxiv.category)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.get('title', ''):
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        if feed.get('status', 200) >= 400 or (feed.bozo and not feed.entries):
            raise RuntimeError(f"Failed to read arXiv feed for {query}")
        raw_papers = []
        all_paper_ids = [i.id.removeprefix("oai:arXiv.org:") for i in feed.entries if i.get("arxiv_announce_type","new") == 'new']
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]
            if not all_paper_ids and self.config.executor.get("dry_run", False):
                logger.info("No new arXiv announcements; preview uses the 10 latest papers in the configured categories")
                search = arxiv.Search(
                    query=" OR ".join(f"cat:{category}" for category in self.config.source.arxiv.category),
                    max_results=10,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending,
                )
                return list(client.results(search))

        # Get full information of each paper from arxiv api
        bar = tqdm(total=len(all_paper_ids))
        for i in range(0,len(all_paper_ids),20):
            search = arxiv.Search(id_list=all_paper_ids[i:i+20])
            batch = list(client.results(search))
            bar.update(len(batch))
            raw_papers.extend(batch)
        bar.close()

        return raw_papers

    def convert_to_paper(self, raw_paper:ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            doi=raw_paper.doi or "",
            arxiv_id=raw_paper.get_short_id(),
        )

    def enrich_paper(self, paper: Paper) -> None:
        if paper.full_text or not paper.pdf_url:
            return
        try:
            with TemporaryDirectory() as temp_dir:
                path = os.path.join(temp_dir, "paper.pdf")
                with urlopen(paper.pdf_url, timeout=30) as response, open(path, "wb") as target:
                    shutil.copyfileobj(response, target)
                paper.full_text = extract_markdown_from_pdf(path)
        except Exception as e:
            logger.warning(f"Using abstract for {paper.title}; PDF unavailable: {e}")
