from dataclasses import asdict
import json
from pathlib import Path

from loguru import logger
from openai import OpenAI
from omegaconf import DictConfig
from pyzotero import zotero
from tqdm import tqdm

from .alerts import AlertHistory, unseen_papers
from .construct_email import render_email
from .library import build_corpus
from .protocol import CorpusPaper
from .reranker import get_reranker_cls
from .retriever import get_retriever_cls
from .utils import glob_match, send_email


class Executor:
    def __init__(self, config: DictConfig):
        self.config = config
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching Zotero library and reading activity")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        types = 'conferencePaper || journalArticle || preprint'
        if self.config.zotero.get("use_reading_activity", True):
            types += ' || attachment || note || annotation'
        items = zot.everything(zot.items(itemType=types))
        corpus = build_corpus(items, collections, self.config.zotero)
        logger.info(f"Fetched {len(corpus)} Zotero papers; {sum(p.is_reading or p.last_read_date is not None for p in corpus)} have reading signals")
        return corpus

    def filter_corpus(self, corpus: list[CorpusPaper]) -> list[CorpusPaper]:
        include = self.config.zotero.include_path
        excluded_tags = {t.casefold() for t in self.config.zotero.get("exclude_tags", [])}
        selected = [p for p in corpus
                    if (not include or any(glob_match(path, include) for path in p.paths))
                    and not any(t.casefold() in excluded_tags for t in p.tags)]
        logger.info(f"Using {len(selected)} Zotero papers for the reading profile")
        return selected

    def run(self):
        settings = self.config.executor
        limit = int(settings.max_paper_num)
        threshold = float(settings.get("min_score", 5.0))
        if limit < 1 or not 0 <= threshold <= 10:
            raise ValueError("max_paper_num must be positive and min_score must be in [0, 10]")
        history = AlertHistory(settings.get("history_path", ".cache/alert-history.json"),
                               int(settings.get("history_days", 90)))
        library = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(library)
        if not corpus:
            # Never interpolate Zotero config here: it contains the API key.
            raise ValueError("No Zotero papers matched. Check library access, include_path, and exclude_tags.")
        candidates = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            candidates.extend(retriever.retrieve_papers())
        retrieved = len(candidates)
        candidates = unseen_papers(candidates, library, history.seen)
        logger.info(f"{retrieved} retrieved; {len(candidates)} remain after library and alert deduplication")
        ranked = self.reranker.rerank(candidates, corpus) if candidates else []
        selected = [p for p in ranked if p.score is not None and p.score >= threshold][:limit]
        logger.info(f"Selected {len(selected)} papers with relevance at least {threshold}/10")
        dry_run = settings.get("dry_run", False)
        if not selected and not settings.send_empty and not dry_run:
            logger.info("No new papers cleared the alert filters. No email will be sent.")
            return []
        if selected and settings.get("generate_summaries", True):
            client = OpenAI(api_key=self.config.llm.api.key, base_url=self.config.llm.api.base_url)
            for paper in tqdm(selected):
                if settings.get("fetch_full_text", True):
                    self.retrievers[paper.source].enrich_paper(paper)
                paper.generate_tldr(client, self.config.llm)
                paper.generate_affiliations(client, self.config.llm)
        content = render_email(selected)
        if dry_run:
            destination = Path(settings.get("output_dir", "outputs/preview")).expanduser()
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "digest.html").write_text(content, encoding="utf-8")
            (destination / "digest.json").write_text(
                json.dumps([asdict(p) for p in selected], indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(f"Preview saved to {destination}; email and alert history unchanged")
        else:
            send_email(self.config, content)
            # A failed delivery must remain eligible for the next run.
            history.record(selected)
            logger.info("Email sent and alert history saved")
        return selected
