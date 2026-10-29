"""Build a reading profile from Zotero metadata, without reading private note text."""

from datetime import datetime, timezone
from html import unescape
import re

from .protocol import CorpusPaper
from .utils import glob_match

PAPER_TYPES = {"conferencePaper", "journalArticle", "preprint"}


def utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def parse_date(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except (ValueError, TypeError):
        return None


def plain_text(value: str) -> str:
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", value or "")).split())


def build_corpus(items: list[dict], collections: list[dict], settings) -> list[CorpusPaper]:
    by_key = {item.get("key", item["data"].get("key")): item["data"] for item in items}
    by_collection = {c.get("key", c["data"].get("key")): c["data"] for c in collections}
    reading_tags = {t.strip().casefold() for t in settings.get("reading_tags", [])}
    reading_paths = settings.get("reading_paths", [])

    def collection_path(key):
        names, visited = [], set()
        while key and key in by_collection and key not in visited:
            visited.add(key)
            collection = by_collection[key]
            names.append(collection["name"])
            key = collection.get("parentCollection")
        return "/".join(reversed(names))

    corpus = {}
    for key, data in by_key.items():
        if data.get("itemType") not in PAPER_TYPES or data.get("deleted"):
            continue
        title = plain_text(data.get("title", ""))
        if not title:
            continue
        paths = [collection_path(k) for k in data.get("collections", []) if k in by_collection]
        tags = [t["tag"] for t in data.get("tags", []) if t.get("tag")]
        signals = [f"reading tag: {t}" for t in tags if t.strip().casefold() in reading_tags]
        signals.extend(f"reading collection: {p}" for p in paths
                       if any(glob_match(p, pattern) for pattern in reading_paths))
        # Bibliographic edits are deliberately not treated as reading activity.
        corpus[key] = CorpusPaper(
            title=title,
            abstract=plain_text(data.get("abstractNote", "")),
            added_date=parse_date(data.get("dateAdded")) or datetime(1970, 1, 1, tzinfo=timezone.utc),
            paths=paths,
            key=key or "",
            url=data.get("url", ""),
            doi=data.get("DOI", ""),
            arxiv_id=" ".join(str(data.get(f, "")) for f in ("archiveID", "archiveLocation", "extra")),
            tags=tags,
            reading_signals=signals,
            is_reading=bool(signals),
        )

    if settings.get("use_reading_activity", True):
        for data in by_key.values():
            kind = data.get("itemType")
            if kind not in {"annotation", "note"} or data.get("deleted"):
                continue
            parent, visited = data.get("parentItem"), set()
            # Annotations belong to attachments, which belong to papers.
            while parent and parent not in corpus and parent not in visited:
                visited.add(parent)
                parent = by_key.get(parent, {}).get("parentItem")
            if parent not in corpus:
                continue
            date = parse_date(data.get("dateModified")) or parse_date(data.get("dateAdded"))
            paper = corpus[parent]
            if date and (paper.last_read_date is None or date > paper.last_read_date):
                paper.last_read_date = date
                paper.reading_signals = [s for s in paper.reading_signals if not s.startswith("activity:")]
                paper.reading_signals.append(f"activity: {kind} updated {date:%Y-%m-%d}")
    return list(corpus.values())
