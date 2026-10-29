"""Paper identities and a durable record of successfully delivered recommendations."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import unquote, urlsplit

from .library import parse_date, plain_text, utc
from .protocol import CorpusPaper, Paper

ARXIV_ID = r"(\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?(?:\.pdf)?(?=$|[\s?#])"


def identities(paper: Paper | CorpusPaper) -> set[str]:
    keys = set()
    title = unicodedata.normalize("NFKC", plain_text(paper.title)).casefold()
    title = "".join(c for c in title if c.isalnum())
    if title:
        keys.add("title:" + title)
    for value in (paper.doi, paper.url, paper.arxiv_id):
        value = unquote(value or "").strip().casefold()
        doi = re.search(r"10\.\d{4,9}/[^\s?#]+", value)
        if doi:
            normalized = doi.group().removesuffix(".full.pdf").rstrip("/")
            if normalized.startswith("10.1101/"):
                normalized = re.sub(r"v\d+$", "", normalized)
            keys.add("doi:" + normalized)
        arxiv = re.search(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv[:.\s]+)" + ARXIV_ID, value)
        if not arxiv:
            arxiv = re.match(ARXIV_ID, value)
        if arxiv:
            keys.add("arxiv:" + arxiv.group(1))
    if paper.url:
        try:
            url = urlsplit(paper.url)
        except ValueError:
            return keys
        if url.scheme in {"http", "https"}:
            keys.add("url:" + url.netloc.casefold() + url.path.rstrip("/"))
    return keys


def unseen_papers(candidates: list[Paper], library: list[CorpusPaper], history: set[str]) -> list[Paper]:
    known = set().union(*(identities(p) for p in library)) if library else set()
    selected = []
    for paper in candidates:
        keys = identities(paper)
        repeated = any(fingerprint(k) in history for k in keys)
        duplicate = bool(keys & known)
        # Keep aliases even when this copy is skipped (e.g. preprint vs publication).
        known.update(keys)
        if not duplicate and not repeated:
            selected.append(paper)
    return selected


def fingerprint(identity: str) -> str:
    return hashlib.sha256(identity.encode()).hexdigest()


class AlertHistory:
    def __init__(self, path: str | None, retention_days: int = 90, now: datetime | None = None):
        if retention_days < 1:
            raise ValueError("history_days must be positive")
        self.path = Path(path).expanduser() if path else None
        self.now = utc(now or datetime.now(timezone.utc))
        self.entries = {}
        if self.path and self.path.exists():
            # Fail visibly on corrupt state instead of quietly resending everything.
            data = json.loads(self.path.read_text())
            if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("sent"), dict):
                raise ValueError("Invalid alert history format")
            cutoff = self.now - timedelta(days=retention_days)
            for key, value in data["sent"].items():
                date = parse_date(value)
                if date and date >= cutoff:
                    self.entries[key] = value

    @property
    def seen(self) -> set[str]:
        return set(self.entries)

    def record(self, papers: list[Paper]):
        if self.path is None:
            return
        for paper in papers:
            for key in identities(paper):
                self.entries[fingerprint(key)] = self.now.isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({"version": 1, "sent": self.entries}, indent=2))
        temporary.replace(self.path)
