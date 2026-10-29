from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Type

import numpy as np
from omegaconf import DictConfig

from ..library import utc
from ..protocol import CorpusPaper, LibraryMatch, Paper


class BaseReranker(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    def rerank(self, candidates: list[Paper], corpus: list[CorpusPaper], *, now: datetime | None = None) -> list[Paper]:
        if not candidates or not corpus:
            return []
        settings = self.config.reranker.get("profile", {})
        half_life = float(settings.get("half_life_days", 45))
        activity_half_life = float(settings.get("activity_half_life_days", 30))
        focus_weight = float(settings.get("focus_weight", 0.7))
        top_k = int(settings.get("neighbor_count", 3))
        if (not np.isfinite([half_life, activity_half_life, focus_weight]).all()
                or half_life <= 0 or activity_half_life <= 0 or not 0 <= focus_weight <= 1 or top_k < 1):
            raise ValueError("Invalid reranker.profile: half-lives and neighbor_count must be positive; focus_weight must be in [0, 1]")
        now = utc(now or datetime.now(timezone.utc))

        def decay(date, days):
            age = max(0.0, (now - utc(date)).total_seconds() / 86400)
            return 2 ** (-age / days)

        activity = []
        for paper in corpus:
            weight = 0.75 * decay(paper.added_date, half_life)
            if paper.last_read_date:
                weight = max(weight, decay(paper.last_read_date, activity_half_life))
            if paper.is_reading:
                weight = 1.0
            activity.append(0.15 + 0.85 * weight)

        # Titles rescue items without abstracts and improve specificity.
        def texts(papers):
            return [f"{p.title}\n{p.abstract}".strip() for p in papers]

        sim = np.asarray(self.get_similarity_score(texts(candidates), texts(corpus)), dtype=float)
        if sim.shape != (len(candidates), len(corpus)) or not np.isfinite(sim).all():
            raise ValueError("Embedding similarity matrix has an invalid shape or non-finite values")
        sim = np.clip(sim, 0, 1)
        focus = sim * np.asarray(activity)
        k = min(top_k, len(corpus))
        # Nearest neighbors preserve small research interests in a large library.
        library_score = 0.75 * sim.max(axis=1) + 0.25 * np.sort(sim, axis=1)[:, -k:].mean(axis=1)
        focus_score = 0.75 * focus.max(axis=1) + 0.25 * np.sort(focus, axis=1)[:, -k:].mean(axis=1)
        scores = 10 * ((1 - focus_weight) * library_score + focus_weight * focus_score)
        contributions = (1 - focus_weight) * sim + focus_weight * focus
        for row, (paper, score) in enumerate(zip(candidates, scores)):
            paper.score = float(score)
            paper.library_matches = []
            for index in np.argsort(-contributions[row], kind="stable")[:k]:
                if sim[row, index] <= 0:
                    continue
                match = corpus[index]
                link = f"zotero://select/library/items/{match.key}" if match.key else match.url
                signals = [f"added {match.added_date:%Y-%m-%d}", *match.reading_signals]
                paper.library_matches.append(LibraryMatch(match.title, link, float(sim[row, index]), signals))
        return sorted(candidates, key=lambda paper: paper.score, reverse=True)

    @abstractmethod
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        raise NotImplementedError


registered_rerankers = {}


def register_reranker(name: str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator


def get_reranker_cls(name: str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]
