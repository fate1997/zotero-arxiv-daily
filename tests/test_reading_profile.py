from datetime import datetime, timedelta, timezone

import numpy as np
from omegaconf import OmegaConf
import pytest

from zotero_arxiv_daily.library import build_corpus
from zotero_arxiv_daily.protocol import CorpusPaper, Paper
from zotero_arxiv_daily.reranker.base import BaseReranker

NOW = datetime(2026, 10, 29, tzinfo=timezone.utc)


def item(key, kind="journalArticle", **fields):
    return {"key": key, "data": {"itemType": kind, "title": key, "dateAdded": "2020-01-01T00:00:00Z", **fields}}


def test_activity_follows_attachment_to_paper_and_keeps_title_only_items():
    items = [
        item("P", abstractNote="", collections=["CHILD"], tags=[{"tag": "Reading"}]),
        item("PDF", "attachment", parentItem="P", dateModified="2026-10-29T00:00:00Z"),
        item("A", "annotation", parentItem="PDF", dateModified="2026-10-27T12:00:00+02:00"),
        item("N", "note", parentItem="P", dateModified="2026-10-28T10:00:00Z", note="private text"),
        item("UNRELATED", dateModified="2026-10-29T00:00:00Z"),
        item("ORPHAN", "note", parentItem="MISSING"),
        item("LOOP", "annotation", parentItem="LOOP"),
    ]
    collections = [
        {"key": "ROOT", "data": {"name": "Projects", "parentCollection": False}},
        {"key": "CHILD", "data": {"name": "Current", "parentCollection": "ROOT"}},
    ]
    papers = build_corpus(items, collections, {"reading_tags": ["reading"], "reading_paths": ["Projects/*"]})
    assert len(papers) == 2
    paper, unrelated = papers
    assert paper.abstract == ""
    assert paper.paths == ["Projects/Current"]
    assert paper.is_reading
    assert paper.last_read_date == datetime(2026, 10, 28, 10, tzinfo=timezone.utc)
    assert "activity: note updated 2026-10-28" in paper.reading_signals
    assert "private text" not in str(paper)
    assert unrelated.last_read_date is None
    assert not unrelated.is_reading


def test_reading_activity_can_be_disabled_and_malformed_dates_are_safe():
    papers = build_corpus([
        item("P", title="<i>A &amp; B</i>", abstractNote="<p>Some <b>text</b></p>", dateAdded="bad"),
        item("N", "note", parentItem="P", dateModified=NOW.isoformat()),
    ], [], {"use_reading_activity": False})
    assert papers[0].title == "A & B"
    assert papers[0].abstract == "Some text"
    assert papers[0].last_read_date is None
    assert papers[0].added_date.year == 1970


class FixedReranker(BaseReranker):
    def __init__(self, similarities, **profile):
        super().__init__(OmegaConf.create({"reranker": {"profile": profile}}))
        self.similarities = np.array(similarities)
        self.texts = None

    def get_similarity_score(self, s1, s2):
        self.texts = s1, s2
        return self.similarities


def candidate(title):
    return Paper("arxiv", title, [], "candidate abstract", f"https://example.org/{title}")


def corpus(title, age=0, **kwargs):
    return CorpusPaper(title, "", NOW - timedelta(days=age), [], **kwargs)


def test_actual_age_changes_rank_and_recent_annotations_revive_old_interest():
    recent, old = corpus("Recent"), corpus("Old", age=1000, key="OLD")
    ranker = FixedReranker([[0.9, 0.1], [0.1, 0.9]])
    candidates = [candidate("new topic"), candidate("old topic")]
    ranked = ranker.rerank(candidates, [recent, old], now=NOW)
    assert ranked[0].title == "new topic"
    assert "Recent" in ranker.texts[1][0]  # title-only items participate
    old.last_read_date = NOW
    old.reading_signals = ["activity: annotation updated 2026-10-29"]
    ranked = ranker.rerank(candidates, [recent, old], now=NOW)
    assert ranked[0].title == "old topic"
    assert ranked[0].library_matches[0].url == "zotero://select/library/items/OLD"
    assert "annotation" in ranked[0].library_matches[0].signals[1]


def test_elapsed_time_matters_even_without_new_additions():
    ranker = FixedReranker([[0.9]])
    library = [corpus("One")]
    first = ranker.rerank([candidate("First")], library, now=NOW)[0].score
    later = ranker.rerank([candidate("Later")], library, now=NOW + timedelta(days=90))[0].score
    assert later < first


def test_niche_interest_survives_large_unrelated_archive():
    ranker = FixedReranker([[0.9] + [0.0] * 100])
    library = [corpus("Current", is_reading=True)] + [corpus(f"Archive {i}", age=1000) for i in range(100)]
    result = ranker.rerank([candidate("Niche")], library, now=NOW)[0]
    assert result.score > 7
    assert len(result.library_matches) == 1


def test_explicit_reading_tag_overrides_age_and_handles_naive_dates():
    old = corpus("Old", age=2000, is_reading=True)
    old.added_date = old.added_date.replace(tzinfo=None)
    ranker = FixedReranker([[0.9, 0.1], [0.1, 0.9]])
    ranked = ranker.rerank([candidate("Tagged"), candidate("New")], [old, corpus("Recent")], now=NOW)
    assert ranked[0].title == "Tagged"
    assert all(0 <= p.score <= 10 for p in ranked)


@pytest.mark.parametrize("profile", [{"half_life_days": 0}, {"activity_half_life_days": -1}, {"focus_weight": 2}, {"neighbor_count": 0}])
def test_invalid_profile_rejected(profile):
    with pytest.raises(ValueError):
        FixedReranker([[0.5]], **profile).rerank([candidate("X")], [corpus("Y")])


def test_empty_and_invalid_embeddings():
    ranker = FixedReranker([[float("nan")]])
    assert ranker.rerank([], [corpus("Y")]) == []
    assert ranker.rerank([candidate("X")], []) == []
    with pytest.raises(ValueError, match="non-finite"):
        ranker.rerank([candidate("X")], [corpus("Y")])
