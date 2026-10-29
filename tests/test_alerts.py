from datetime import datetime, timedelta, timezone
import json

import pytest

from zotero_arxiv_daily.alerts import AlertHistory, identities, unseen_papers
from zotero_arxiv_daily.protocol import Paper


def paper(title="A study", url="https://arxiv.org/abs/2401.12345v1", **fields):
    return Paper("arxiv", title, [], "abstract", url, **fields)


def test_known_papers_match_versions_dois_and_normalized_titles():
    known = paper(doi="10.1000/EXAMPLE")
    candidates = [
        paper("Changed title", "https://arxiv.org/pdf/2401.12345v3.pdf"),
        paper("Published title", "https://doi.org/10.1000/example"),
        paper("A STUDY!", "https://example.org/different"),
        paper("Fresh", "https://example.org/fresh"),
        paper("Fresh!", "https://another.org/fresh"),
    ]
    result = unseen_papers(candidates, [known], set())
    assert [p.title for p in result] == ["Fresh"]


def test_old_arxiv_identifiers_and_zotero_extra():
    assert "arxiv:hep-th/9901001" in identities(paper(url="https://arxiv.org/pdf/hep-th/9901001v2.pdf"))
    assert "arxiv:2401.12345" in identities(paper(url="", arxiv_id="arXiv: 2401.12345v3\nOther metadata"))


def test_history_roundtrip_expiry_and_no_plaintext_titles(tmp_path):
    path = tmp_path / "history.json"
    now = datetime(2026, 10, 29, tzinfo=timezone.utc)
    state = AlertHistory(str(path), now=now)
    assert not path.exists()
    state.record([paper()])
    assert "A study" not in path.read_text()
    loaded = AlertHistory(str(path), now=now + timedelta(days=1))
    assert unseen_papers([paper("Updated title", "https://arxiv.org/abs/2401.12345v2")], [], loaded.seen) == []
    expired = AlertHistory(str(path), now=now + timedelta(days=91))
    assert expired.seen == set()
    assert json.loads(path.read_text())["version"] == 1


def test_corrupt_history_fails_visibly_and_disabled_history_does_not_write(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"version": 2}')
    with pytest.raises(ValueError, match="history"):
        AlertHistory(str(path))
    state = AlertHistory(None)
    state.record([paper()])
    assert not state.seen


def test_biorxiv_versions_normalize_but_unrelated_dois_are_not_arxiv_ids():
    known = paper("First title", "https://www.biorxiv.org/content/10.1101/2026.10.28.123456v1.full.pdf")
    updated = paper("Updated title", "https://doi.org/10.1101/2026.10.28.123456")
    assert unseen_papers([updated], [known], set()) == []
    keys = identities(paper(url="https://doi.org/10.1000/example/1234567"))
    assert not any(k.startswith("arxiv:") for k in keys)


def test_malformed_url_does_not_break_title_deduplication():
    assert "title:astudy" in identities(paper(url="http://[invalid"))
