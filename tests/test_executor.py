from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from zotero_arxiv_daily.alerts import AlertHistory
from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.protocol import CorpusPaper, Paper


def setup_executor(config, monkeypatch, tmp_path):
    config.executor.history_path = str(tmp_path / "history.json")
    config.executor.output_dir = str(tmp_path / "preview")
    config.executor.generate_summaries = False
    executor = Executor(config)
    library = [CorpusPaper("Library", "abstract", datetime.now(timezone.utc), [], key="KEY")]
    monkeypatch.setattr(executor, "fetch_zotero_corpus", lambda: library)
    candidates = [Paper("arxiv", title, [], "abstract", "https://example.org/" + title) for title in ["Strong", "Weak"]]
    retriever = Mock()
    retriever.retrieve_papers.return_value = candidates
    executor.retrievers = {"arxiv": retriever}
    def rank(papers, corpus):
        for p in papers:
            p.score = 8.0 if p.title == "Strong" else 2.0
        return papers
    executor.reranker = Mock()
    executor.reranker.rerank.side_effect = rank
    sender = Mock()
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", sender)
    return executor, sender, retriever, candidates


def test_preview_filters_without_sending_or_consuming_history(config, monkeypatch, tmp_path):
    executor, sender, retriever, _ = setup_executor(config, monkeypatch, tmp_path)
    config.executor.dry_run = True
    selected = executor.run()
    assert [p.title for p in selected] == ["Strong"]
    assert (tmp_path / "preview/digest.html").exists()
    assert '"title": "Strong"' in (tmp_path / "preview/digest.json").read_text()
    assert not (tmp_path / "history.json").exists()
    sender.assert_not_called()
    retriever.enrich_paper.assert_not_called()


def test_successful_send_is_not_repeated(config, monkeypatch, tmp_path):
    executor, sender, _, candidates = setup_executor(config, monkeypatch, tmp_path)
    assert len(executor.run()) == 1
    assert AlertHistory(config.executor.history_path).seen
    assert executor.run() == []
    sender.assert_called_once()
    assert executor.reranker.rerank.call_args.args[0] == [candidates[1]]


def test_failed_send_does_not_record_papers(config, monkeypatch, tmp_path):
    executor, sender, _, _ = setup_executor(config, monkeypatch, tmp_path)
    sender.side_effect = RuntimeError("SMTP unavailable")
    with pytest.raises(RuntimeError, match="SMTP"):
        executor.run()
    assert not (tmp_path / "history.json").exists()


def test_all_candidates_below_threshold_do_not_trigger_email(config, monkeypatch, tmp_path):
    executor, sender, _, _ = setup_executor(config, monkeypatch, tmp_path)
    config.executor.min_score = 9
    assert executor.run() == []
    sender.assert_not_called()
    config.executor.send_empty = True
    assert executor.run() == []
    sender.assert_called_once()


def test_only_selected_papers_get_full_text_and_summaries(config, monkeypatch, tmp_path):
    executor, _, retriever, candidates = setup_executor(config, monkeypatch, tmp_path)
    config.executor.generate_summaries = True
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", Mock())
    for p in candidates:
        p.generate_tldr = Mock()
        p.generate_affiliations = Mock()
    executor.run()
    retriever.enrich_paper.assert_called_once_with(candidates[0])
    candidates[0].generate_tldr.assert_called_once()
    candidates[1].generate_tldr.assert_not_called()


def test_empty_profile_error_does_not_leak_credentials(config, monkeypatch, tmp_path):
    executor, sender, _, _ = setup_executor(config, monkeypatch, tmp_path)
    config.zotero.include_path = "does-not-exist"
    with pytest.raises(ValueError) as error:
        executor.run()
    assert config.zotero.api_key not in str(error.value)
    sender.assert_not_called()


def test_library_filter_uses_tags_case_insensitively(config, monkeypatch, tmp_path):
    executor, _, _, _ = setup_executor(config, monkeypatch, tmp_path)
    papers = executor.fetch_zotero_corpus()
    papers[0].tags = ["Ignore"]
    config.zotero.exclude_tags = ["ignore"]
    assert executor.filter_corpus(papers) == []


def test_fetch_paginates_activity_and_collection_data(config, monkeypatch, tmp_path):
    executor = Executor(config)
    client = Mock()
    client.everything.side_effect = lambda data: data
    client.collections.return_value = []
    client.items.return_value = [{"key": "P", "data": {
        "itemType": "preprint", "title": "Title only", "dateAdded": "2026-10-29T00:00:00Z",
    }}]
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", Mock(return_value=client))
    assert executor.fetch_zotero_corpus()[0].title == "Title only"
    assert client.everything.call_count == 2
    assert "annotation" in client.items.call_args.kwargs["itemType"]


def test_already_owned_papers_are_excluded_even_outside_profile(config, monkeypatch, tmp_path):
    executor, sender, retriever, candidates = setup_executor(config, monkeypatch, tmp_path)
    profile = CorpusPaper("Profile", "abstract", datetime.now(timezone.utc), ["Included"])
    known = CorpusPaper("Strong", "abstract", datetime.now(timezone.utc), ["Excluded"])
    monkeypatch.setattr(executor, "fetch_zotero_corpus", lambda: [profile, known])
    config.zotero.include_path = "Included"
    executor.run()
    sender.assert_not_called()
    assert executor.reranker.rerank.call_args.args[0] == [candidates[1]]
    assert executor.reranker.rerank.call_args.args[1] == [profile]
