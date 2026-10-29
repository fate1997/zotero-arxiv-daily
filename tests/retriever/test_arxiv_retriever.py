from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever
import feedparser
import pickle
import pytest

@pytest.mark.integration
def test_arxiv_retriever(config, monkeypatch):

    parsed_result = feedparser.parse("tests/retriever/arxiv_rss_example.xml")
    raw_parser = feedparser.parse
    def mock_feedparser_parse(url):
        if url == f"https://rss.arxiv.org/atom/{'+'.join(config.source.arxiv.category)}":
            return parsed_result
        return raw_parser(url)
    monkeypatch.setattr(feedparser, "parse", mock_feedparser_parse)
    
    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()
    parsed_results = [i for i in parsed_result.entries if i.get("arxiv_announce_type","new") == 'new']
    assert len(papers) == len(parsed_results)
    paper_titles = [i.title for i in papers]
    parsed_titles = [i.title for i in parsed_results]
    assert set(paper_titles) == set(parsed_titles)


def test_metadata_conversion_does_not_fetch_pdf(config, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    download = Mock(side_effect=AssertionError("PDF should not be fetched before selection"))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.arxiv_retriever.urlopen", download)
    raw = SimpleNamespace(title="Title", authors=[SimpleNamespace(name="Author")], summary="Abstract",
                          pdf_url="https://arxiv.org/pdf/2401.12345", entry_id="https://arxiv.org/abs/2401.12345",
                          doi=None, get_short_id=lambda: "2401.12345v1")
    paper = ArxivRetriever(config).convert_to_paper(raw)
    assert paper.full_text is None
    assert paper.arxiv_id == "2401.12345v1"
    download.assert_not_called()


def test_failed_pdf_falls_back_to_abstract(config, monkeypatch):
    from unittest.mock import Mock
    from zotero_arxiv_daily.protocol import Paper

    monkeypatch.setattr("zotero_arxiv_daily.retriever.arxiv_retriever.urlopen", Mock(side_effect=TimeoutError))
    paper = Paper("arxiv", "Title", [], "Usable abstract", "https://arxiv.org/abs/2401.12345",
                  pdf_url="https://arxiv.org/pdf/2401.12345")
    ArxivRetriever(config).enrich_paper(paper)
    assert paper.full_text is None
    assert paper.abstract == "Usable abstract"
