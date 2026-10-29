from zotero_arxiv_daily.construct_email import render_email
from zotero_arxiv_daily.protocol import LibraryMatch, Paper


def test_digest_explains_matches_and_escapes_external_content():
    paper = Paper("arxiv", '<script>alert(1)</script>', ['A & B'], '<b>abstract</b>',
                  "https://arxiv.org/abs/2401.12345", score=7.2,
                  pdf_url='javascript:alert(1)',
                  library_matches=[LibraryMatch('My <paper>', 'zotero://select/library/items/ABC', 0.9,
                                                 ['reading tag: <reading>', 'added 2026-10-28'])])
    html = render_email([paper])
    assert 'Why this paper' in html
    assert 'zotero://select/library/items/ABC' in html
    assert 'reading tag: &lt;reading&gt;' in html
    assert '&lt;script&gt;' in html
    assert '<script>' not in html
    assert 'javascript:' not in html
    assert 'None' not in html
    assert '7.2/10' in html
    assert 'Abstract:' in html
    assert 'Paper page / save to Zotero' in html


def test_empty_digest_explains_filtering():
    assert 'cleared your alert filters' in render_email([])
