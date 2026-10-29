"""Render an explainable digest; all external text and URLs are escaped."""

from html import escape
from urllib.parse import urlsplit

from .protocol import LibraryMatch, Paper

framework = """<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;background:#f3f5f7;color:#243040;font-family:Arial,sans-serif">
<div style="max-width:760px;margin:auto;padding:24px">
<h1 style="font-size:26px">Your research reading radar</h1>
__CONTENT__
<p style="font-size:12px;color:#657080">Relevance is a ranking score, not a probability.
Reading signals reflect additions, configured tags/collections, and synced note or annotation activity.
Open a paper's page to save it with the Zotero Connector.</p>
<p style="font-size:12px;color:#657080">To unsubscribe, disable the daily workflow in your GitHub repository.</p>
</div></body></html>"""


def safe_url(value: str | None, *, zotero: bool = False) -> str:
    try:
        parsed = urlsplit(value or "")
    except ValueError:
        return ""
    if parsed.scheme in {"https", "http"} and parsed.netloc:
        return escape(value, quote=True)
    if zotero and parsed.scheme == "zotero" and parsed.netloc == "select":
        return escape(value, quote=True)
    return ""


def get_empty_html():
    return '<p>No new papers cleared your alert filters today.</p>'


def get_block_html(title: str, authors: str, rate, tldr: str, pdf_url: str | None,
                   affiliations: str | None = None, *, url: str = "", source: str = "",
                   matches: list[LibraryMatch] | None = None, summary_label: str = "TL;DR"):
    page = safe_url(url)
    heading = escape(title)
    if page:
        heading = f'<a href="{page}" style="color:#164b73">{heading}</a>'
    links = []
    if page:
        links.append(f'<a href="{page}">Paper page / save to Zotero</a>')
    pdf = safe_url(pdf_url)
    if pdf:
        links.append(f'<a href="{pdf}">PDF</a>')
    related = []
    for match in matches or []:
        label = escape(match.title)
        link = safe_url(match.url, zotero=True)
        if link:
            label = f'<a href="{link}">{label}</a>'
        signals = escape("; ".join(match.signals))
        related.append(f'<li>{label}<br><span style="color:#657080;font-size:12px">{signals}</span></li>')
    why = ('<p><strong>Why this paper:</strong> Similar to these papers in your Zotero:</p>'
           + '<ul>' + ''.join(related) + '</ul>') if related else ''
    affiliation = f'<br><i>{escape(affiliations)}</i>' if affiliations else ''
    return f"""<table role="presentation" width="100%" style="margin:18px 0;padding:20px;background:white;border:1px solid #dde3e8;border-radius:8px">
<tr><td>
<p style="font-size:12px;color:#657080">{escape(source)} · Relevance: {escape(str(rate))}/10</p>
<h2 style="font-size:20px;margin:10px 0">{heading}</h2>
<p style="font-size:13px;color:#657080">{escape(authors)}{affiliation}</p>
<p><strong>{escape(summary_label)}:</strong> {escape(tldr or '')}</p>
{why}
<p>{' &nbsp; · &nbsp; '.join(links)}</p>
</td></tr></table>"""


def render_email(papers: list[Paper]) -> str:
    if not papers:
        return framework.replace('__CONTENT__', get_empty_html())
    parts = [f'<p>{len(papers)} new papers matched your Zotero reading profile.</p>']
    for paper in papers:
        authors = paper.authors
        if len(authors) > 5:
            authors = authors[:3] + ['…'] + authors[-2:]
        affiliations = ', '.join(paper.affiliations[:5]) if paper.affiliations else None
        parts.append(get_block_html(
            paper.title, ', '.join(authors), f'{paper.score:.1f}' if paper.score is not None else 'Unknown',
            paper.tldr or paper.abstract, paper.pdf_url, affiliations,
            url=paper.url, source=paper.source, matches=paper.library_matches,
            summary_label="TL;DR" if paper.tldr else "Abstract",
        ))
    return framework.replace('__CONTENT__', ''.join(parts))
