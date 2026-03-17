from paper import ArxivPaper
import math
from tqdm import tqdm
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
import smtplib
import datetime
import time
from loguru import logger
import markdown

framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """
  return block_template

def get_block_html(title:str, authors:str, rate:str,arxiv_id:str, abstract:str, pdf_url:str, code_url:str=None, affiliations:str=None):
    code = f'<a href="{code_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #5bc0de; padding: 8px 16px; border-radius: 4px; margin-left: 8px;">Code</a>' if code_url else ''
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>arXiv ID:</strong> <a href="https://arxiv.org/abs/{arxiv_id}" target="_blank">{arxiv_id}</a>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>TLDR:</strong> {abstract}
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>
            {code}
        </td>
    </tr>
</table>
"""
    return block_template.format(title=title, authors=authors,rate=rate,arxiv_id=arxiv_id, abstract=abstract, pdf_url=pdf_url, code=code, affiliations=affiliations)

def get_stars(score: float | None):
    # 没有得分的情况
    if score is None:
        return '<span style="font-size: 0.9em; color: #555;">Relevance: N/A</span>'

    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'

    # 自动适配分数范围：
    # 如果 score > 1，就当它是 0–10 区间；否则当它是 0–1 区间
    if score > 1:
        s = max(0.0, min(score, 10.0)) / 10.0   # 映射到 0–1
    else:
        s = max(0.0, min(score, 1.0))           # 已经是 0–1

    # 映射到 0–5 星
    star_value = s * 5
    full_star_num = int(star_value)
    half_star_num = 1 if (star_value - full_star_num) >= 0.5 and full_star_num < 5 else 0

    # 文本等级：High / Medium / Low
    if s >= 0.7:
        level = "High"
    elif s >= 0.4:
        level = "Medium"
    else:
        level = "Low"

    stars_html = full_star * full_star_num + half_star * half_star_num

    return (
        f'<div class="star-wrapper">{stars_html}</div>'
        f'<span style="font-size: 0.85em; color: #555;"> '
        f'({level}, score={score:.2f})</span>'
    )


def render_email(papers:list[ArxivPaper]):
    parts = []
    if len(papers) == 0:
        return framework.replace('__CONTENT__', get_empty_html())

    papers = sorted(
        papers,
        key=lambda p: (p.score is None, -(p.score or 0))
    )

    skipped_papers = []
    for p in tqdm(papers, desc='Rendering Email'):
        try:
            rate = get_stars(p.score)
            author_list = [a.name for a in p.authors]
            num_authors = len(author_list)

            if num_authors <= 5:
                authors = ', '.join(author_list)
            else:
                authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])

            try:
                paper_affiliations = p.affiliations
            except Exception as e:
                logger.warning(f"Failed to extract affiliations for {p.arxiv_id}: {e}")
                paper_affiliations = None

            if paper_affiliations is not None:
                affiliations = paper_affiliations[:5]
                affiliations = ', '.join(affiliations)
                if len(paper_affiliations) > 5:
                    affiliations += ', ...'
            else:
                affiliations = 'Unknown Affiliation'

            try:
                tldr_html = markdown.markdown(p.tldr)
            except Exception as e:
                logger.warning(f"Failed to generate TLDR for {p.arxiv_id}: {e}")
                summary_fallback = p.summary.strip().replace('\n', ' ')
                if len(summary_fallback) > 1200:
                    summary_fallback = summary_fallback[:1200].rstrip() + '...'
                tldr_html = markdown.markdown(summary_fallback or 'No summary available.')

            parts.append(
                get_block_html(
                    p.title,
                    authors,
                    rate,
                    p.arxiv_id,
                    tldr_html,
                    p.pdf_url,
                    p.code_url,
                    affiliations,
                )
            )
        except Exception as e:
            paper_id = getattr(p, 'arxiv_id', 'unknown')
            logger.exception(f"Skipping paper {paper_id} during email rendering because of an unexpected error: {e}")
            skipped_papers.append(paper_id)
        time.sleep(2)

    if not parts:
        logger.warning('All papers were skipped during email rendering. Sending an empty email instead.')
        return framework.replace('__CONTENT__', get_empty_html())

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    if skipped_papers:
        logger.warning(f"Skipped {len(skipped_papers)} paper(s) during rendering: {', '.join(skipped_papers)}")
    return framework.replace('__CONTENT__', content)

def send_email(sender:str, receiver:str, password:str,smtp_server:str,smtp_port:int, html:str,):
    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    msg = MIMEText(html, 'html', 'utf-8')
    msg['From'] = _format_addr('Github Action <%s>' % sender)
    msg['To'] = _format_addr('You <%s>' % receiver)
    today = datetime.datetime.now().strftime('%Y/%m/%d')
    msg['Subject'] = Header(f'Daily arXiv {today}', 'utf-8').encode()

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
    except Exception as e:
        logger.warning(f"Failed to use TLS. {e}")
        logger.warning(f"Try to use SSL.")
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)

    server.login(sender, password)
    server.sendmail(sender, [receiver], msg.as_string())
    server.quit()
