<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="assets/logo.svg" alt="logo"></a>
</p>

<h3 align="center">Zotero-arXiv-Daily</h3>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  ![Stars](https://img.shields.io/github/stars/TideDra/zotero-arxiv-daily?style=flat)
  [![GitHub Issues](https://img.shields.io/github/issues/TideDra/zotero-arxiv-daily)](https://github.com/TideDra/zotero-arxiv-daily/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/TideDra/zotero-arxiv-daily)](https://github.com/TideDra/zotero-arxiv-daily/pulls)
  [![License](https://img.shields.io/github/license/TideDra/zotero-arxiv-daily)](/LICENSE)
  [<img src="https://api.gitsponsors.com/api/badge/img?id=893025857" height="20">](https://api.gitsponsors.com/api/badge/link?p=PKMtRut1dWWuC1oFdJweyDSvJg454/GkdIx4IinvBblaX2AY4rQ7FYKAK1ZjApoiNhYEeduIEhfeZVIwoIVlvcwdJXVFD2nV2EE5j6lYXaT/RHrcsQbFl3aKe1F3hliP26OMayXOoZVDidl05wj+yg==)

</div>

---

<p align="center"> Recommend new arxiv papers of your interest daily according to your Zotero library.
    <br> 
</p>

> [!IMPORTANT]
> Please keep an eye on this repo, and merge your forked repo in time when there is any update of this upstream, in order to enjoy new features and fix found bugs.

## 🧐 About <a name = "about"></a>

> Track new scientific researches of your interest by just forking (and staring) this repo!😊

*Zotero-arXiv-Daily* finds arxiv papers that may attract you based on the context of your Zotero library, and then sends the result to your mailbox📮. It can be deployed as Github Action Workflow with **zero cost**, **no installation**, and **few configuration** of Github Action environment variables for daily **automatic** delivery.

## ✨ Features
- Totally free! All the calculation can be done in the Github Action runner locally within its quota (for public repo).
- AI-generated TL;DR for you to quickly pick up target papers.
- Affiliations of the paper are resolved and presented.
- Links to paper pages, PDFs, and matching Zotero items in the email.
- Reading-aware ranking using recent additions, synced notes/annotations, and optional reading tags or collections.
- Explainable Zotero matches, configurable relevance cutoff, and a short daily digest.
- Suppress papers already in your library or recently sent; preview locally before delivery.
- Fast deployment via fork this repo and set environment variables in the Github Action Page.
- Support LLM API for generating TL;DR of papers.
- Ignore unwanted Zotero papers using glob pattern.
- Support multiple sources of papers to retrieve:
  - arxiv
  - biorxiv
  - medrxiv

## 📷 Screenshot
![screenshot](./assets/screenshot.png)

## 🚀 Usage
### Quick Start
1. Fork (and star😘) this repo.
![fork](./assets/fork.png)

2. Set Github Action environment variables.
![secrets](./assets/secrets.png)

Below are all the secrets you need to set. They are invisible to anyone including you once they are set, for security.

| Key |Description | Example |
| :---  | :---  | :--- |
| ZOTERO_ID  | User ID of your Zotero account. **User ID is not your username, but a sequence of numbers**Get your ID from [here](https://www.zotero.org/settings/security). You can find it at the position shown in this [screenshot](https://github.com/TideDra/zotero-arxiv-daily/blob/main/assets/userid.png). | 12345678  |
| ZOTERO_KEY | An Zotero API key with read access. Get a key from [here](https://www.zotero.org/settings/security).  | AB5tZ877P2j7Sm2Mragq041H   |
| SENDER | The email account of the SMTP server that sends you email. | abc@qq.com |
| SENDER_PASSWORD | The password of the sender account. Note that it's not necessarily the password for logging in the e-mail client, but the authentication code for SMTP service. Ask your email provider for this.   | abcdefghijklmn |
| RECEIVER | The e-mail address that receives the paper list. | abc@outlook.com |
| OPENAI_API_KEY | API Key when using the API to access LLMs. You can get FREE API for using advanced open source LLMs in [SiliconFlow](https://cloud.siliconflow.cn/i/b3XhBRAm). | sk-xxx |
| OPENAI_API_BASE | API URL when using the API to access LLMs. | https://api.siliconflow.cn/v1 |
| MODEL_NAME | Model ID supported by your LLM provider. Reference it with `${oc.env:MODEL_NAME}` in `llm.generation_kwargs.model` to use the secret. | Your provider's model ID |

Then you should also set a public variable `CUSTOM_CONFIG` for your custom configuration.
![vars](./assets/repo_var.png)
![custom_config](./assets/config_var.png)
Paste the following content into the value of `CUSTOM_CONFIG` variable:
```yaml
zotero:
  user_id: ${oc.env:ZOTERO_ID}
  api_key: ${oc.env:ZOTERO_KEY}
  include_path: null

email:
  sender: ${oc.env:SENDER}
  receiver: ${oc.env:RECEIVER}
  smtp_server: smtp.qq.com
  smtp_port: 465
  sender_password: ${oc.env:SENDER_PASSWORD}

llm:
  api:
    key: ${oc.env:OPENAI_API_KEY}
    base_url: ${oc.env:OPENAI_API_BASE}
  generation_kwargs:
    model: ${oc.env:MODEL_NAME,gpt-4o-mini}

source:
  arxiv:
    category: ["cs.AI","cs.CV","cs.LG","cs.CL"]

executor:
  debug: ${oc.env:DEBUG,null}
  source: ['arxiv']
```
>[!NOTE]
> `${oc.env:XXX,yyy}` means the value of the environment variable `XXX`. If the variable is not set, the default value `yyy` will be used.

All configuration options and defaults are in [`config/base.yaml`](config/base.yaml).
Your `CUSTOM_CONFIG` only needs to contain overrides; new defaults are merged automatically.

### Make the alerts follow your reading

The profile is refreshed from Zotero on every run. Recent **date added** values matter by elapsed time,
so adding a paper today influences the next digest. Synced notes and annotations can bring an older
paper back into focus. The default explicit reading tags are `reading` and `currently-reading`
(case-insensitive); change these to match your workflow:

```yaml
zotero:
  include_path: null
  reading_tags: [reading, currently-reading]
  reading_paths: [] # For example: ["Current reading", "Current reading/**"]
  use_reading_activity: true
  exclude_tags: [] # For example: [ignore, irrelevant]

reranker:
  profile:
    half_life_days: 45
    activity_half_life_days: 30
    focus_weight: 0.7
    neighbor_count: 3

executor:
  max_paper_num: 20
  min_score: 5.0
  history_days: 90
```

- `include_path` selects collections for learning your interests. `exclude_tags` excludes items from
  that profile. Duplicate checking still uses the full fetched library.
- Tagging a paper as currently reading, or placing it in a configured reading collection, keeps it
  in focus until you remove that tag or membership. Reading collection patterns match full paths;
  use both `Reading` and `Reading/**` to cover a root collection and its descendants.
- Recent note/annotation **timestamps** are activity signals. A PDF merely being opened, an unsynced
  highlight, or a bibliographic metadata edit is not treated as reading. Imported notes can have
  recent timestamps too; use explicit tags or set `use_reading_activity: false` if that is noisy.
  The [Pyzotero documentation](https://pyzotero.readthedocs.io/en/latest/) describes the underlying
  notes and annotations API. Note and annotation bodies are not used in embeddings or summaries.
- A shorter half-life adapts faster. Lower `focus_weight` to favor enduring library interests.
  Scores are heuristics on a 0–10 scale, not probabilities, and depend on the embedding model.
  If the digest is too quiet, lower `min_score`; `0` disables the relevance cutoff.
- Each recommendation includes links to similar Zotero papers and the addition/reading signals
  behind those matches. Some email clients block `zotero://` links; the titles remain visible.
  Open the recommendation's paper page to save it using the Zotero Connector. The workflow does
  not write to your library.

Repeated recommendations are suppressed for 90 days using `.cache/alert-history.json`, which stores
hashed identities and send timestamps. DOI, arXiv ID (ignoring versions), and normalized title
matches also suppress papers already in your library. History is written only after SMTP accepts
the digest; failed sends and previews do not consume recommendations. Set `history_path: null`
to disable local history. The daily GitHub workflow pins this path and restores/saves it with
Actions Cache. Cache eviction or a crash between delivery and saving history can allow repeats.
Keep this file on persistent storage if running on your own scheduler; do not run two local sends
at once. GitHub runs for the same branch are serialized.

That's all! Now you can test the workflow by manually triggering it:
![test](./assets/test.png)

> [!NOTE]
> The Test workflow limits the current feed to 10 entries and saves an HTML/JSON preview as a `digest-preview` artifact; it does not send email or update history. If there are no new arXiv announcements, debug previews use the 10 latest papers in your configured categories. The preview cutoff defaults to `0` so you can inspect scores; set the `min_score` workflow input to your daily cutoff to test filtering. Duplicate checks still apply. The daily workflow uses only new announcements and sends when recommendations remain. Its manual `send_empty` input can send an empty digest to check delivery on a quiet day; scheduled behavior is unchanged.

Download the preview artifact after the Test workflow finishes. The daily workflow sends the real digest.

The Test workflow summarizes at most 3 papers by default; increase its `max_papers` input for a larger
preview. The `MODEL_NAME` secret must name a model supported by your configured API provider.
If you omit it, the example configuration falls back to `gpt-4o-mini`, which is not available at every provider.

By default, the main workflow runs on 22:00 UTC everyday. You can change this time by editting the workflow config `.github/workflows/main.yml`.

Workflows check out the commit from your own repository that triggered the run. Legacy `REPOSITORY`
and `REF` variables no longer redirect the daily job to upstream code, so your fork's changes are
used by both previews and daily emails.

### Local Running
Supported by [uv](https://github.com/astral-sh/uv), this workflow can easily run on your local device if uv is installed:
```bash
# set all the environment variables
# export ZOTERO_ID=xxxx
# ...
cd zotero-arxiv-daily
uv run src/zotero_arxiv_daily/main.py
```

To inspect recommendations locally without sending email or requiring LLM credentials:

```bash
uv run src/zotero_arxiv_daily/main.py executor.dry_run=true executor.generate_summaries=false
```

This still fetches your synced Zotero library and the current paper feed, and uses the configured
embedding backend. It saves `outputs/preview/digest.html` and `digest.json`. Set `executor.min_score=0`
for an initial calibration preview if too few papers appear. Remove `executor.generate_summaries=false`
to include LLM summaries. `executor.fetch_full_text=false` uses abstracts for summaries instead of PDFs.
PDFs are otherwise fetched only for selected recommendations, with an abstract fallback on failure.
Preview files contain library titles and reading signals; they are ignored by Git.

SMTP uses SSL directly on port 465 and STARTTLS on other ports, with a connection timeout. Override
`email.smtp_security` with `ssl` or `starttls` for other provider setups. `plain` and
`email.smtp_auth: false` are available for local test servers. TLS failures stop delivery.

### Tests

```bash
uv run pytest
# Optional: requires an SMTP test server, mock LLM/embedding API, internet and model downloads
uv run pytest --run-integration
```

The default suite uses deterministic embeddings and mocked services to check ranking, reading
activity, duplicate suppression, preview/send behavior, HTML escaping, and delivery failures.
Integration tests use the test endpoints configured in `tests/conftest.py`.

## 🚀 Sync with the latest version
This project is in active development. You can subscribe this repo via `Watch` so that you can be notified once we publish new release.

![Watch](./assets/subscribe_release.png)


## 📖 How it works
Each run fetches your Zotero paper metadata plus synced notes and annotations, builds a reading
profile, and retrieves candidate papers. Titles and abstracts are embedded (title-only library items
are retained). Papers you already own or were recently sent are removed before embedding candidates.

The ranker combines enduring semantic similarity with similarity weighted by current activity.
For each component, 75% comes from the strongest match and 25% from the mean of the closest three
matches. This keeps a small research topic from disappearing inside a large unrelated archive.
Addition activity decays with a 45-day half-life and starts at 0.75; note/annotation activity decays
with a 30-day half-life and starts at 1. Explicit reading tags/collections hold activity at 1. A 0.15
floor preserves older interests; the remaining 0.85 follows activity. The final score blends 70%
current focus with 30% enduring similarity and scales to 0–10. These defaults are configurable.
The displayed library matches are ranked by their blended contribution.

Only candidates above the cutoff enter the digest, up to the daily cap. Selected papers receive
optional full-text summaries; missing PDFs fall back to abstracts. The digest explains each match
using Zotero metadata, rather than asking an LLM to invent a reason for relevance.

## 📌 Limitations
- Reading activity is inferred from synced metadata; the workflow cannot observe every paper you open or read. Relevance is a heuristic and should be calibrated with previews.
- Retrieval uses the current feeds; it does not backfill papers missed during downtime. Duplicate detection can miss papers whose titles and identifiers both changed.
- High `MAX_PAPER_NUM` can lead the execution time exceed the limitation of Github Action runner (6h per execution for public repo, and 2000 mins per month for private repo). Commonly, the quota given to public repo is definitely enough for individual use. If you have special requirements, you can deploy the workflow in your own server, or use a self-hosted Github Action runner, or pay for the exceeded execution time.

## 👯‍♂️ Contribution
Any issue and PR are welcomed! But remember that **each PR should merge to the `dev` branch**.

## 📃 License
Distributed under the AGPLv3 License. See `LICENSE` for detail.

## ❤️ Acknowledgement
- [pyzotero](https://github.com/urschrei/pyzotero)
- [arxiv](https://github.com/lukasschwab/arxiv.py)
- [sentence_transformers](https://github.com/UKPLab/sentence-transformers)

## ☕ Buy Me A Coffee
If you find this project helpful, welcome to sponsor me via WeChat or via [ko-fi](https://ko-fi.com/tidedra).
![wechat_qr](assets/wechat_sponsor.JPG)


## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TideDra/zotero-arxiv-daily&type=Date)](https://star-history.com/#TideDra/zotero-arxiv-daily&Date)
