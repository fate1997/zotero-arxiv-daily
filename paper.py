from typing import Optional, Any
from functools import cached_property
from tempfile import TemporaryDirectory
import arxiv
import tarfile
import re
import time
import ast
from llm import get_llm
import requests
from requests.adapters import HTTPAdapter
from loguru import logger
import tiktoken
from contextlib import ExitStack
from urllib.error import HTTPError, URLError
from urllib3.util.retry import Retry



class ArxivPaper:
    def __init__(self,paper:arxiv.Result):
        self._paper = paper
        self.score = None
    
    @property
    def title(self) -> str:
        return self._paper.title
    
    @property
    def summary(self) -> str:
        return self._paper.summary
    
    @property
    def authors(self) -> list[str]:
        return self._paper.authors
    
    @cached_property
    def arxiv_id(self) -> str:
        return re.sub(r'v\d+$', '', self._paper.get_short_id())
    
    @property
    def pdf_url(self) -> str:
        if self._paper.pdf_url is not None:
            return self._paper.pdf_url
        
        pdf_url = f"https://arxiv.org/pdf/{self.arxiv_id}.pdf"
        if self._paper.links is not None:
            pdf_url = self._paper.links[0].href.replace('abs','pdf')

        ## Assign pdf_url to self._paper.pdf_url for pdf downloading (Issue #119)
        self._paper.pdf_url = pdf_url

        return pdf_url
        
    _PWC_SESSION: Optional[requests.Session] = None  # 类级单例

    @staticmethod
    def _make_session() -> requests.Session:
        s = requests.Session()
        retries = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({"User-Agent": "zotero-arxiv-daily/1.0 (contact: you@example.com)"})
        return s

    @classmethod
    def _session(cls) -> requests.Session:
        if cls._PWC_SESSION is None:
            cls._PWC_SESSION = cls._make_session()
        return cls._PWC_SESSION

    @staticmethod
    def _get_json(s: requests.Session, url: str) -> Optional[dict[str, Any]]:
        try:
            r = s.get(url, timeout=(5, 15))  # (connect, read)
        except requests.RequestException as e:
            logger.debug(f"Request failed: url={url} err={e}")
            return None
    
        ct = (r.headers.get("Content-Type") or "").lower()
        if r.status_code != 200:
            logger.debug(f"Non-200: url={url} status={r.status_code} ct={ct} body={r.text[:200]!r}")
            return None
    
        if "application/json" not in ct:
            logger.debug(f"Non-JSON: url={url} status=200 ct={ct} body={r.text[:200]!r}")
            return None
    
        try:
            return r.json()
        except ValueError as e:
            logger.debug(f"JSON decode failed: url={url} status=200 ct={ct} err={e} body={r.text[:200]!r}")
            return None

    _CODE_URL_CACHE: dict[str, Optional[str]] = {}
    _SOURCE_DOWNLOAD_LAST_TS: float = 0.0

    @classmethod
    def _download_source_with_retry(cls, paper: arxiv.Result, dirpath: str, arxiv_id: str) -> str:
        """Download arXiv source with throttling and graceful retry."""
        attempts = 4
        min_interval_seconds = 3.0
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            wait_seconds = max(0.0, min_interval_seconds - (time.time() - cls._SOURCE_DOWNLOAD_LAST_TS))
            if wait_seconds > 0:
                time.sleep(wait_seconds)

            try:
                file = paper.download_source(dirpath=dirpath)
                cls._SOURCE_DOWNLOAD_LAST_TS = time.time()
                return file
            except HTTPError as e:
                cls._SOURCE_DOWNLOAD_LAST_TS = time.time()
                last_error = e
                if e.code == 404:
                    raise
                if e.code in {429, 500, 502, 503, 504}:
                    backoff = min(60, 5 * (2 ** (attempt - 1)))
                    logger.warning(
                        f"Transient HTTP error {e.code} when downloading source for {arxiv_id} "
                        f"(attempt {attempt}/{attempts}). Sleeping {backoff}s before retry."
                    )
                    if attempt < attempts:
                        time.sleep(backoff)
                        continue
                raise
            except (URLError, TimeoutError) as e:
                cls._SOURCE_DOWNLOAD_LAST_TS = time.time()
                last_error = e
                backoff = min(60, 5 * (2 ** (attempt - 1)))
                logger.warning(
                    f"Network error when downloading source for {arxiv_id} "
                    f"(attempt {attempt}/{attempts}): {e}. Sleeping {backoff}s before retry."
                )
                if attempt < attempts:
                    time.sleep(backoff)
                    continue
                raise
            except Exception as e:
                cls._SOURCE_DOWNLOAD_LAST_TS = time.time()
                last_error = e
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to download source for {arxiv_id} for an unknown reason.")

    @cached_property
    def code_url(self) -> Optional[str]:
        cache = type(self)._CODE_URL_CACHE  

        if self.arxiv_id in cache:
            return cache[self.arxiv_id]
    
        s = self._session()
        paper_list = self._get_json(s, f"https://paperswithcode.com/api/v1/papers/?arxiv_id={self.arxiv_id}")
        if not paper_list or paper_list.get("count", 0) == 0:
            cache[self.arxiv_id] = None     
            return None
    
        paper_id = paper_list["results"][0]["id"]
        repo_list = self._get_json(s, f"https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/")
        if not repo_list or repo_list.get("count", 0) == 0:
            cache[self.arxiv_id] = None    
            return None
    
        url = repo_list["results"][0].get("url")
        cache[self.arxiv_id] = url        
        return url

    
    @cached_property
    def tex(self) -> Optional[dict[str, str]]:
        with ExitStack() as stack:
            tmpdirname = stack.enter_context(TemporaryDirectory())
            try:
                file = type(self)._download_source_with_retry(
                    paper=self._paper,
                    dirpath=tmpdirname,
                    arxiv_id=self.arxiv_id,
                )
            except HTTPError as e:
                if e.code == 404:
                    logger.warning(f"Source for {self.arxiv_id} not found (404). Skipping source analysis.")
                elif e.code == 429:
                    logger.warning(
                        f"Source download for {self.arxiv_id} hit arXiv rate limits (429). "
                        "Skipping source-based analysis for this paper."
                    )
                else:
                    logger.warning(
                        f"HTTP error {e.code} when downloading source for {self.arxiv_id}. "
                        "Skipping source-based analysis for this paper."
                    )
                return None
            except Exception as e:
                logger.warning(
                    f"Error when downloading source for {self.arxiv_id}: {e}. "
                    "Skipping source-based analysis for this paper."
                )
                return None

            try:
                tar = stack.enter_context(tarfile.open(file))
            except tarfile.ReadError:
                logger.debug(f"Failed to find main tex file of {self.arxiv_id}: Not a tar file.")
                return None

            tex_files = [f for f in tar.getnames() if f.endswith('.tex')]
            if len(tex_files) == 0:
                logger.debug(f"Failed to find main tex file of {self.arxiv_id}: No tex file.")
                return None

            bbl_file = [f for f in tar.getnames() if f.endswith('.bbl')]
            match len(bbl_file):
                case 0:
                    if len(tex_files) > 1:
                        logger.debug(
                            f"Cannot find main tex file of {self.arxiv_id} from bbl: "
                            "There are multiple tex files while no bbl file."
                        )
                        main_tex = None
                    else:
                        main_tex = tex_files[0]
                case 1:
                    main_name = bbl_file[0].replace('.bbl', '')
                    main_tex = f"{main_name}.tex"
                    if main_tex not in tex_files:
                        logger.debug(
                            f"Cannot find main tex file of {self.arxiv_id} from bbl: "
                            "The bbl file does not match any tex file."
                        )
                        main_tex = None
                case _:
                    logger.debug(
                        f"Cannot find main tex file of {self.arxiv_id} from bbl: There are multiple bbl files."
                    )
                    main_tex = None

            if main_tex is None:
                logger.debug(f"Trying to choose tex file containing the document block as main tex file of {self.arxiv_id}")

            file_contents: dict[str, str] = {}
            for t in tex_files:
                f = tar.extractfile(t)
                if f is None:
                    continue
                content = f.read().decode('utf-8', errors='ignore')
                content = re.sub(r'%.*\n', '\n', content)
                content = re.sub(r'\\begin{comment}.*?\\end{comment}', '', content, flags=re.DOTALL)
                content = re.sub(r'\\iffalse.*?\\fi', '', content, flags=re.DOTALL)
                content = re.sub(r'\n+', '\n', content)
                content = re.sub(r'\\\\', '', content)
                content = re.sub(r'[ \t\r\f]{3,}', ' ', content)
                if main_tex is None and re.search(r'\\begin\{document\}', content):
                    main_tex = t
                    logger.debug(f"Choose {t} as main tex file of {self.arxiv_id}")
                file_contents[t] = content

            if main_tex is not None:
                main_source: str = file_contents[main_tex]
                include_files = re.findall(r'\\input\{(.+?)\}', main_source) + re.findall(r'\\include\{(.+?)\}', main_source)
                for f in include_files:
                    file_name = f if f.endswith('.tex') else f + '.tex'
                    main_source = main_source.replace(f'\\input{{{f}}}', file_contents.get(file_name, ''))
                    main_source = main_source.replace(f'\\include{{{f}}}', file_contents.get(file_name, ''))
                file_contents["all"] = main_source
                return file_contents

            logger.debug(
                f"Failed to find main tex file of {self.arxiv_id}: No tex file containing the document block."
            )
            return None

    @cached_property
    def tldr(self) -> str:
        introduction = ""
        conclusion = ""
        if self.tex is not None:
            content = self.tex.get("all")
            if content is None:
                content = "\n".join(self.tex.values())
            #remove cite
            content = re.sub(r'~?\\cite.?\{.*?\}', '', content)
            #remove figure
            content = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', content, flags=re.DOTALL)
            #remove table
            content = re.sub(r'\\begin\{table\}.*?\\end\{table\}', '', content, flags=re.DOTALL)
            #find introduction and conclusion
            # end word can be \section or \end{document} or \bibliography or \appendix
            match = re.search(r'\\section\{Introduction\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', content, flags=re.DOTALL)
            if match:
                introduction = match.group(0)
            match = re.search(r'\\section\{Conclusion\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', content, flags=re.DOTALL)
            if match:
                conclusion = match.group(0)
        llm = get_llm()
        prompt = """
You are an expert research assistant specializing in:
DFT and electronic-structure methods,
molecular simulation (MD, MC, HPMC, coarse-grained models),
crystal structure prediction (CSP) and polymorph modelling,
symmetry and unit-cell detection,
machine learning for molecular and materials modelling,
solvent effects on crystals and crystal energies,
and porous / molecular materials.

You are given the TITLE, ABSTRACT, INTRODUCTION and CONCLUSION (if any) of a paper
in LaTeX-like format. Using ONLY the information provided (you may infer method
*type* at a high level, but must not fabricate specific algorithmic details),
produce a precise, structured summary in __LANG__.

Follow this structure exactly:

**TL;DR (2–3 sentences):**
   A concise description of the main idea, method category, and contribution.

**Research Problem:**
   What problem the authors aim to solve and why it matters.

**Method (inferred if necessary):**
   Classify the method into one or more of the following categories and add a short explanation:
   - DFT / electronic-structure methods
   - ab initio MD / AIMD
   - classical MD / MC / HPMC / statistical mechanics
   - coarse-grained or multiscale modelling
   - CSP / structure search / polymorph prediction
   - ML model (GNN, transformer, diffusion, force field, property predictor)
   - symmetry / space-group / unit-cell analysis
   - solvent or solvation-energy modelling
   - other (specify briefly)

**Conclusion (1–2 sentences):**
   Summarize what the authors claim to have achieved or demonstrated. If the text is vague,
   give a high-level conclusion and say that details are unclear from the abstract.

**Key Contributions (bullet points):**
   - What is new or original?
   - What improves over existing work (accuracy, efficiency, scalability, robustness, etc.)?
   - Any new datasets, benchmarks, or software, if mentioned.

**Relevance to My Research (High / Medium / Low):**
   Rate relevance based on these areas:
   DFT, CSP, MD/MC or HPMC simulation, coarse-grained modelling, solvent effects,
   symmetry and unit-cell tools, porous / molecular materials, or ML for molecular simulation.
   Give ONE short sentence explaining the rating.

**Potential Impact (1 sentence):**
   State what this work enables or improves in practice.

Now read the following LaTeX content and produce the structured summary in __LANG__:

\\title{__TITLE__}
\\begin{abstract}
__ABSTRACT__
\\end{abstract}

__INTRODUCTION__
__CONCLUSION__
        """
        prompt = prompt.replace('__LANG__', llm.lang)
        prompt = prompt.replace('__TITLE__', self.title)
        prompt = prompt.replace('__ABSTRACT__', self.summary)
        prompt = prompt.replace('__INTRODUCTION__', introduction)
        prompt = prompt.replace('__CONCLUSION__', conclusion)

        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)
        
        tldr = llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert assistant for computational chemistry and materials modelling papers (DFT, CSP, MD/MC, coarse-grained models, symmetry, solvent effects, ML for materials). You summarise papers accurately and never invent unsupported details.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        return tldr

    @cached_property
    def affiliations(self) -> Optional[list[str]]:
        tex = self.tex
        if not tex:  # None 或 空 dict
            return None
    
        content = tex.get("all")
        if not content:
            # 如果我们现在 tex() 在找不到 main tex 时直接返回 None，
            # 理论上不会走到这里；这里当作额外保险
            logger.debug(f"No main tex content for {self.arxiv_id} when extracting affiliations.")
            return None
        #search for affiliations
        possible_regions = [r'\\author.*?\\maketitle',r'\\begin{document}.*?\\begin{abstract}']
        matches = [re.search(p, content, flags=re.DOTALL) for p in possible_regions]
        match = next((m for m in matches if m), None)
        if match:
            information_region = match.group(0)
        else:
            logger.debug(f"Failed to extract affiliations of {self.arxiv_id}: No author information found.")
            return None
        prompt = f"Given the author information of a paper in latex format, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]'. Following is the author information:\n{information_region}"
        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)
        llm = get_llm()
        try:
            affiliations = llm.generate(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from the author information of a paper. You should return a python list of affiliations sorted by the author order, like ['TsingHua University','Peking University']. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as e:
            logger.debug(f"Failed to extract affiliations of {self.arxiv_id}: LLM call failed: {e}")
            return None

        try:
            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = ast.literal_eval(affiliations)
            seen = set()
            normalized_affiliations = []
            for affiliation in affiliations:
                affiliation = str(affiliation).strip()
                if not affiliation or affiliation in seen:
                    continue
                seen.add(affiliation)
                normalized_affiliations.append(affiliation)
        except Exception as e:
            logger.debug(f"Failed to extract affiliations of {self.arxiv_id}: {e}")
            return None
        return normalized_affiliations
