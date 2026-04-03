"""Research source adapters — arxiv, web, fetch, parse tool implementations and registration.

These are research-specific tools. A future Quant app would register different adapters
(e.g., market data, SEC filings) while reusing the core ToolRegistry.
"""

from __future__ import annotations

import hashlib
import json as _json
import re
import time as _time
import urllib.parse
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import httpx

from researchops.core.tools.builtins import cite, sandbox_exec
from researchops.core.tools.registry import ToolRegistry
from researchops.core.tools.schema import ToolDefinition

# ── Shared HTTP client ──────────────────────────────────────────────────

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_ACADEMIC_UA = "ResearchOps/1.0 (https://github.com/researchops; mailto:bot@researchops.dev)"

_WEB_TIMEOUT = httpx.Timeout(connect=10, read=20, write=10, pool=10)
_ACADEMIC_TIMEOUT = httpx.Timeout(connect=10, read=25, write=10, pool=10)
_DOWNLOAD_TIMEOUT = httpx.Timeout(connect=10, read=40, write=10, pool=10)

_TRANSPORT = httpx.HTTPTransport(retries=2)


def _web_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": _USER_AGENT},
        timeout=_WEB_TIMEOUT,
        transport=_TRANSPORT,
        follow_redirects=True,
    )


def _academic_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": _ACADEMIC_UA, "Accept": "application/json"},
        timeout=_ACADEMIC_TIMEOUT,
        transport=_TRANSPORT,
        follow_redirects=True,
    )


# ── Noise patterns (shared with reader) ────────────────────────────────

NOISE_PATTERNS: list[str] = [
    "skip to content", "table of contents", "sign up", "subscribe",
    "newsletter", "share this", "cookie", "privacy policy",
    "terms of service", "related articles", "read more", "click here",
    "all rights reserved", "powered by", "follow us", "log in",
    "sign in", "leave a comment", "comments", "share on", "tweet",
    "pin it", "by any measure", "meta charset", "meta name",
    "meta http", "viewport", "doctype html", "<!doctype",
    "content-type", "text/css", "text/javascript", "stylesheet",
    "rel stylesheet", "font-family", "font-size", "display swap",
    "link rel", "xml type", "robots content", "max-image-preview",
    "max-snippet", "og:title", "og:description", "og:image",
    "twitter:card", "twitter:site", "twitter:title",
]


# ── Web search ─────────────────────────────────────────────────────────

class _TitleExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._in_title = False
        self.title = ""

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data


def web_search(query: str, max_results: int = 5) -> list[dict]:
    try:
        encoded = urllib.parse.urlencode({"q": query})
        url = f"https://html.duckduckgo.com/html/?{encoded}"
        with _web_client() as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception:
        return []

    results = []
    for match in re.finditer(
        r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        html, re.DOTALL,
    ):
        href = match.group(1)
        title_raw = re.sub(r"<[^>]+>", "", match.group(2)).strip()
        if href.startswith("//duckduckgo.com/l/"):
            qs = urllib.parse.urlparse(href).query
            actual = urllib.parse.parse_qs(qs).get("uddg", [href])[0]
            href = actual
        results.append({"url": href, "title": title_raw})
        if len(results) >= max_results:
            break
    return results


# ── Fetch page ─────────────────────────────────────────────────────────

_MIN_CONTENT_BYTES = 1024


def _detect_type(data: bytes) -> str:
    if data[:5] == b"%PDF-":
        return "pdf"
    head = data[:512].lstrip()
    if head[:1] == b"<" or b"<html" in head[:256].lower() or b"<!doctype" in head[:256].lower():
        return "html"
    return "unknown"


def fetch_page(url: str, dest_dir: str) -> dict:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    slug = hashlib.sha256(url.encode()).hexdigest()[:12]

    try:
        with _web_client() as client:
            resp = client.get(url, timeout=_DOWNLOAD_TIMEOUT)
            data = resp.content
            http_status = resp.status_code
            content_type = resp.headers.get("Content-Type", "")
    except Exception as exc:
        return {
            "status": "failed", "http_status": 0, "content_type": "",
            "bytes": 0, "local_path": "", "content_hash": "",
            "detected_type": "unknown", "error": str(exc),
        }

    detected = _detect_type(data)
    url_lower = url.lower().rstrip("/")
    url_looks_pdf = url_lower.endswith(".pdf") or ".pdf?" in url_lower
    if url_looks_pdf and detected == "html":
        return {
            "status": "failed", "http_status": http_status,
            "content_type": content_type, "bytes": len(data),
            "local_path": "", "content_hash": "", "detected_type": detected,
            "error": "URL indicates PDF but content is HTML (likely captcha/redirect)",
        }

    if len(data) < _MIN_CONTENT_BYTES:
        return {
            "status": "failed", "http_status": http_status,
            "content_type": content_type, "bytes": len(data),
            "local_path": "", "content_hash": "", "detected_type": detected,
            "error": f"Content too small ({len(data)} bytes < {_MIN_CONTENT_BYTES})",
        }

    suffix_map = {"pdf": ".pdf", "html": ".html"}
    suffix = suffix_map.get(detected, ".bin")
    local = dest / f"{slug}{suffix}"
    local.write_bytes(data)
    content_hash = hashlib.sha256(data).hexdigest()

    return {
        "status": "success", "http_status": http_status,
        "content_type": content_type, "bytes": len(data),
        "local_path": str(local), "content_hash": content_hash,
        "detected_type": detected, "error": "",
    }


# ── arXiv search ───────────────────────────────────────────────────────

_ARXIV_API = "http://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_ARXIV_REQUEST_DELAY = 0.5


def arxiv_search(query: str, max_results: int = 10) -> list[dict]:
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0, "max_results": max_results,
        "sortBy": "relevance", "sortOrder": "descending",
    })
    url = f"{_ARXIV_API}?{params}"

    _time.sleep(_ARXIV_REQUEST_DELAY)
    last_exc = ""
    for attempt in range(3):
        try:
            with _academic_client() as client:
                resp = client.get(url)
                if resp.status_code == 429:
                    raise httpx.HTTPStatusError("429", request=resp.request, response=resp)
                resp.raise_for_status()
                data = resp.content
            break
        except httpx.HTTPStatusError as exc:
            last_exc = str(exc)
            if "429" in last_exc:
                _time.sleep(2 ** attempt + 1)
                continue
            return [{"error": last_exc}]
        except Exception as exc:
            return [{"error": str(exc)}]
    else:
        return [{"error": f"arXiv rate-limited after 3 attempts: {last_exc}"}]

    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return []

    results = []
    for entry in root.findall("atom:entry", _NS):
        arxiv_id = _text(entry, "atom:id", _NS).split("/abs/")[-1]
        title = _text(entry, "atom:title", _NS).replace("\n", " ").strip()
        published = _text(entry, "atom:published", _NS)[:10]
        authors = [_text(a, "atom:name", _NS) for a in entry.findall("atom:author", _NS) if _text(a, "atom:name", _NS)]
        categories = [c.get("term", "") for c in entry.findall("arxiv:primary_category", _NS) if c.get("term")]
        abstract = _text(entry, "atom:summary", _NS).replace("\n", " ").strip()
        pdf_url = ""
        for link_el in entry.findall("atom:link", _NS):
            if link_el.get("title") == "pdf":
                pdf_url = link_el.get("href", "")
                break
        results.append({
            "arxiv_id": arxiv_id, "title": title, "authors": authors,
            "published": published, "categories": categories,
            "abstract": abstract, "pdf_url": pdf_url,
        })
    return results


def _text(el: ET.Element, tag: str, ns: dict) -> str:
    child = el.find(tag, ns)
    return (child.text or "").strip() if child is not None else ""


# ── arXiv PDF download ─────────────────────────────────────────────────

_MIN_PDF_BYTES = 4096


def arxiv_download_pdf(pdf_url: str, dest_dir: str) -> dict:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    slug = hashlib.sha256(pdf_url.encode()).hexdigest()[:12]

    try:
        with _academic_client() as client:
            resp = client.get(pdf_url, timeout=_DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            data = resp.content
            http_status = resp.status_code
    except Exception as exc:
        return {"status": "failed", "http_status": 0, "bytes": 0, "local_path": "",
                "content_hash": "", "detected_type": "unknown", "error": str(exc)}

    if not data.startswith(b"%PDF-"):
        return {"status": "failed", "http_status": http_status, "bytes": len(data),
                "local_path": "", "content_hash": "",
                "detected_type": "html" if data.lstrip()[:1] == b"<" else "unknown",
                "error": "Response is not a PDF (missing %PDF- header)"}

    if len(data) < _MIN_PDF_BYTES:
        return {"status": "failed", "http_status": http_status, "bytes": len(data),
                "local_path": "", "content_hash": "", "detected_type": "pdf",
                "error": f"PDF too small ({len(data)} bytes)"}

    local = dest / f"{slug}.pdf"
    local.write_bytes(data)
    content_hash = hashlib.sha256(data).hexdigest()
    return {"status": "success", "http_status": http_status, "bytes": len(data),
            "local_path": str(local), "content_hash": content_hash,
            "detected_type": "pdf", "error": ""}


# ── Semantic Scholar search ────────────────────────────────────────────

_S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_REQUEST_DELAY = 1.5  # Conservative: S2 free tier is 100 req/5min


def semantic_scholar_search(query: str, max_results: int = 10) -> list[dict]:
    """Search Semantic Scholar for academic papers (broader coverage than arXiv)."""
    fields = "title,abstract,year,authors,externalIds,openAccessPdf,publicationTypes"
    params = urllib.parse.urlencode({
        "query": query,
        "fields": fields,
        "limit": min(max_results, 100),
    })
    url = f"{_S2_API}?{params}"

    _time.sleep(_S2_REQUEST_DELAY)
    last_exc = ""
    for attempt in range(3):
        try:
            with _academic_client() as client:
                resp = client.get(url)
                if resp.status_code == 429:
                    raise httpx.HTTPStatusError("429", request=resp.request, response=resp)
                resp.raise_for_status()
                data = resp.json()
            break
        except httpx.HTTPStatusError as exc:
            last_exc = str(exc)
            if "429" in last_exc:
                _time.sleep(2 ** attempt * 2)
                continue
            return [{"error": last_exc}]
        except Exception as exc:
            return [{"error": str(exc)}]
    else:
        return [{"error": f"S2 rate-limited after 3 attempts: {last_exc}"}]

    results = []
    for paper in data.get("data", []):
        ext_ids = paper.get("externalIds") or {}
        arxiv_id = ext_ids.get("ArXiv", "")
        oa = paper.get("openAccessPdf") or {}
        pdf_url = oa.get("url", "")
        year = paper.get("year")
        results.append({
            "arxiv_id": arxiv_id,
            "s2_paper_id": paper.get("paperId", ""),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", "") or "",
            "authors": [a.get("name", "") for a in (paper.get("authors") or [])],
            "published": str(year) if year else "",
            "categories": [t for t in (paper.get("publicationTypes") or []) if t],
            "pdf_url": pdf_url,
            "source": "semantic_scholar",
        })
    return results


# ── Wikipedia search & summary ─────────────────────────────────────────

_WIKI_API = "https://en.wikipedia.org/api/rest_v1"
_WIKI_ACTION_API = "https://en.wikipedia.org/w/api.php"


def wikipedia_search(query: str, max_results: int = 5) -> list[dict]:
    """Search Wikipedia and return article titles + summaries as high-quality background."""
    # Step 1: find matching titles via the action API
    # Wikimedia requires a descriptive UA for API access, not a browser-like one
    search_params = urllib.parse.urlencode({
        "action": "query", "list": "search",
        "srsearch": query, "srlimit": max_results,
        "format": "json", "utf8": "1",
    })
    search_url = f"{_WIKI_ACTION_API}?{search_params}"

    try:
        with _academic_client() as client:
            resp = client.get(search_url)
            resp.raise_for_status()
            hits = resp.json().get("query", {}).get("search", [])
    except Exception as exc:
        return [{"error": str(exc)}]

    if not hits:
        return []

    # Step 2: fetch summary for each article via the REST API
    results: list[dict] = []
    for hit in hits[:max_results]:
        title = hit.get("title", "")
        if not title:
            continue
        encoded_title = urllib.parse.quote(title.replace(" ", "_"))
        summary_url = f"{_WIKI_API}/page/summary/{encoded_title}"
        try:
            with _academic_client() as client:
                resp = client.get(summary_url)
                if resp.status_code != 200:
                    continue
                data = resp.json()
        except Exception:
            continue

        extract = data.get("extract", "")
        if not extract or len(extract) < 50:
            continue

        results.append({
            "title": data.get("title", title),
            "summary": extract,
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{encoded_title}"),
            "description": data.get("description", ""),
            "source": "wikipedia",
        })

    return results


# ── Document parser ────────────────────────────────────────────────────

_NOISE_CLASS_ID_KEYWORDS = [
    "nav", "menu", "sidebar", "footer", "header", "breadcrumb", "share",
    "subscribe", "cookie", "banner", "modal", "popup", "advert", "ads",
    "toc", "table-of-contents", "social", "comment", "related", "newsletter",
    "widget", "promo",
]
_MENU_CHARS = set("|/·•►▸–—")
_DECOMPOSE_TAGS = [
    "head", "script", "style", "noscript", "svg", "canvas", "footer", "header",
    "nav", "aside", "form", "button", "input", "iframe", "figcaption", "meta", "link",
]
_CODE_LINE_RE = re.compile(
    r"[{};]|function\s*\(|var\s|const\s|let\s|padding[\s:]+|margin[\s:]+|"
    r"\d+px|rgb\(|rgba\(|#[0-9a-fA-F]{3,8}\b|querySelector|addEventListener|"
    r"@media|@import|\.className|\.style\.",
    re.IGNORECASE,
)
_MIN_QUALITY_CHARS = 200
_HTML_ATTR_RE = re.compile(
    r"(charset|utf-8|viewport|initial-scale|device-width|content-type|"
    r"text/html|text/css|text/javascript|font-family|font-size|"
    r"\.css|\.js\b|\.woff|\.ttf|\.eot|\.svg\b|data-|aria-|"
    r"xmlns|w3\.org|schema\.org|application/ld\+json)",
    re.IGNORECASE,
)


def _compute_quality_score(text: str) -> float:
    if not text.strip():
        return 0.0
    length = len(text.strip())
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    code_count = sum(1 for ln in lines if _CODE_LINE_RE.search(ln))
    code_density = code_count / len(lines)
    length_score = min(1.0, length / 5000)
    clean_score = 1.0 - min(1.0, code_density * 2)
    return round(length_score * 0.4 + clean_score * 0.6, 3)


def _quality_gate(text: str, title: str) -> dict:
    if len(text.strip()) < _MIN_QUALITY_CHARS:
        return {"text": "", "title": title, "quality_score": 0.0, "warning": "low_quality", "raw_len": len(text.strip())}
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return {"text": "", "title": title, "quality_score": 0.0, "warning": "low_quality", "raw_len": 0}
    code_count = sum(1 for ln in lines if _CODE_LINE_RE.search(ln))
    density = code_count / len(lines)
    if density > 0.45:
        return {"text": "", "title": title, "quality_score": 0.0, "warning": "code_heavy", "code_density": round(density, 3)}
    qs = _compute_quality_score(text)
    return {"text": text, "title": title, "quality_score": qs}


def parse_doc(file_path: str, format: str = "auto") -> dict:
    p = Path(file_path)
    if not p.exists():
        return {"text": "", "title": "", "quality_score": 0.0, "error": f"File not found: {file_path}"}

    if format == "pdf" or (format == "auto" and p.suffix.lower() == ".pdf"):
        head = p.read_bytes()[:8]
        if not head.startswith(b"%PDF-"):
            return {"text": "", "title": "", "quality_score": 0.0, "warning": "parse_rejected",
                    "error": "File claims to be PDF but lacks %PDF- header"}

    fmt = format
    if fmt == "auto":
        suffix = p.suffix.lower()
        if suffix in (".html", ".htm"):
            fmt = "html"
        elif suffix == ".pdf":
            fmt = "pdf"
        else:
            fmt = "text"

    if fmt == "html":
        result = _parse_html(p)
    elif fmt == "pdf":
        result = _parse_pdf(p)
    else:
        result = _parse_text(p)

    if result is None:
        return {"text": "", "title": "", "quality_score": 0.0, "error": "parser returned None"}

    if result.get("error"):
        result.setdefault("quality_score", 0.0)
        return result
    gate = _quality_gate(result.get("text", ""), result.get("title", ""))
    if gate.get("warning"):
        return gate
    return gate


def _parse_html(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    title_m = re.search(r"<title>(.*?)</title>", raw, re.I | re.S)
    title = title_m.group(1).strip() if title_m else ""

    try:
        import trafilatura
        extracted = trafilatura.extract(raw, include_comments=False, include_tables=True)
        if extracted and len(extracted.strip()) > 150:
            text = _clean_lines(extracted, title=title)
            if len(text.strip()) > 150:
                return {"text": text, "title": title}
    except Exception:
        pass

    try:
        from readability import Document
        doc = Document(raw)
        title = title or doc.short_title()
        summary_html = doc.summary()
        if summary_html:
            summary_text = re.sub(r"<[^>]+>", " ", summary_html)
            summary_text = _clean_lines(summary_text, title=title)
            if len(summary_text.strip()) > 150:
                return {"text": summary_text, "title": title}
    except Exception:
        pass

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()
        for tag_name in _DECOMPOSE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        _remove_noise_elements(soup)
        text = _extract_main_content(soup)
        text = _clean_lines(text, title=title)
        return {"text": text, "title": title}
    except Exception:
        pass

    text = re.sub(r"<[^>]+>", " ", raw)
    text = _clean_lines(text, title=title)
    return {"text": text, "title": title}


def _remove_noise_elements(soup) -> None:
    for el in list(soup.find_all(True)):
        if el is None:
            continue
        try:
            classes = " ".join(el.get("class", []) or [])
            el_id = el.get("id", "") or ""
        except Exception:
            continue
        combined = f"{classes} {el_id}".lower()
        if any(kw in combined for kw in _NOISE_CLASS_ID_KEYWORDS):
            el.decompose()


def _extract_main_content(soup) -> str:
    for tag_name in ("article", "main"):
        candidate = soup.find(tag_name)
        if candidate:
            text = candidate.get_text(separator="\n", strip=True)
            if len(text) > 100:
                return text
    best_block = None
    best_score = 0.0
    for tag in soup.find_all(["div", "section"]):
        text = tag.get_text(separator=" ", strip=True)
        text_len = len(text)
        if text_len < 200:
            continue
        link_text_len = sum(len(a.get_text()) for a in tag.find_all("a"))
        ratio = text_len / max(1, link_text_len)
        score = text_len * ratio
        if score > best_score:
            best_score = score
            best_block = tag
    if best_block is not None:
        return best_block.get_text(separator="\n", strip=True)
    body = soup.find("body")
    if body:
        return body.get_text(separator="\n", strip=True)
    return soup.get_text(separator="\n", strip=True)


def _clean_lines(text: str, *, title: str = "") -> str:
    lines = text.split("\n")
    cleaned: list[str] = []
    seen_normalized: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        lower = stripped.lower()
        if any(pat in lower for pat in NOISE_PATTERNS):
            continue
        if _HTML_ATTR_RE.search(stripped):
            continue
        if len(stripped) <= 20 and sum(1 for c in stripped if c in _MENU_CHARS) >= 2:
            continue
        if len(stripped) <= 5 and not stripped[0].isalnum():
            continue
        if re.match(r"^[\d\s]+$", stripped):
            continue
        if title and stripped == title and len(cleaned) > 0:
            continue
        norm = re.sub(r"\s+", "", lower)
        if norm in seen_normalized:
            continue
        seen_normalized.add(norm)
        cleaned.append(stripped)

    result_lines: list[str] = []
    blank_count = 0
    for line in cleaned:
        if not line:
            blank_count += 1
            if blank_count <= 2:
                result_lines.append("")
        else:
            blank_count = 0
            result_lines.append(line)

    paragraphs: list[str] = []
    current: list[str] = []
    for line in result_lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def _parse_pdf(path: Path) -> dict:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        text = "\n\n".join(pages)
        info = reader.metadata
        title = info.title if info and info.title else ""
        text = _clean_lines(text, title=title)
        return {"text": text, "title": title}
    except ImportError:
        return {"text": "", "title": "", "error": "pypdf not installed"}
    except Exception as exc:
        return {"text": "", "title": "", "error": str(exc)}


def _parse_text(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    first_line = text.split("\n", 1)[0].strip()
    return {"text": text, "title": first_line}


# ── Tool registration ──────────────────────────────────────────────────

def register_research_tools(registry: ToolRegistry) -> None:
    """Register all research-specific + core tools into the registry."""
    registry.register(
        ToolDefinition(name="web_search", version="1.0.0", description="Search the web",
                       input_schema={"query": "str", "max_results": "int"},
                       output_schema={"results": "list[dict]"}, risk_level="medium",
                       permissions=["net"], timeout_default=15, cache_policy="session"),
        web_search,
    )
    registry.register(
        ToolDefinition(name="fetch", version="1.0.0", description="Download a URL to local storage",
                       input_schema={"url": "str", "dest_dir": "str"},
                       output_schema={"status": "str", "local_path": "str", "detected_type": "str"},
                       risk_level="medium", permissions=["net"], timeout_default=30, cache_policy="session"),
        fetch_page,
    )
    registry.register(
        ToolDefinition(name="parse", version="1.0.0", description="Parse HTML or PDF to extract text",
                       input_schema={"file_path": "str", "format": "str"},
                       output_schema={"text": "str", "title": "str", "quality_score": "float"},
                       risk_level="low", permissions=[], timeout_default=15, cache_policy="session"),
        parse_doc,
    )
    registry.register(
        ToolDefinition(name="sandbox_exec", version="1.0.0", description="Execute Python in sandbox",
                       input_schema={"script_path": "str", "timeout": "int", "allow_net": "bool"},
                       output_schema={"exit_code": "int", "stdout": "str", "stderr": "str"},
                       risk_level="high", permissions=["sandbox"], timeout_default=60, cache_policy="none"),
        sandbox_exec,
    )
    registry.register(
        ToolDefinition(name="cite", version="1.0.0", description="Map source/claim IDs to citation markers",
                       input_schema={"source_id": "str", "claim_id": "str"},
                       output_schema={"marker": "str"}, risk_level="low",
                       permissions=[], timeout_default=5, cache_policy="persistent"),
        cite,
    )
    registry.register(
        ToolDefinition(name="arxiv_search", version="1.0.0", description="Search arXiv via Atom API",
                       input_schema={"query": "str", "max_results": "int"},
                       output_schema={"results": "list[dict]"}, risk_level="medium",
                       permissions=["net"], timeout_default=25, cache_policy="session"),
        arxiv_search,
    )
    registry.register(
        ToolDefinition(name="arxiv_download_pdf", version="1.0.0", description="Download PDF from arXiv",
                       input_schema={"pdf_url": "str", "dest_dir": "str"},
                       output_schema={"status": "str", "local_path": "str", "detected_type": "str"},
                       risk_level="medium", permissions=["net"], timeout_default=40, cache_policy="session"),
        arxiv_download_pdf,
    )
    registry.register(
        ToolDefinition(
            name="semantic_scholar_search", version="1.0.0",
            description="Search Semantic Scholar for academic papers (broader than arXiv)",
            input_schema={"query": "str", "max_results": "int"},
            output_schema={"results": "list[dict]"}, risk_level="medium",
            permissions=["net"], timeout_default=25, cache_policy="session",
        ),
        semantic_scholar_search,
    )
    registry.register(
        ToolDefinition(
            name="wikipedia_search", version="1.0.0",
            description="Search Wikipedia for background knowledge summaries",
            input_schema={"query": "str", "max_results": "int"},
            output_schema={"results": "list[dict]"}, risk_level="low",
            permissions=["net"], timeout_default=20, cache_policy="session",
        ),
        wikipedia_search,
    )
