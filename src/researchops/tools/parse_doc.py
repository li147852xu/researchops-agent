from __future__ import annotations

import re
from pathlib import Path

NOISE_PATTERNS: list[str] = [
    "skip to content",
    "table of contents",
    "sign up",
    "subscribe",
    "newsletter",
    "share this",
    "cookie",
    "privacy policy",
    "terms of service",
    "related articles",
    "read more",
    "click here",
    "all rights reserved",
    "powered by",
    "follow us",
    "log in",
    "sign in",
    "leave a comment",
    "comments",
    "share on",
    "tweet",
    "pin it",
    "by any measure",
    "meta charset",
    "meta name",
    "meta http",
    "viewport",
    "doctype html",
    "<!doctype",
    "content-type",
    "text/css",
    "text/javascript",
    "stylesheet",
    "rel stylesheet",
    "font-family",
    "font-size",
    "display swap",
    "link rel",
    "xml type",
    "robots content",
    "max-image-preview",
    "max-snippet",
    "og:title",
    "og:description",
    "og:image",
    "twitter:card",
    "twitter:site",
    "twitter:title",
]

_NOISE_CLASS_ID_KEYWORDS = [
    "nav", "menu", "sidebar", "footer", "header", "breadcrumb", "share",
    "subscribe", "cookie", "banner", "modal", "popup", "advert", "ads",
    "toc", "table-of-contents", "social", "comment", "related", "newsletter",
    "widget", "promo",
]

_MENU_CHARS = set("|/·•►▸–—")

_DECOMPOSE_TAGS = [
    "head", "script", "style", "noscript", "svg", "canvas", "footer", "header",
    "nav", "aside", "form", "button", "input", "iframe", "figcaption",
    "meta", "link",
]


_CODE_LINE_RE = re.compile(
    r"[{};]|function\s*\(|var\s|const\s|let\s|padding[\s:]+|margin[\s:]+|"
    r"\d+px|rgb\(|rgba\(|#[0-9a-fA-F]{3,8}\b|querySelector|addEventListener|"
    r"@media|@import|\.className|\.style\.",
    re.IGNORECASE,
)

_MIN_QUALITY_CHARS = 500


def _quality_gate(text: str, title: str) -> dict:
    if len(text.strip()) < _MIN_QUALITY_CHARS:
        return {
            "text": "",
            "title": title,
            "warning": "low_quality",
            "raw_len": len(text.strip()),
        }
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return {"text": "", "title": title, "warning": "low_quality", "raw_len": 0}
    code_count = sum(1 for ln in lines if _CODE_LINE_RE.search(ln))
    density = code_count / len(lines)
    if density > 0.30:
        return {
            "text": "",
            "title": title,
            "warning": "code_heavy",
            "code_density": round(density, 3),
        }
    return {"text": text, "title": title}


def parse_doc(file_path: str, format: str = "auto") -> dict:
    """Extract text from HTML, PDF, or plain text files."""
    p = Path(file_path)
    if not p.exists():
        return {"text": "", "title": "", "error": f"File not found: {file_path}"}

    if format == "pdf" or (format == "auto" and p.suffix.lower() == ".pdf"):
        head = p.read_bytes()[:8]
        if not head.startswith(b"%PDF-"):
            return {
                "text": "",
                "title": "",
                "warning": "parse_rejected",
                "error": "File claims to be PDF but lacks %PDF- header",
            }

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

    if result.get("error"):
        return result

    gate = _quality_gate(result.get("text", ""), result.get("title", ""))
    if gate.get("warning"):
        return gate
    return result


def _parse_html(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        for tag_name in _DECOMPOSE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        _remove_noise_elements(soup)

        text = _extract_main_content(soup)
        text = _clean_lines(text, title=title)

    except ImportError:
        title_m = re.search(r"<title>(.*?)</title>", raw, re.I | re.S)
        title = title_m.group(1).strip() if title_m else ""
        text = re.sub(r"<[^>]+>", " ", raw)
        text = _clean_lines(text, title=title)

    return {"text": text, "title": title}


def _remove_noise_elements(soup) -> None:  # type: ignore[no-untyped-def]
    for el in soup.find_all(True):
        classes = " ".join(el.get("class", []))
        el_id = el.get("id", "") or ""
        combined = f"{classes} {el_id}".lower()
        if any(kw in combined for kw in _NOISE_CLASS_ID_KEYWORDS):
            el.decompose()


def _extract_main_content(soup) -> str:  # type: ignore[no-untyped-def]
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


_HTML_ATTR_RE = re.compile(
    r"(charset|utf-8|viewport|initial-scale|device-width|content-type|"
    r"text/html|text/css|text/javascript|font-family|font-size|"
    r"\.css|\.js\b|\.woff|\.ttf|\.eot|\.svg\b|data-|aria-|"
    r"xmlns|w3\.org|schema\.org|application/ld\+json)",
    re.IGNORECASE,
)


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
