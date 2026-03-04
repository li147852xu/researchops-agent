from __future__ import annotations

from pathlib import Path


def parse_doc(file_path: str, format: str = "auto") -> dict:
    """Extract text from HTML, PDF, or plain text files."""
    p = Path(file_path)
    if not p.exists():
        return {"text": "", "title": "", "error": f"File not found: {file_path}"}

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
        return _parse_html(p)
    elif fmt == "pdf":
        return _parse_pdf(p)
    else:
        return _parse_text(p)


def _parse_html(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
    except ImportError:
        import re

        title_m = re.search(r"<title>(.*?)</title>", raw, re.I | re.S)
        title = title_m.group(1).strip() if title_m else ""
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
    return {"text": text, "title": title}


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
        return {"text": text, "title": title}
    except ImportError:
        return {"text": "", "title": "", "error": "pypdf not installed"}
    except Exception as exc:
        return {"text": "", "title": "", "error": str(exc)}


def _parse_text(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    first_line = text.split("\n", 1)[0].strip()
    return {"text": text, "title": first_line}
