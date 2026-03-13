from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

_MIN_CONTENT_BYTES = 1024

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _detect_type(data: bytes) -> str:
    if data[:5] == b"%PDF-":
        return "pdf"
    head = data[:512].lstrip()
    if head[:1] == b"<" or b"<html" in head[:256].lower() or b"<!doctype" in head[:256].lower():
        return "html"
    return "unknown"


def fetch_page(url: str, dest_dir: str) -> dict:
    """Download a URL to dest_dir with structured result and magic-number type detection."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    slug = hashlib.sha256(url.encode()).hexdigest()[:12]

    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
            http_status = resp.status
            content_type = resp.headers.get("Content-Type", "")
    except Exception as exc:
        return {
            "status": "failed",
            "http_status": 0,
            "content_type": "",
            "bytes": 0,
            "local_path": "",
            "content_hash": "",
            "detected_type": "unknown",
            "error": str(exc),
        }

    detected = _detect_type(data)

    url_lower = url.lower().rstrip("/")
    url_looks_pdf = url_lower.endswith(".pdf") or ".pdf?" in url_lower
    if url_looks_pdf and detected == "html":
        return {
            "status": "failed",
            "http_status": http_status,
            "content_type": content_type,
            "bytes": len(data),
            "local_path": "",
            "content_hash": "",
            "detected_type": detected,
            "error": "URL indicates PDF but content is HTML (likely captcha/redirect)",
        }

    if len(data) < _MIN_CONTENT_BYTES:
        return {
            "status": "failed",
            "http_status": http_status,
            "content_type": content_type,
            "bytes": len(data),
            "local_path": "",
            "content_hash": "",
            "detected_type": detected,
            "error": f"Content too small ({len(data)} bytes < {_MIN_CONTENT_BYTES})",
        }

    suffix_map = {"pdf": ".pdf", "html": ".html"}
    suffix = suffix_map.get(detected, ".bin")
    local = dest / f"{slug}{suffix}"
    local.write_bytes(data)
    content_hash = hashlib.sha256(data).hexdigest()

    return {
        "status": "success",
        "http_status": http_status,
        "content_type": content_type,
        "bytes": len(data),
        "local_path": str(local),
        "content_hash": content_hash,
        "detected_type": detected,
        "error": "",
    }
