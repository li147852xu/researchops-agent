from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

_MIN_PDF_BYTES = 4096


def arxiv_download_pdf(pdf_url: str, dest_dir: str) -> dict:
    """Download a PDF from arXiv, verify it's a real PDF via magic number."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    slug = hashlib.sha256(pdf_url.encode()).hexdigest()[:12]

    try:
        req = urllib.request.Request(pdf_url, headers={"User-Agent": "ResearchOps/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            http_status = resp.status
    except Exception as exc:
        return {
            "status": "failed",
            "http_status": 0,
            "bytes": 0,
            "local_path": "",
            "content_hash": "",
            "detected_type": "unknown",
            "error": str(exc),
        }

    if not data.startswith(b"%PDF-"):
        return {
            "status": "failed",
            "http_status": http_status,
            "bytes": len(data),
            "local_path": "",
            "content_hash": "",
            "detected_type": "html" if data.lstrip()[:1] == b"<" else "unknown",
            "error": "Response is not a PDF (missing %PDF- header)",
        }

    if len(data) < _MIN_PDF_BYTES:
        return {
            "status": "failed",
            "http_status": http_status,
            "bytes": len(data),
            "local_path": "",
            "content_hash": "",
            "detected_type": "pdf",
            "error": f"PDF too small ({len(data)} bytes)",
        }

    local = dest / f"{slug}.pdf"
    local.write_bytes(data)
    content_hash = hashlib.sha256(data).hexdigest()

    return {
        "status": "success",
        "http_status": http_status,
        "bytes": len(data),
        "local_path": str(local),
        "content_hash": content_hash,
        "detected_type": "pdf",
        "error": "",
    }
