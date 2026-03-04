from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


def fetch_page(url: str, dest_dir: str) -> dict:
    """Download a URL to dest_dir, return local_path and content hash."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    slug = hashlib.sha256(url.encode()).hexdigest()[:12]
    suffix = ".html"
    if url.lower().endswith(".pdf"):
        suffix = ".pdf"
    elif url.lower().endswith(".txt"):
        suffix = ".txt"

    local = dest / f"{slug}{suffix}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchOps/0.1"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
    except Exception as exc:
        return {"local_path": "", "content_hash": "", "error": str(exc)}

    local.write_bytes(data)
    content_hash = hashlib.sha256(data).hexdigest()

    return {"local_path": str(local), "content_hash": content_hash}
