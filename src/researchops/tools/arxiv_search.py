from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

_ARXIV_API = "http://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def arxiv_search(query: str, max_results: int = 10) -> list[dict]:
    """Search arXiv via the Atom API and return structured paper metadata."""
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    url = f"{_ARXIV_API}?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchOps/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
    except Exception as exc:
        return [{"error": str(exc)}]

    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return []

    results = []
    for entry in root.findall("atom:entry", _NS):
        arxiv_id = _text(entry, "atom:id", _NS).split("/abs/")[-1]
        title = _text(entry, "atom:title", _NS).replace("\n", " ").strip()
        published = _text(entry, "atom:published", _NS)[:10]

        authors = []
        for author_el in entry.findall("atom:author", _NS):
            name = _text(author_el, "atom:name", _NS)
            if name:
                authors.append(name)

        categories = []
        for cat_el in entry.findall("arxiv:primary_category", _NS):
            term = cat_el.get("term", "")
            if term:
                categories.append(term)

        abstract = _text(entry, "atom:summary", _NS).replace("\n", " ").strip()

        pdf_url = ""
        for link_el in entry.findall("atom:link", _NS):
            if link_el.get("title") == "pdf":
                pdf_url = link_el.get("href", "")
                break

        results.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "published": published,
            "categories": categories,
            "abstract": abstract,
            "pdf_url": pdf_url,
        })

    return results


def _text(el: ET.Element, tag: str, ns: dict) -> str:
    child = el.find(tag, ns)
    return (child.text or "").strip() if child is not None else ""
