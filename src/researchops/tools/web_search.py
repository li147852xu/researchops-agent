from __future__ import annotations

import urllib.parse
import urllib.request
from html.parser import HTMLParser


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
    """Lightweight web search via DuckDuckGo HTML.

    Falls back gracefully: returns empty list if network unavailable.
    """
    try:
        encoded = urllib.parse.urlencode({"q": query})
        url = f"https://html.duckduckgo.com/html/?{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchOps/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []

    results = []
    import re

    for match in re.finditer(
        r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
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
