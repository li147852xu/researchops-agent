from __future__ import annotations

import json
import re
from pathlib import Path

from researchops.retrieval.base import RetrieverBase


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w{2,}", text.lower())


class BM25Retriever(RetrieverBase):
    def __init__(self, run_dir: Path):
        self._run_dir = run_dir
        self._claims: list[dict] = []
        self._bm25 = None
        self._index_path = run_dir / "retrieval_index.json"

    def index(self, claims: list[dict]) -> None:
        self._claims = claims
        if not claims:
            return
        corpus = [_tokenize(c.get("text", "")) for c in claims]
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(corpus)
        except ImportError:
            self._bm25 = None
        self._save()

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        if not self._claims:
            return []
        if self._bm25 is None:
            return self._claims[:top_k]
        tokens = _tokenize(query)
        if not tokens:
            return self._claims[:top_k]
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._claims[i] for i in ranked[:top_k]]

    def retrieve_for_rq(self, rq_id: str, top_k: int = 10) -> list[dict]:
        filtered = [c for c in self._claims if rq_id in c.get("supports_rq", [])]
        if not filtered or self._bm25 is None:
            return filtered[:top_k]
        return filtered[:top_k]

    def _save(self) -> None:
        self._index_path.write_text(
            json.dumps({"claims_count": len(self._claims)}, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        if self._index_path.exists():
            pass
