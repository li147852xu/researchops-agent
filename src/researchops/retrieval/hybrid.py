"""Hybrid retriever combining BM25 + Embedding via Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Any

from researchops.retrieval.base import RetrieverBase
from researchops.retrieval.bm25 import BM25Retriever
from researchops.retrieval.embedding import EmbeddingRetriever


class HybridRetriever(RetrieverBase):
    """BM25 + dense embedding retrieval fused with RRF (Reciprocal Rank Fusion).

    RRF score for a document d:  sum_over_lists( 1 / (k + rank(d)) )
    where k is a constant (default 60) that controls the influence of rank.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        embedding: EmbeddingRetriever,
        rrf_k: int = 60,
    ):
        self._bm25 = bm25
        self._embedding = embedding
        self._rrf_k = rrf_k
        self._claims: list[dict[str, Any]] = []

    def index(self, claims: list[dict[str, Any]]) -> None:
        self._claims = claims
        self._bm25.index(claims)
        self._embedding.index(claims)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        fetch_k = top_k * 3
        bm25_results = self._bm25.retrieve(query, top_k=fetch_k)
        emb_results = self._embedding.retrieve(query, top_k=fetch_k)
        return self._rrf_fuse(bm25_results, emb_results, top_k)

    def retrieve_for_rq(self, rq_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        fetch_k = top_k * 3
        bm25_results = self._bm25.retrieve_for_rq(rq_id, top_k=fetch_k)
        emb_results = self._embedding.retrieve_for_rq(rq_id, top_k=fetch_k)
        if not bm25_results and not emb_results:
            return []
        return self._rrf_fuse(bm25_results, emb_results, top_k)

    def _rrf_fuse(
        self,
        list_a: list[dict[str, Any]],
        list_b: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        item_map: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(list_a):
            cid = item.get("claim_id", "")
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self._rrf_k + rank + 1)
            item_map[cid] = item

        for rank, item in enumerate(list_b):
            cid = item.get("claim_id", "")
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self._rrf_k + rank + 1)
            if cid not in item_map:
                item_map[cid] = item

        ranked_ids = sorted(scores, key=lambda c: scores[c], reverse=True)[:top_k]
        return [item_map[cid] for cid in ranked_ids if cid in item_map]

    @property
    def embedding(self) -> EmbeddingRetriever:
        """Expose the embedding sub-retriever for direct encoding."""
        return self._embedding
