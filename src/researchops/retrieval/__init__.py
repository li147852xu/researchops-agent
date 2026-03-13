from __future__ import annotations

from pathlib import Path

from researchops.retrieval.base import RetrieverBase


def create_retriever(
    mode: str,
    run_dir: Path,
    embedder_model: str = "all-MiniLM-L6-v2",
) -> RetrieverBase:
    if mode == "hybrid":
        from researchops.retrieval.bm25 import BM25Retriever
        from researchops.retrieval.embedding import EmbeddingRetriever
        from researchops.retrieval.hybrid import HybridRetriever

        bm25 = BM25Retriever(run_dir)
        emb = EmbeddingRetriever(run_dir, model_name=embedder_model)
        return HybridRetriever(bm25, emb)
    if mode == "bm25":
        from researchops.retrieval.bm25 import BM25Retriever
        return BM25Retriever(run_dir)
    from researchops.retrieval.base import NullRetriever
    return NullRetriever()
