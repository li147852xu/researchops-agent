from __future__ import annotations

from pathlib import Path

from researchops.retrieval.base import RetrieverBase


def create_retriever(mode: str, run_dir: Path) -> RetrieverBase:
    if mode == "bm25":
        from researchops.retrieval.bm25 import BM25Retriever
        return BM25Retriever(run_dir)
    from researchops.retrieval.base import NullRetriever
    return NullRetriever()
