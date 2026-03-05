from __future__ import annotations

import tempfile
from pathlib import Path

SAMPLE_CLAIMS = [
    {"claim_id": "c1", "text": "Quantum computing uses qubits for parallel computation", "source_id": "s1", "supports_rq": ["rq_1"], "claim_type": "definition", "polarity": "support"},
    {"claim_id": "c2", "text": "Error correction is essential for fault-tolerant quantum systems", "source_id": "s1", "supports_rq": ["rq_1"], "claim_type": "method", "polarity": "support"},
    {"claim_id": "c3", "text": "Machine learning improves drug discovery pipelines", "source_id": "s2", "supports_rq": ["rq_2"], "claim_type": "result", "polarity": "support"},
    {"claim_id": "c4", "text": "Deep learning models require large datasets for training", "source_id": "s2", "supports_rq": ["rq_2"], "claim_type": "limitation", "polarity": "neutral"},
    {"claim_id": "c5", "text": "Quantum error rates remain a significant challenge", "source_id": "s3", "supports_rq": ["rq_1"], "claim_type": "limitation", "polarity": "oppose"},
]


def test_null_retriever_returns_claims():
    from researchops.retrieval.base import NullRetriever

    r = NullRetriever()
    r.index(SAMPLE_CLAIMS)
    results = r.retrieve("quantum computing", top_k=3)
    assert len(results) == 3


def test_null_retriever_for_rq():
    from researchops.retrieval.base import NullRetriever

    r = NullRetriever()
    r.index(SAMPLE_CLAIMS)
    results = r.retrieve_for_rq("rq_1")
    assert all("rq_1" in c["supports_rq"] for c in results)


def test_bm25_retriever_index_and_retrieve():
    from researchops.retrieval.bm25 import BM25Retriever

    with tempfile.TemporaryDirectory() as tmpdir:
        r = BM25Retriever(Path(tmpdir))
        r.index(SAMPLE_CLAIMS)

        results = r.retrieve("quantum computing qubits", top_k=3)
        assert len(results) <= 3
        assert any("quantum" in c["text"].lower() for c in results)


def test_bm25_retriever_for_rq():
    from researchops.retrieval.bm25 import BM25Retriever

    with tempfile.TemporaryDirectory() as tmpdir:
        r = BM25Retriever(Path(tmpdir))
        r.index(SAMPLE_CLAIMS)

        results = r.retrieve_for_rq("rq_2", top_k=5)
        assert all("rq_2" in c["supports_rq"] for c in results)


def test_bm25_retriever_empty():
    from researchops.retrieval.bm25 import BM25Retriever

    with tempfile.TemporaryDirectory() as tmpdir:
        r = BM25Retriever(Path(tmpdir))
        r.index([])
        results = r.retrieve("anything")
        assert results == []


def test_create_retriever_factory():
    from researchops.retrieval import create_retriever
    from researchops.retrieval.base import NullRetriever

    with tempfile.TemporaryDirectory() as tmpdir:
        r = create_retriever("none", Path(tmpdir))
        assert isinstance(r, NullRetriever)

        r2 = create_retriever("bm25", Path(tmpdir))
        assert r2 is not None
