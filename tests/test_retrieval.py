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


def test_create_retriever_hybrid_factory():
    """Test factory creates hybrid retriever when mode='hybrid'."""
    pytest = __import__("pytest")
    try:
        from researchops.retrieval import create_retriever
        from researchops.retrieval.hybrid import HybridRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            r = create_retriever("hybrid", Path(tmpdir))
            assert isinstance(r, HybridRetriever)
    except ImportError:
        pytest.skip("sentence-transformers not installed")


def test_hybrid_rrf_fusion():
    """Test RRF fusion logic directly."""
    from researchops.retrieval.hybrid import HybridRetriever

    class FakeBM25:
        def index(self, claims): pass
        def retrieve(self, query, top_k=10):
            return [
                {"claim_id": "c1", "text": "claim 1"},
                {"claim_id": "c2", "text": "claim 2"},
                {"claim_id": "c3", "text": "claim 3"},
            ]
        def retrieve_for_rq(self, rq_id, top_k=10):
            return []

    class FakeEmbed:
        def index(self, claims): pass
        def retrieve(self, query, top_k=10):
            return [
                {"claim_id": "c2", "text": "claim 2"},
                {"claim_id": "c4", "text": "claim 4"},
                {"claim_id": "c1", "text": "claim 1"},
            ]
        def retrieve_for_rq(self, rq_id, top_k=10):
            return []

    hybrid = HybridRetriever(FakeBM25(), FakeEmbed(), rrf_k=60)
    results = hybrid.retrieve("test query", top_k=3)
    ids = [r["claim_id"] for r in results]
    # c1 and c2 appear in both lists so should have highest RRF scores
    assert "c1" in ids
    assert "c2" in ids
    assert len(results) == 3
