"""Tests for new platform modules: retrieval enhancement, quality scoring,
market algorithms, evaluation harness, persistence."""

from __future__ import annotations

from pathlib import Path

# ── Retrieval Enhancement ──────────────────────────────────────────────

class TestRetrievalEnhancement:
    def test_expand_query_returns_original(self):
        from researchops.retrieval.enhancement import expand_query

        result = expand_query("something unique")
        assert result[0] == "something unique"

    def test_expand_query_synonym_expansion(self):
        from researchops.retrieval.enhancement import expand_query

        result = expand_query("machine learning applications")
        assert len(result) > 1
        assert any("ML" in v or "statistical" in v for v in result[1:])

    def test_decompose_query_splits_compound(self):
        from researchops.retrieval.enhancement import decompose_query

        result = decompose_query("deep learning architectures and training optimization methods")
        assert len(result) >= 2

    def test_decompose_query_preserves_simple(self):
        from researchops.retrieval.enhancement import decompose_query

        result = decompose_query("transformer architecture overview")
        assert len(result) == 1

    def test_score_source_quality_high_authority(self):
        from researchops.retrieval.enhancement import score_source_quality

        score = score_source_quality("arxiv.org", published_date="2025-01-15", content_length=3000)
        assert score.domain_authority >= 0.8
        assert score.composite > 0.6

    def test_score_source_quality_low_authority(self):
        from researchops.retrieval.enhancement import score_source_quality

        score = score_source_quality("reddit.com", content_length=200)
        assert score.domain_authority <= 0.3

    def test_calibrate_relevance_ordering(self):
        from researchops.retrieval.enhancement import calibrate_relevance

        results = [
            {"claim_id": "c1", "text": "claim one", "source_id": "s1"},
            {"claim_id": "c2", "text": "claim with $500M revenue", "source_id": "s2"},
        ]
        qualities = {"s1": 0.3, "s2": 0.9}
        calibrated = calibrate_relevance(results, source_qualities=qualities, boost_numerical=True)
        assert len(calibrated) == 2
        assert all(c.calibrated_score > 0 for c in calibrated)

    def test_calibrate_relevance_empty(self):
        from researchops.retrieval.enhancement import calibrate_relevance

        assert calibrate_relevance([]) == []


# ── Quality Scoring ────────────────────────────────────────────────────

class TestQualityScoring:
    def test_evidence_density_basic(self):
        from researchops.core.quality import compute_evidence_density

        report = (
            "## Introduction\n\n"
            "This is a test paragraph with a citation. [@src_1]\n\n"
            "This paragraph has no citation but is long enough to count as a real paragraph.\n\n"
            "## Methods\n\n"
            "Methods paragraph with two citations. [@src_1] [@src_2]\n\n"
        )
        sections = compute_evidence_density(report)
        assert len(sections) >= 2
        intro = sections[0]
        assert intro.heading == "Introduction"
        assert intro.total_citations >= 1

    def test_overall_density(self):
        from researchops.core.quality import overall_evidence_density

        report = "## Section\n\nParagraph with citation. [@src_1]\n\n"
        density = overall_evidence_density(report)
        assert 0 < density <= 1.0

    def test_find_citation_gaps(self):
        from researchops.core.quality import find_citation_gaps

        report = (
            "## Section A\n\n"
            "This is a long enough paragraph without any citations at all in this section of the report.\n\n"
            "This paragraph has a citation. [@src_1]\n\n"
        )
        gaps = find_citation_gaps(report)
        assert len(gaps) >= 1
        assert gaps[0].citation_count == 0

    def test_score_claim_confidence(self):
        from researchops.core.quality import score_claim_confidence

        claims = [
            {"claim_id": "c1", "text": "Revenue grew 45% YoY to $10B", "claim_type": "metric",
             "evidence_spans": ["45% YoY"], "supports_rq": ["rq_1"]},
            {"claim_id": "c2", "text": "The market is growing",
             "supports_rq": []},
        ]
        scores = score_claim_confidence(claims)
        assert len(scores) == 2
        assert scores[0].confidence > scores[1].confidence

    def test_detect_conflicts_polarity(self):
        from researchops.core.quality import detect_conflicts

        claims = [
            {"claim_id": "c1", "text": "Growth is increasing", "supports_rq": ["rq_1"], "polarity": "positive"},
            {"claim_id": "c2", "text": "Growth is declining", "supports_rq": ["rq_1"], "polarity": "negative"},
        ]
        conflicts = detect_conflicts(claims)
        assert len(conflicts) >= 1
        assert conflicts[0].conflict_type == "polarity_opposition"


# ── Market Algorithms ──────────────────────────────────────────────────

class TestMarketAlgorithms:
    def test_extract_numerical_claims(self):
        from researchops.apps.market.algorithms import extract_numerical_claims

        claims = [
            {"claim_id": "c1", "text": "Revenue reached $47.5B in FY2024"},
            {"claim_id": "c2", "text": "The company is growing"},
        ]
        numerical = extract_numerical_claims(claims)
        assert len(numerical) >= 1
        assert numerical[0].claim_id == "c1"
        assert len(numerical[0].values) > 0

    def test_numerical_claim_rate(self):
        from researchops.apps.market.algorithms import numerical_claim_rate

        claims = [
            {"claim_id": "c1", "text": "Revenue $10B in FY2024"},
            {"claim_id": "c2", "text": "Revenue $20B in FY2025"},
            {"claim_id": "c3", "text": "The company is growing"},
        ]
        rate = numerical_claim_rate(claims)
        assert 0 < rate <= 1.0

    def test_score_financial_freshness(self):
        from researchops.apps.market.algorithms import score_financial_freshness

        text = "In Q1 2025, the company reported strong earnings. Revenue for 2025-01-15 was up."
        result = score_financial_freshness("src_1", text)
        assert result.date_mentions > 0
        assert result.score > 0

    def test_link_ticker_mentions(self):
        from researchops.apps.market.algorithms import link_ticker_mentions

        text = "NVIDIA (NVDA) competes with AMD in the AI chip market. Microsoft Azure uses NVDA GPUs."
        mentions = link_ticker_mentions(text, target_ticker="NVDA")
        assert len(mentions) > 0
        assert mentions[0].ticker == "NVDA"
        assert mentions[0].mention_count >= 2

    def test_prioritize_sources(self):
        from researchops.apps.market.algorithms import prioritize_sources

        sources = [
            {"domain": "blogspot.com", "title": "Random blog"},
            {"domain": "reuters.com", "title": "NVDA news"},
            {"domain": "sec.gov", "title": "NVDA 10-K"},
        ]
        sorted_sources = prioritize_sources(sources, target_ticker="NVDA")
        assert sorted_sources[0]["domain"] == "sec.gov"
        assert all("priority_score" in s for s in sorted_sources)


# ── Evaluation Harness ─────────────────────────────────────────────────

class TestEvalHarness:
    def test_evaluate_demo_research_run(self):
        from researchops.core.evaluation.harness import evaluate_run

        demo_dir = Path(__file__).parent.parent / "runs" / "demo_research"
        if not demo_dir.exists():
            return
        report = evaluate_run(demo_dir, app_type="research")
        assert report.run_id == "demo_research"
        assert report.evidence_density > 0

    def test_evaluate_demo_market_run(self):
        from researchops.core.evaluation.harness import evaluate_run

        demo_dir = Path(__file__).parent.parent / "runs" / "demo_market"
        if not demo_dir.exists():
            return
        report = evaluate_run(demo_dir, app_type="market")
        assert report.run_id == "demo_market"
        assert report.evidence_density > 0


# ── Persistence ────────────────────────────────────────────────────────

class TestPersistence:
    def test_run_index_basic(self, tmp_path: Path):
        from researchops.core.persistence import RunIndex

        index = RunIndex(db_path=tmp_path / "test.db")
        index.record_run(
            run_id="test_run",
            app_type="research",
            topic="test topic",
            run_dir=str(tmp_path),
            eval_data={"citation_coverage": 0.85},
        )

        runs = index.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "test_run"
        assert runs[0]["citation_coverage"] == 0.85

    def test_run_index_get_run(self, tmp_path: Path):
        from researchops.core.persistence import RunIndex

        index = RunIndex(db_path=tmp_path / "test.db")
        index.record_run(
            run_id="test_run_2",
            app_type="market",
            topic="NVDA analysis",
            run_dir=str(tmp_path),
        )

        result = index.get_run("test_run_2")
        assert result is not None
        assert result["app_type"] == "market"

    def test_run_index_filter_by_type(self, tmp_path: Path):
        from researchops.core.persistence import RunIndex

        index = RunIndex(db_path=tmp_path / "test.db")
        index.record_run(run_id="r1", app_type="research", topic="t1", run_dir="d1")
        index.record_run(run_id="r2", app_type="market", topic="t2", run_dir="d2")

        research = index.list_runs(app_type="research")
        assert len(research) == 1
        assert research[0]["run_id"] == "r1"

    def test_record_artifact(self, tmp_path: Path):
        from researchops.core.persistence import RunIndex

        index = RunIndex(db_path=tmp_path / "test.db")
        index.record_artifact(
            run_id="test_run",
            artifact_type="plan",
            file_path="/tmp/plan.json",
            size_bytes=1024,
        )


# ── API Schemas ────────────────────────────────────────────────────────

class TestAPISchemas:
    def test_run_request(self):
        from researchops.api.schemas import RunRequest

        req = RunRequest(topic="test topic", mode="deep", sources="hybrid")
        assert req.topic == "test topic"
        assert req.llm.provider == "openai_compat"

    def test_run_request_with_extra(self):
        from researchops.api.schemas import RunRequest

        req = RunRequest(
            topic="NVDA analysis", mode="fast",
            extra={"ticker": "NVDA", "analysis_type": "competitive"},
        )
        assert req.extra["ticker"] == "NVDA"

    def test_health_response(self):
        from researchops.api.schemas import HealthResponse

        resp = HealthResponse(version="2.0.0")
        assert resp.status == "healthy"
        assert resp.version == "2.0.0"
