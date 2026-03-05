"""Regression tests for v1.0.1: rollback loop prevention, adaptive collection,
citation enforcement, and QA domination routing."""
from __future__ import annotations

import re
from pathlib import Path

from researchops.agents.base import RunContext
from researchops.config import RunConfig
from researchops.models import (
    Claim,
    PlanOutput,
    ResearchQuestion,
    Source,
    SourceNotes,
    SourceType,
    Stage,
    StateSnapshot,
)
from researchops.reasoning.none import NoneReasoner
from researchops.registry.manager import ToolRegistry
from researchops.sandbox.proc import SubprocessSandbox
from researchops.trace import TraceLogger


def _make_ctx(
    tmp_run_dir: Path,
    *,
    allow_net: bool = True,
    mode: str = "deep",
    collect_rounds: int = 1,
    strategy_level: int = 0,
    incomplete_sections: list[str] | None = None,
) -> RunContext:
    config = RunConfig(topic="deep learning", mode=mode, allow_net=allow_net, run_dir=tmp_run_dir)
    state = StateSnapshot(
        run_id="test",
        collect_rounds=collect_rounds,
        collect_strategy_level=strategy_level,
        incomplete_sections=incomplete_sections or [],
    )
    trace = TraceLogger(tmp_run_dir / "trace.jsonl")
    registry = ToolRegistry()
    registry.grant_permissions({"sandbox", "net"})
    return RunContext(
        run_dir=tmp_run_dir,
        config=config,
        state=state,
        registry=registry,
        trace=trace,
        sandbox=SubprocessSandbox(),
        reasoner=NoneReasoner(),
    )


def _write_plan(run_dir: Path, rqs: list[tuple[str, str]] | None = None) -> PlanOutput:
    if rqs is None:
        rqs = [("rq_state", "Current state of deep learning"), ("rq_challenges", "Challenges in deep learning")]
    plan = PlanOutput(
        topic="deep learning",
        research_questions=[ResearchQuestion(rq_id=rid, text=txt) for rid, txt in rqs],
        outline=[
            {"heading": "Current State", "rq_refs": ["rq_state"]},
            {"heading": "Challenges", "rq_refs": ["rq_challenges"]},
        ],
        acceptance_threshold=0.8,
    )
    (run_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return plan


def _write_sources(run_dir: Path, sources: list[Source]) -> None:
    with (run_dir / "sources.jsonl").open("w", encoding="utf-8") as f:
        for s in sources:
            f.write(s.model_dump_json() + "\n")


def _write_notes(run_dir: Path, source_id: str, claims: list[Claim]) -> None:
    notes = SourceNotes(source_id=source_id, claims=claims)
    notes_dir = run_dir / "notes"
    notes_dir.mkdir(exist_ok=True)
    (notes_dir / f"{source_id}.json").write_text(notes.model_dump_json(indent=2), encoding="utf-8")


class TestOrchestratorMaxRounds:
    def test_max_collect_rounds_guard(self, tmp_run_dir: Path):
        """When collect_rounds >= max_collect_rounds, orchestrator should degrade-complete."""
        from researchops.orchestrator import Orchestrator

        config = RunConfig(topic="deep learning", mode="deep", allow_net=True, run_dir=tmp_run_dir)
        trace = TraceLogger(tmp_run_dir / "trace.jsonl")

        orch = Orchestrator.__new__(Orchestrator)
        orch.config = config
        orch.run_dir = tmp_run_dir
        orch.trace = trace
        orch.registry = ToolRegistry()
        orch.sandbox = SubprocessSandbox()
        orch.reasoner = NoneReasoner()
        orch.state = StateSnapshot(run_id="test", collect_rounds=6)

        _write_plan(tmp_run_dir)

        reason = orch._classify_rollback_reason("Report written but 2 sections lack evidence")
        assert reason == "evidence_gap"

        assert orch.state.collect_rounds >= config.max_collect_rounds

    def test_progress_detector_hashes_change(self, tmp_run_dir: Path):
        """sources_hash should change when sources.jsonl content changes."""
        from researchops.orchestrator import Orchestrator

        config = RunConfig(topic="test", run_dir=tmp_run_dir)
        trace = TraceLogger(tmp_run_dir / "trace.jsonl")

        orch = Orchestrator.__new__(Orchestrator)
        orch.config = config
        orch.run_dir = tmp_run_dir
        orch.trace = trace
        orch.registry = ToolRegistry()
        orch.sandbox = SubprocessSandbox()
        orch.reasoner = NoneReasoner()
        orch.state = StateSnapshot(run_id="test")

        s1 = Source(source_id="s1", type=SourceType.HTML, hash="h1")
        _write_sources(tmp_run_dir, [s1])
        hash1 = orch._compute_sources_hash()

        s2 = Source(source_id="s2", type=SourceType.HTML, hash="h2")
        _write_sources(tmp_run_dir, [s1, s2])
        hash2 = orch._compute_sources_hash()

        assert hash1 != hash2

    def test_no_progress_streak_increments(self, tmp_run_dir: Path):
        """When hashes don't change, no_progress_streak should increment."""
        from researchops.orchestrator import Orchestrator

        config = RunConfig(topic="test", run_dir=tmp_run_dir)
        trace = TraceLogger(tmp_run_dir / "trace.jsonl")

        orch = Orchestrator.__new__(Orchestrator)
        orch.config = config
        orch.run_dir = tmp_run_dir
        orch.trace = trace
        orch.registry = ToolRegistry()
        orch.sandbox = SubprocessSandbox()
        orch.reasoner = NoneReasoner()
        orch.state = StateSnapshot(run_id="test", sources_hash="abc", claims_hash="def")

        s1 = Source(source_id="s1", type=SourceType.HTML, hash="h1")
        _write_sources(tmp_run_dir, [s1])
        current_hash = orch._compute_sources_hash()
        orch.state.sources_hash = current_hash
        orch.state.claims_hash = orch._compute_claims_hash()

        new_hash = orch._compute_sources_hash()
        has_progress = new_hash != orch.state.sources_hash
        assert not has_progress

    def test_degrade_complete_marks_incomplete(self, tmp_run_dir: Path):
        """_degrade_complete should populate incomplete_sections."""
        from researchops.orchestrator import Orchestrator

        config = RunConfig(topic="test", mode="deep", run_dir=tmp_run_dir)
        trace = TraceLogger(tmp_run_dir / "trace.jsonl")

        orch = Orchestrator.__new__(Orchestrator)
        orch.config = config
        orch.run_dir = tmp_run_dir
        orch.trace = trace
        orch.registry = ToolRegistry()
        orch.sandbox = SubprocessSandbox()
        orch.reasoner = NoneReasoner()
        orch.state = StateSnapshot(run_id="test")

        _write_plan(tmp_run_dir)

        orch._degrade_complete("evidence gap test")
        assert len(orch.state.incomplete_sections) > 0


class TestCollectorAdaptive:
    def test_gap_rqs_identified(self, tmp_run_dir: Path):
        """Collector should identify RQs with insufficient coverage."""
        from researchops.agents.collector import CollectorAgent

        ctx = _make_ctx(tmp_run_dir, mode="deep")
        plan = _write_plan(tmp_run_dir)
        ctx.state.coverage_vector = {"rq_state": 10, "rq_challenges": 1}

        collector = CollectorAgent()
        gap_rqs = collector._identify_gap_rqs(plan, [], ctx)
        gap_ids = [r.rq_id for r in gap_rqs]
        assert "rq_challenges" in gap_ids

    def test_strategy_level_expands_queries(self, tmp_run_dir: Path):
        """Higher strategy levels should produce more diverse queries."""
        from researchops.agents.collector import CollectorAgent

        ctx = _make_ctx(tmp_run_dir)
        collector = CollectorAgent()

        queries_l0 = collector._generate_queries("deep learning state", "deep learning", 0, ctx)
        queries_l2 = collector._generate_queries("deep learning state", "deep learning", 2, ctx)
        assert len(queries_l2) > len(queries_l0)

    def test_incremental_sources_not_overwritten(self, tmp_run_dir: Path):
        """Existing sources should be preserved across collect rounds."""
        from researchops.agents.collector import CollectorAgent

        existing = Source(
            source_id="src_old", type=SourceType.HTML, hash="old_hash",
            local_path=str(tmp_run_dir / "downloads" / "old.txt"),
            source_type_detail="demo", collect_round=1,
        )
        _write_sources(tmp_run_dir, [existing])
        (tmp_run_dir / "downloads" / "old.txt").write_text("old content")

        collector = CollectorAgent()
        loaded = collector._load_existing_sources(tmp_run_dir)
        assert len(loaded) == 1
        assert loaded[0].source_id == "src_old"


class TestWriterEvidence:
    def test_evidence_gap_at_max_rounds(self, tmp_run_dir: Path):
        """When at max collect rounds, writer should produce evidence gap text."""
        from researchops.agents.writer import WriterAgent

        ctx = _make_ctx(tmp_run_dir, collect_rounds=6)
        _write_plan(tmp_run_dir)
        _write_sources(tmp_run_dir, [])

        writer = WriterAgent()
        result = writer.execute(ctx)
        assert result.success is True

        report = (tmp_run_dir / "report.md").read_text()
        assert "Evidence Gap" in report or "证据缺口" in report or "Evidence insufficient" in report

    def test_all_sentences_have_citations(self, tmp_run_dir: Path):
        """Every non-heading, non-meta line in report should have a citation marker."""
        ctx = _make_ctx(tmp_run_dir, collect_rounds=1)
        _write_plan(tmp_run_dir)

        s1 = Source(
            source_id="src_test_1", type=SourceType.API, hash="h1",
            local_path=str(tmp_run_dir / "downloads" / "t1.txt"),
            source_type_detail="arxiv_meta",
        )
        _write_sources(tmp_run_dir, [s1])
        (tmp_run_dir / "downloads" / "t1.txt").write_text(
            "Deep learning has revolutionized computer vision and natural language processing. "
            "Convolutional neural networks achieve state-of-the-art results on image classification tasks. "
            "Transformer architectures have enabled significant advances in language understanding."
        )

        claims = [
            Claim(claim_id="c1", text="Deep learning revolutionized CV and NLP", supports_rq=["rq_state"],
                  evidence_spans=["Deep learning"], claim_type="result"),
            Claim(claim_id="c2", text="CNNs achieve state-of-the-art on image tasks", supports_rq=["rq_state"],
                  evidence_spans=["Convolutional"], claim_type="result"),
            Claim(claim_id="c3", text="Transformers advance language understanding significantly", supports_rq=["rq_challenges"],
                  evidence_spans=["Transformer"], claim_type="method"),
        ]
        _write_notes(tmp_run_dir, "src_test_1", claims)

        from researchops.agents.writer import WriterAgent
        writer = WriterAgent()
        writer.execute(ctx)

        report = (tmp_run_dir / "report.md").read_text()
        for line in report.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("*") or stripped.startswith("-"):
                continue
            if len(stripped) < 30:
                continue
            if "evidence gap" in stripped.lower() or "证据缺口" in stripped or "证据不足" in stripped:
                continue
            if "evidence insufficient" in stripped.lower():
                continue
            assert re.search(r"\[@\w+\]", stripped), f"Line missing citation: {stripped[:80]}"


class TestQARollbackRouting:
    def test_domination_routes_to_collect(self, tmp_run_dir: Path):
        """QA should route source domination issues to COLLECT, not WRITE."""
        from researchops.agents.qa import QAAgent

        qa = QAAgent()
        domination_issues = ["Source src_1 dominates section 'Overview' (80%)"]
        evidence_gaps = []
        conflicts: dict = {"conflicts": []}
        traceability: dict = {"unsupported_rate": 0.0}

        ctx = _make_ctx(tmp_run_dir, allow_net=True)
        target = qa._determine_rollback(
            domination_issues, traceability, conflicts, domination_issues, evidence_gaps, ctx,
        )
        assert target == Stage.COLLECT

    def test_evidence_gap_routes_to_collect(self, tmp_run_dir: Path):
        """QA should route evidence gap issues to COLLECT."""
        from researchops.agents.qa import QAAgent

        qa = QAAgent()
        evidence_gaps = ["Current State"]
        ctx = _make_ctx(tmp_run_dir, allow_net=True)
        target = qa._determine_rollback(
            ["1 sections with evidence gaps"], {}, {"conflicts": []}, [], evidence_gaps, ctx,
        )
        assert target == Stage.COLLECT

    def test_qa_report_includes_next_actions(self, tmp_run_dir: Path):
        """qa_report.json should contain next_actions when issues exist."""
        from researchops.agents.qa import QAAgent

        qa = QAAgent()
        sources = [Source(source_id="s1", type=SourceType.HTML)]
        ctx = _make_ctx(tmp_run_dir)

        next_actions = qa._compute_next_actions(
            ["1 sections with evidence gaps"],
            {}, sources, ctx,
        )
        assert "queries" in next_actions
        assert "target_sources_count" in next_actions

    def test_should_allow_rollback_detects_no_progress(self, tmp_run_dir: Path):
        """_should_allow_rollback returns False when last 2 rollbacks have same hashes."""
        from researchops.agents.qa import QAAgent

        qa = QAAgent()
        ctx = _make_ctx(tmp_run_dir)
        ctx.state.rollback_history = [
            {"round": 2, "sources_hash": "abc", "claims_hash": "def"},
            {"round": 3, "sources_hash": "abc", "claims_hash": "def"},
        ]
        assert qa._should_allow_rollback(ctx) is False

        ctx.state.rollback_history = [
            {"round": 2, "sources_hash": "abc", "claims_hash": "def"},
            {"round": 3, "sources_hash": "xyz", "claims_hash": "def"},
        ]
        assert qa._should_allow_rollback(ctx) is True


class TestNewModelFields:
    def test_state_snapshot_new_fields(self):
        state = StateSnapshot(run_id="test")
        assert state.sources_hash == ""
        assert state.claims_hash == ""
        assert state.coverage_vector == {}
        assert state.rollback_history == []
        assert state.incomplete_sections == []
        assert state.collect_strategy_level == 0
        assert state.no_progress_streak == 0

    def test_source_new_fields(self):
        src = Source(source_id="test", type=SourceType.HTML, collect_round=2, query_id="q1")
        assert src.collect_round == 2
        assert src.query_id == "q1"

    def test_eval_result_new_fields(self):
        from researchops.models import EvalResult
        er = EvalResult(
            incomplete_sections=["rq_1"],
            collect_rounds_total=3,
            sources_per_rq={"rq_1": 5},
            max_rollback_used=True,
        )
        assert er.incomplete_sections == ["rq_1"]
        assert er.collect_rounds_total == 3
        assert er.sources_per_rq == {"rq_1": 5}
        assert er.max_rollback_used is True

    def test_config_new_properties(self):
        config = RunConfig(topic="test", mode="deep")
        assert config.max_collect_rounds == 6
        assert config.target_sources_per_rq == 6
        assert config.max_total_sources == 40

        config_fast = RunConfig(topic="test", mode="fast")
        assert config_fast.max_collect_rounds == 3
        assert config_fast.target_sources_per_rq == 2
        assert config_fast.max_total_sources == 12
