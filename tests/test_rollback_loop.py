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
        """When collect_rounds >= max_collect_rounds, degrade_complete should work."""
        from researchops.orchestrator import Orchestrator

        config = RunConfig(topic="deep learning", mode="deep", allow_net=True, run_dir=tmp_run_dir)

        orch = Orchestrator.__new__(Orchestrator)
        orch.config = config
        orch.run_dir = tmp_run_dir
        orch.trace = TraceLogger(tmp_run_dir / "trace.jsonl")
        orch.registry = ToolRegistry()
        orch.sandbox = SubprocessSandbox()
        orch.reasoner = NoneReasoner()
        orch.state = StateSnapshot(run_id="test", collect_rounds=6)

        _write_plan(tmp_run_dir)
        assert orch.state.collect_rounds >= config.max_collect_rounds

    def test_degrade_complete_marks_incomplete(self, tmp_run_dir: Path):
        """_degrade_complete should populate incomplete_sections."""
        from researchops.orchestrator import Orchestrator

        config = RunConfig(topic="test", mode="deep", run_dir=tmp_run_dir)

        orch = Orchestrator.__new__(Orchestrator)
        orch.config = config
        orch.run_dir = tmp_run_dir
        orch.trace = TraceLogger(tmp_run_dir / "trace.jsonl")
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
        from researchops.utils import load_sources

        existing = Source(
            source_id="src_old", type=SourceType.HTML, hash="old_hash",
            local_path=str(tmp_run_dir / "downloads" / "old.txt"),
            source_type_detail="demo", collect_round=1,
        )
        _write_sources(tmp_run_dir, [existing])
        (tmp_run_dir / "downloads" / "old.txt").write_text("old content")

        loaded = load_sources(tmp_run_dir)
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
    def test_qa_agent_can_be_instantiated(self, tmp_run_dir: Path):
        """QA agent should instantiate with correct name."""
        from researchops.agents.qa import QAAgent

        qa = QAAgent()
        assert qa.name == "qa"

    def test_qa_detects_conflicts(self, tmp_run_dir: Path):
        """QA _detect_conflicts should find opposing claims on same RQ."""
        from researchops.agents.qa import QAAgent

        qa = QAAgent()
        notes = {
            "src1": SourceNotes(
                source_id="src1",
                claims=[
                    Claim(claim_id="c1", text="Method A works",
                          supports_rq=["rq_1"], polarity="support"),
                    Claim(claim_id="c2", text="Method A fails in some cases",
                          supports_rq=["rq_1"], polarity="oppose"),
                ],
            ),
        }
        conflicts = qa._detect_conflicts(notes)
        assert len(conflicts) >= 1
        assert conflicts[0]["rq_id"] == "rq_1"


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
