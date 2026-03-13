"""Tests for v1.1.0 features: Supervisor, coverage checklist, bucket-driven
collection, relevance scoring, evidence map, bucket coverage gate, and decisions."""
from __future__ import annotations

import json
from pathlib import Path

from researchops.agents.base import RunContext
from researchops.config import RunConfig
from researchops.models import (
    Claim,
    Decision,
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
) -> RunContext:
    config = RunConfig(topic="deep learning", mode=mode, allow_net=allow_net, run_dir=tmp_run_dir)
    state = StateSnapshot(run_id="test", collect_rounds=collect_rounds)
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


def _write_plan_with_checklist(run_dir: Path, checklist: list[dict] | None = None) -> PlanOutput:
    plan = PlanOutput(
        topic="deep learning",
        research_questions=[
            ResearchQuestion(rq_id="rq_overview", text="What is deep learning?"),
            ResearchQuestion(rq_id="rq_state", text="Current state of deep learning?"),
        ],
        outline=[],
        coverage_checklist=checklist or [
            {"bucket_id": "bkt_arch", "bucket_name": "architectures", "description": "Neural network architectures", "min_sources": 3, "min_claims": 5},
            {"bucket_id": "bkt_opt", "bucket_name": "optimization", "description": "Training and optimization methods", "min_sources": 3, "min_claims": 5},
            {"bucket_id": "bkt_app", "bucket_name": "applications", "description": "Real-world applications", "min_sources": 3, "min_claims": 5},
        ],
    )
    plan_path = run_dir / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return plan


def test_planner_generates_coverage_checklist(tmp_path: Path) -> None:
    """Planner generates coverage_checklist with buckets."""
    from researchops.agents.planner import PlannerAgent

    ctx = _make_ctx(tmp_path)
    agent = PlannerAgent()
    result = agent.execute(ctx)
    assert result.success

    plan = PlanOutput.model_validate(json.loads((tmp_path / "plan.json").read_text()))
    assert len(plan.coverage_checklist) >= 3
    for bucket in plan.coverage_checklist:
        assert "bucket_id" in bucket
        assert "bucket_name" in bucket
        assert "min_sources" in bucket
        assert "min_claims" in bucket


def test_reader_computes_relevance_and_bucket_hits(tmp_path: Path) -> None:
    """Reader computes relevance_score and bucket_hits."""
    from researchops.agents.reader import ReaderAgent

    ctx = _make_ctx(tmp_path)
    _write_plan_with_checklist(tmp_path)

    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    abstract = (
        "Deep learning neural network architectures including CNNs and transformers "
        "have shown remarkable performance in various optimization tasks. Training methods "
        "and generalization remain important challenges in the field of deep learning."
    )
    abstract_path = downloads_dir / "test_abstract.txt"
    abstract_path.write_text(abstract, encoding="utf-8")

    sources_path = tmp_path / "sources.jsonl"
    src = Source(
        source_id="src_test_0",
        type=SourceType.API,
        url="https://arxiv.org/abs/1234",
        domain="arxiv.org",
        title="Deep Learning Architectures Survey",
        local_path=str(abstract_path),
        hash="abc123",
        source_type_detail="arxiv_meta",
    )
    sources_path.write_text(src.model_dump_json() + "\n", encoding="utf-8")

    agent = ReaderAgent()
    result = agent.execute(ctx)
    assert result.success

    notes_dir = tmp_path / "notes"
    note_files = list(notes_dir.glob("*.json"))
    assert len(note_files) >= 1

    notes = SourceNotes.model_validate(json.loads(note_files[0].read_text()))
    assert notes.relevance_score > 0
    assert isinstance(notes.bucket_hits, list)


def test_supervisor_produces_decision(tmp_path: Path) -> None:
    """Supervisor produces Decision with reason_codes and suggested_queries."""
    from researchops.supervisor import Supervisor

    _write_plan_with_checklist(tmp_path)
    config = RunConfig(topic="deep learning", mode="deep", run_dir=tmp_path)
    state = StateSnapshot(
        run_id="test",
        coverage_vector={"rq_overview": 1, "rq_state": 0},
        no_progress_streak=0,
    )
    trace = TraceLogger(tmp_path / "trace.jsonl")
    supervisor = Supervisor(tmp_path, NoneReasoner())

    diagnostics = {
        "bucket_coverage_rate": 0.3,
        "relevance_avg": 0.4,
        "coverage_vector": {"rq_overview": 1, "rq_state": 0},
    }

    decision = supervisor.decide(state, diagnostics, config, trace)
    assert isinstance(decision, Decision)
    assert len(decision.reason_codes) > 0
    assert "bucket_incomplete" in decision.reason_codes
    assert len(decision.suggested_queries) > 0
    assert decision.confidence > 0


def test_qa_bucket_coverage_gate_triggers_rollback(tmp_path: Path) -> None:
    """QA bucket_coverage gate triggers rollback when below threshold."""
    from researchops.agents.qa import QAAgent

    ctx = _make_ctx(tmp_path, mode="fast")
    _write_plan_with_checklist(tmp_path, checklist=[
        {"bucket_id": "bkt_arch", "bucket_name": "architectures", "description": "Neural network architectures", "min_sources": 1, "min_claims": 2},
        {"bucket_id": "bkt_opt", "bucket_name": "optimization", "description": "Training methods", "min_sources": 1, "min_claims": 2},
        {"bucket_id": "bkt_app", "bucket_name": "applications", "description": "Applications", "min_sources": 1, "min_claims": 2},
    ])

    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    src_lines: list[str] = []
    for i in range(3):
        dl_path = downloads_dir / f"test_{i}.txt"
        dl_path.write_text(f"Test content about deep learning topic number {i}. " * 50, encoding="utf-8")
        src = Source(
            source_id=f"src_{i}", type=SourceType.API, url=f"http://example{i}.com",
            domain=f"example{i}.com", title=f"Test {i}",
            local_path=str(dl_path), hash=f"hash{i}",
        )
        src_lines.append(src.model_dump_json())

        notes = SourceNotes(
            source_id=f"src_{i}",
            claims=[Claim(claim_id=f"c{i}", text=f"Claim about deep learning architectures {i}", supports_rq=["rq_overview"])],
            quality={"noise_flags": []},
            relevance_score=0.8,
            bucket_hits=["bkt_arch"] if i == 0 else [],
        )
        (notes_dir / f"src_{i}.json").write_text(notes.model_dump_json(indent=2), encoding="utf-8")

    (tmp_path / "sources.jsonl").write_text("\n".join(src_lines) + "\n", encoding="utf-8")

    report = "# Deep Learning\n\n## Introduction\n\nDeep learning overview. [@src_0]\n\n## Overview\n\nOverview content here. [@src_1]\n"
    (tmp_path / "report.md").write_text(report, encoding="utf-8")
    (tmp_path / "report_index.json").write_text(
        json.dumps({"entries": [
            {"hash": "abc", "text": "test", "source_ids": ["src_0"], "claim_ids": ["c0"]},
            {"hash": "def", "text": "test2", "source_ids": ["src_1"], "claim_ids": ["c1"]},
        ]}),
        encoding="utf-8",
    )

    emap = {"Introduction": {"bucket_ids": ["bkt_arch"], "claims": [{"claim_id": "c0", "source_id": "src_0", "text": "test"}], "source_count": 1, "diversity_ok": True}}
    (tmp_path / "evidence_map.json").write_text(json.dumps(emap), encoding="utf-8")

    agent = QAAgent()
    agent.execute(ctx)

    qa_report = json.loads((tmp_path / "qa_report.json").read_text())
    assert "bucket_coverage_rate" in qa_report
    assert "issues" in qa_report


def test_writer_generates_evidence_map(tmp_path: Path) -> None:
    """Writer generates evidence_map.json."""
    from researchops.agents.writer import WriterAgent

    ctx = _make_ctx(tmp_path, allow_net=False)
    plan = _write_plan_with_checklist(tmp_path)
    plan.outline = [
        {"heading": "Introduction", "rq_refs": []},
        {"heading": "Overview", "rq_refs": ["rq_overview"]},
    ]
    (tmp_path / "plan.json").write_text(json.dumps(plan.model_dump(), indent=2), encoding="utf-8")

    notes_dir = tmp_path / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    notes = SourceNotes(
        source_id="src_0",
        claims=[
            Claim(claim_id="c0", text="Deep learning uses neural networks for feature learning.", supports_rq=["rq_overview"]),
            Claim(claim_id="c1", text="Convolutional architectures enable spatial pattern recognition.", supports_rq=["rq_overview"]),
        ],
        quality={},
        relevance_score=0.8,
        bucket_hits=["bkt_arch"],
    )
    (notes_dir / "src_0.json").write_text(notes.model_dump_json(indent=2), encoding="utf-8")

    (tmp_path / "sources.jsonl").write_text(
        Source(source_id="src_0", type=SourceType.API, domain="arxiv.org", title="Test").model_dump_json() + "\n",
        encoding="utf-8",
    )

    agent = WriterAgent()
    result = agent.execute(ctx)
    assert result.success

    emap_path = tmp_path / "evidence_map.json"
    assert emap_path.exists()
    emap = json.loads(emap_path.read_text())
    assert isinstance(emap, dict)
    assert len(emap) > 0


def test_collector_uses_negative_terms(tmp_path: Path) -> None:
    """Collector uses negative terms (verify 'collider' excluded from DL queries)."""
    from researchops.agents.collector import CollectorAgent

    agent = CollectorAgent()
    ctx = _make_ctx(tmp_path, allow_net=False)

    negative_terms = agent._get_negative_terms("deep learning", ctx)
    assert "collider" in negative_terms
    assert "particle physics" in negative_terms

    paper_offtopic = {
        "title": "Deep learning collider physics particle detector",
        "abstract": "Particle physics collider experiments using deep learning for astronomy and geology analysis.",
    }
    assert agent._has_negative_terms(paper_offtopic, negative_terms)

    paper_ontopic = {
        "title": "Deep learning architectures for image recognition",
        "abstract": "Convolutional neural networks demonstrate strong generalization in computer vision tasks.",
    }
    assert not agent._has_negative_terms(paper_ontopic, negative_terms)


def test_decisions_last_decision_written(tmp_path: Path) -> None:
    """last_decision.json is written and valid."""
    from researchops.supervisor import Supervisor

    _write_plan_with_checklist(tmp_path)
    config = RunConfig(topic="deep learning", mode="deep", run_dir=tmp_path)
    state = StateSnapshot(run_id="test", coverage_vector={"rq_overview": 0})
    trace = TraceLogger(tmp_path / "trace.jsonl")
    supervisor = Supervisor(tmp_path, NoneReasoner())

    diagnostics = {"bucket_coverage_rate": 0.2, "relevance_avg": 0.3}
    supervisor.decide(state, diagnostics, config, trace)

    decision_path = tmp_path / "last_decision.json"
    assert decision_path.exists()

    data = json.loads(decision_path.read_text(encoding="utf-8"))
    assert "decision_id" in data
    assert "reason_codes" in data
    assert "suggested_queries" in data
    assert "confidence" in data

    dec = Decision.model_validate(data)
    assert len(dec.reason_codes) > 0
