from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from researchops.registry.manager import ToolRegistry
from researchops.trace import TraceLogger


def test_md_constraint():
    """Only README.md and CHANGELOG.md should exist in the repo (excluding runs/ and .pytest_cache/)."""
    repo_root = Path(__file__).parent.parent
    md_files = []
    for f in repo_root.rglob("*.md"):
        rel = f.relative_to(repo_root)
        parts = rel.parts
        if any(p.startswith(".") for p in parts):
            continue
        if "runs" in parts or "__pycache__" in parts:
            continue
        md_files.append(str(rel))
    allowed = {"README.md", "CHANGELOG.md"}
    assert set(md_files) == allowed, f"Unexpected .md files: {set(md_files) - allowed}"


def test_plan_refinement_triggers_rollback(tmp_run_dir: Path):
    """When coverage is low and allow_net=true, orchestrator should roll back to COLLECT."""
    from researchops.config import RunConfig
    from researchops.models import PlanOutput, ResearchQuestion, StateSnapshot

    plan = PlanOutput(
        topic="test",
        research_questions=[
            ResearchQuestion(rq_id="rq_alpha", text="What is alpha?"),
            ResearchQuestion(rq_id="rq_beta", text="What is beta?"),
        ],
        outline=[],
        acceptance_threshold=0.9,
    )
    (tmp_run_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    notes_dir = tmp_run_dir / "notes"
    notes_dir.mkdir(exist_ok=True)
    note = {"source_id": "s1", "claims": [
        {"claim_id": "c1", "text": "x", "supports_rq": ["rq_alpha"], "evidence_spans": ["e"]}
    ]}
    (notes_dir / "s1.json").write_text(json.dumps(note), encoding="utf-8")

    from researchops.agents.base import RunContext
    from researchops.reasoning.none import NoneReasoner
    from researchops.sandbox.proc import SubprocessSandbox

    config = RunConfig(topic="test", allow_net=True, run_dir=tmp_run_dir)
    trace = TraceLogger(tmp_run_dir / "trace.jsonl")
    registry = ToolRegistry()
    registry.grant_permissions({"sandbox", "net"})

    from researchops.orchestrator import Orchestrator
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = config
    orch.run_dir = tmp_run_dir
    orch.trace = trace
    orch.registry = registry
    orch.sandbox = SubprocessSandbox()
    orch.reasoner = NoneReasoner()
    orch.state = StateSnapshot(run_id="test", refinement_count=0)

    ctx = RunContext(
        run_dir=tmp_run_dir, config=config, state=orch.state,
        registry=registry, trace=trace, sandbox=orch.sandbox, reasoner=orch.reasoner,
    )

    result = orch._check_plan_coverage(ctx)
    assert result == "refinement"
    assert orch.state.refinement_count == 1
    assert orch.state.collect_rounds == 2

    events = trace.read_all()
    refine_events = [e for e in events if e.action == "plan.refine"]
    assert len(refine_events) >= 1


def test_conflict_scan_produces_json(tmp_run_dir: Path):
    """Conflict scan should detect opposing claims and write qa_conflicts.json."""
    from researchops.agents.qa import QAAgent
    from researchops.models import Claim, SourceNotes

    notes = {
        "src1": SourceNotes(
            source_id="src1",
            claims=[
                Claim(claim_id="c1", text="Method A works well", supports_rq=["rq_1"], polarity="support"),
                Claim(claim_id="c2", text="Method A has limitations and cannot scale", supports_rq=["rq_1"], polarity="oppose"),
            ],
        ),
    }

    qa = QAAgent()
    result = qa._conflict_scan(notes)
    assert len(result["conflicts"]) > 0
    assert result["conflicts"][0]["rq_id"] == "rq_1"


def test_replay_no_tools_monkeypatch(tmp_run_dir: Path):
    """Replay --no-tools --json should never call ToolRegistry.invoke."""
    trace_path = tmp_run_dir / "trace.jsonl"
    logger = TraceLogger(trace_path)
    logger.log(stage="PLAN", action="start")
    logger.log(stage="COLLECT", tool="web_search", action="invoke", input_summary="q=test")
    logger.log(stage="ORCHESTRATOR", action="run_complete", duration_ms=50)

    invoke_called = False
    original_invoke = ToolRegistry.invoke

    def patched_invoke(self, *args, **kwargs):
        nonlocal invoke_called
        invoke_called = True
        return original_invoke(self, *args, **kwargs)

    with patch.object(ToolRegistry, "invoke", patched_invoke):
        import io
        from contextlib import redirect_stdout

        from researchops.orchestrator import replay_run

        buf = io.StringIO()
        with redirect_stdout(buf):
            replay_run(tmp_run_dir, from_step=0, no_tools=True, json_output=True)
        output = buf.getvalue()

    assert not invoke_called, "ToolRegistry.invoke should never be called during replay --no-tools"
    data = json.loads(output)
    assert isinstance(data, list)
    dry_runs = [e for e in data if e.get("dry_run")]
    assert len(dry_runs) >= 1


def test_openai_compat_headers():
    """OpenAICompatReasoner should pass custom base_url and headers to httpx."""
    from researchops.reasoning.openai_compat import OpenAICompatReasoner

    reasoner = OpenAICompatReasoner(
        api_key="test-key-123",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        provider_label="deepseek",
        extra_headers={"X-Custom": "value"},
    )

    assert reasoner.base_url == "https://api.deepseek.com/v1"
    assert reasoner.provider_label == "deepseek"
    headers = reasoner._build_headers()
    assert headers["Authorization"] == "Bearer test-key-123"
    assert headers["X-Custom"] == "value"

    stats = reasoner.get_stats()
    assert stats["provider_label"] == "deepseek"


def test_source_strategy_enums():
    """SourceStrategy and RetrievalMode enums should be importable and valid."""
    from researchops.config import RetrievalMode, SourceStrategy

    assert SourceStrategy.DEMO.value == "demo"
    assert SourceStrategy.ARXIV.value == "arxiv"
    assert SourceStrategy.WEB.value == "web"
    assert SourceStrategy.HYBRID.value == "hybrid"
    assert RetrievalMode.NONE.value == "none"
    assert RetrievalMode.BM25.value == "bm25"


def test_source_model_has_type_detail():
    """Source model should have source_type_detail field."""
    from researchops.models import Source, SourceType

    src = Source(source_id="test", type=SourceType.API, source_type_detail="arxiv_meta")
    assert src.source_type_detail == "arxiv_meta"


def test_source_notes_has_bibliographic():
    """SourceNotes should have bibliographic and quality fields."""
    from researchops.models import SourceNotes

    notes = SourceNotes(
        source_id="test",
        bibliographic={"paper_id": "2301.00001"},
        quality={"readability_score": 0.8},
    )
    assert notes.bibliographic["paper_id"] == "2301.00001"
    assert notes.quality["readability_score"] == 0.8


def test_eval_result_has_new_metrics():
    """EvalResult should have papers_per_rq, low_quality_source_rate, section_nonempty_rate."""
    from researchops.models import EvalResult

    result = EvalResult(papers_per_rq=2.5, low_quality_source_rate=0.1, section_nonempty_rate=0.9)
    assert result.papers_per_rq == 2.5
    assert result.low_quality_source_rate == 0.1
    assert result.section_nonempty_rate == 0.9


def test_qa_source_quality_check(tmp_run_dir: Path):
    """QA source_quality_check should detect high low-quality ratio."""
    from researchops.agents.base import RunContext
    from researchops.agents.qa import QAAgent
    from researchops.config import RunConfig
    from researchops.models import StateSnapshot
    from researchops.reasoning.none import NoneReasoner
    from researchops.registry.manager import ToolRegistry
    from researchops.sandbox.proc import SubprocessSandbox

    config = RunConfig(topic="test", run_dir=tmp_run_dir)
    trace = TraceLogger(tmp_run_dir / "trace.jsonl")

    ctx = RunContext(
        run_dir=tmp_run_dir, config=config,
        state=StateSnapshot(run_id="test"),
        registry=ToolRegistry(),
        trace=trace,
        sandbox=SubprocessSandbox(),
        reasoner=NoneReasoner(),
        shared={"low_quality_sources": ["s1", "s2", "s3"]},
    )

    from researchops.models import Source, SourceType
    sources_path = tmp_run_dir / "sources.jsonl"
    sources = [
        Source(source_id="s1", type=SourceType.HTML),
        Source(source_id="s2", type=SourceType.HTML),
        Source(source_id="s3", type=SourceType.HTML),
        Source(source_id="s4", type=SourceType.HTML),
    ]
    with sources_path.open("w") as f:
        for s in sources:
            f.write(s.model_dump_json() + "\n")

    qa = QAAgent()
    issues = qa._source_quality_check(ctx)
    assert len(issues) > 0
    assert "source_quality" in issues[0]
