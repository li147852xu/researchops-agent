"""Generic multi-agent pipeline — LangGraph graph builder, node functions, and edge routing.

This module is the heart of the platform.  It provides a **single** pipeline
topology that every app shares.  App-specific behaviour is injected via the
:class:`~researchops.apps.registry.AppSpec` (custom planner, evaluator,
tool registration).

Graph topology::

    plan -> collect -> read --(coverage ok)--> verify -> write --(ok)--> qa --(pass)--> eval -> END
              ^                |                                   |          |
              |                +-(coverage low)-> supervisor <-----+          |
              |                                      |                       |
              +--------------------------------------+                       |
                                                     +-- (rewrite) -> write -+
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from researchops.apps.registry import AppSpec

logger = logging.getLogger(__name__)


# ── Generic pipeline state ─────────────────────────────────────────────

class WorkingMemory(TypedDict, total=False):
    retriever: Any
    low_quality_sources: list[str]
    rq_claim_counts: dict[str, int]
    rq_source_counts: dict[str, int]
    evidence_limited: bool
    has_conflicts: bool
    conflict_count: int


class PipelineState(TypedDict, total=False):
    """Shared LangGraph state used by every app on the platform."""
    run_id: str
    run_dir: str
    topic: str
    config: dict[str, Any]
    plan: dict[str, Any] | None
    sources: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    report: str
    qa_result: dict[str, Any] | None
    eval_result: dict[str, Any] | None
    diagnostics: dict[str, Any]
    stage: str
    rollback_target: str | None
    collect_rounds: int
    max_collect_rounds: int
    write_rounds: int
    refinement_count: int
    decision_history: list[dict[str, Any]]
    completed_stages: list[str]
    memory: WorkingMemory
    last_error: str | None


# ── Context cache (one per pipeline invocation) ───────────────────────

_CTX_CACHE: dict[str, Any] = {}


def clear_ctx_cache() -> None:
    _CTX_CACHE.clear()


def _build_ctx(state: PipelineState, app_spec: AppSpec):
    """Construct a :class:`RunContext` from pipeline state and app spec."""
    from researchops.core.checkpoint import load_state, save_state
    from researchops.core.context import RunContext
    from researchops.core.config import BaseAppConfig
    from researchops.core.sandbox.proc import SubprocessSandbox
    from researchops.core.state import StateSnapshot
    from researchops.core.tools.registry import ToolRegistry
    from researchops.core.tracing import TraceLogger
    from researchops.reasoning import create_reasoner

    run_dir = Path(state["run_dir"])
    _cache = _CTX_CACHE

    if "ctx" not in _cache or _cache.get("_run_id") != state.get("run_id"):
        config = app_spec.config_class.model_validate(state["config"])
        trace = TraceLogger(run_dir / "trace.jsonl")

        registry = ToolRegistry()
        app_spec.register_tools(registry)
        registry.set_persistent_cache_path(run_dir / "cache.json")
        perms: set[str] = {"sandbox"}
        if config.allow_net:
            perms.add("net")
        registry.grant_permissions(perms)

        sandbox = SubprocessSandbox()
        reasoner = create_reasoner(config)

        st = load_state(run_dir)
        if st is None:
            st = StateSnapshot(
                run_id=state.get("run_id", run_dir.name),
                config_snapshot=config.safe_dump(),
            )
            save_state(st, run_dir)

        _cache["ctx"] = RunContext(
            run_dir=run_dir, config=config, state=st,
            registry=registry, trace=trace, sandbox=sandbox,
            reasoner=reasoner,
        )
        _cache["_run_id"] = state.get("run_id")

    ctx = _cache["ctx"]
    ctx.state.collect_rounds = state.get("collect_rounds", ctx.state.collect_rounds)
    ctx.state.write_rounds = state.get("write_rounds", ctx.state.write_rounds)
    ctx.state.refinement_count = state.get("refinement_count", ctx.state.refinement_count)
    ctx.state.incomplete_sections = []
    ctx.shared.update(state.get("memory", {}))
    return ctx


def _sync_state_back(ctx, stage: str) -> dict[str, Any]:
    from researchops.core.checkpoint import save_state
    save_state(ctx.state, ctx.run_dir)
    return {
        "stage": stage,
        "collect_rounds": ctx.state.collect_rounds,
        "write_rounds": ctx.state.write_rounds,
        "refinement_count": ctx.state.refinement_count,
        "memory": dict(ctx.shared),
    }


# ── Node function factories ────────────────────────────────────────────
# Each factory captures the AppSpec so the same graph topology works for
# any registered app.

def _make_plan_node(app_spec: AppSpec):
    def plan_node(state: PipelineState) -> dict[str, Any]:
        ctx = _build_ctx(state, app_spec)
        for d in ["notes", "code", "code/logs", "artifacts", "downloads"]:
            (ctx.run_dir / d).mkdir(parents=True, exist_ok=True)

        if app_spec.custom_planner is not None:
            app_spec.custom_planner(ctx)
            ctx.trace.log(stage="PLAN", agent="custom_planner", action="complete")
        else:
            from researchops.agents.planning import PlannerAgent
            PlannerAgent().execute(ctx)

        plan_data = None
        plan_path = ctx.run_dir / "plan.json"
        if plan_path.exists():
            plan_data = json.loads(plan_path.read_text(encoding="utf-8"))

        updates = _sync_state_back(ctx, "COLLECT")
        updates["plan"] = plan_data
        updates["completed_stages"] = ["PLAN"]
        return updates
    return plan_node


def _make_collect_node(app_spec: AppSpec):
    def collect_node(state: PipelineState) -> dict[str, Any]:
        from researchops.agents.collection import CollectorAgent

        ctx = _build_ctx(state, app_spec)
        CollectorAgent().execute(ctx)

        sources_data: list[dict] = []
        sources_path = ctx.run_dir / "sources.jsonl"
        if sources_path.exists():
            for line in sources_path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    sources_data.append(json.loads(line))

        updates = _sync_state_back(ctx, "READ")
        updates["sources"] = sources_data
        completed = list(state.get("completed_stages", []))
        if "COLLECT" not in completed:
            completed.append("COLLECT")
        updates["completed_stages"] = completed
        return updates
    return collect_node


def _make_read_node(app_spec: AppSpec):
    def read_node(state: PipelineState) -> dict[str, Any]:
        from researchops.agents.reading import ReaderAgent
        from researchops.retrieval import create_retriever
        from researchops.utils import compute_coverage, load_claim_dicts

        ctx = _build_ctx(state, app_spec)

        retriever = create_retriever(
            ctx.config.retrieval.value, ctx.run_dir,
            embedder_model=ctx.config.embedder_model,
        )
        retriever.index(load_claim_dicts(ctx.run_dir))
        ctx.shared["retriever"] = retriever

        ReaderAgent().execute(ctx)

        all_claims_post = load_claim_dicts(ctx.run_dir)
        retriever.index(all_claims_post)
        ctx.shared["retriever"] = retriever

        rq_counts, coverage = compute_coverage(ctx.run_dir)

        updates = _sync_state_back(ctx, "VERIFY")
        updates["claims"] = all_claims_post
        updates["diagnostics"] = {"coverage": coverage, "rq_claim_counts": rq_counts}
        completed = list(state.get("completed_stages", []))
        if "READ" not in completed:
            completed.append("READ")
        updates["completed_stages"] = completed
        return updates
    return read_node


def _make_verify_node(app_spec: AppSpec):
    def verify_node(state: PipelineState) -> dict[str, Any]:
        from researchops.agents.verification import VerifierAgent

        ctx = _build_ctx(state, app_spec)
        VerifierAgent().execute(ctx)

        updates = _sync_state_back(ctx, "WRITE")
        completed = list(state.get("completed_stages", []))
        if "VERIFY" not in completed:
            completed.append("VERIFY")
        updates["completed_stages"] = completed
        return updates
    return verify_node


def _make_write_node(app_spec: AppSpec):
    def write_node(state: PipelineState) -> dict[str, Any]:
        from researchops.agents.writing import WriterAgent

        ctx = _build_ctx(state, app_spec)
        ctx.state.write_rounds += 1
        result = WriterAgent().execute(ctx)

        report = ""
        report_path = ctx.run_dir / "report.md"
        if report_path.exists():
            report = report_path.read_text(encoding="utf-8")

        updates = _sync_state_back(ctx, "QA")
        updates["report"] = report
        updates["rollback_target"] = result.rollback_to.value if result.rollback_to else None
        completed = list(state.get("completed_stages", []))
        if "WRITE" not in completed:
            completed.append("WRITE")
        updates["completed_stages"] = completed
        return updates
    return write_node


def _make_qa_node(app_spec: AppSpec):
    def qa_node(state: PipelineState) -> dict[str, Any]:
        from researchops.agents.qa import QAAgent

        ctx = _build_ctx(state, app_spec)
        result = QAAgent().execute(ctx)

        qa_data: dict[str, Any] = {
            "passed": result.success,
            "message": result.message,
            "issues": result.data.get("issues", []),
        }

        updates = _sync_state_back(ctx, "DONE" if result.success else state.get("stage", "QA"))
        updates["qa_result"] = qa_data
        updates["rollback_target"] = result.rollback_to.value if result.rollback_to else None
        completed = list(state.get("completed_stages", []))
        if result.success and "QA" not in completed:
            completed.append("QA")
        updates["completed_stages"] = completed
        return updates
    return qa_node


def _make_supervisor_node(app_spec: AppSpec):
    def supervisor_node(state: PipelineState) -> dict[str, Any]:
        from researchops.core.orchestration.supervisor import Supervisor

        ctx = _build_ctx(state, app_spec)
        supervisor = Supervisor(ctx.run_dir, ctx.reasoner)

        diagnostics = state.get("diagnostics", {})
        qa_report_path = ctx.run_dir / "qa_report.json"
        if qa_report_path.exists():
            try:
                qa_data = json.loads(qa_report_path.read_text(encoding="utf-8"))
                diagnostics["bucket_coverage_rate"] = qa_data.get("bucket_coverage_rate", 1.0)
                diagnostics["relevance_avg"] = qa_data.get("relevance_avg", 1.0)
            except Exception:
                pass

        diagnostics["coverage_vector"] = ctx.state.coverage_vector

        history = list(state.get("decision_history", []))

        if len(history) >= 5:
            logger.warning("Supervisor: max decision loops reached (%d), forcing progression", len(history))
            ctx.trace.log(
                stage="SUPERVISOR", action="decide",
                meta={"forced_stop": True, "loop_count": len(history), "reason": "max loops reached"},
            )
            completed = list(state.get("completed_stages", []))
            return {
                "rollback_target": None,
                "collect_rounds": state.get("collect_rounds", 1),
                "decision_history": history,
                "diagnostics": diagnostics,
                "stage": "WRITE",
                "completed_stages": completed,
            }

        decision = supervisor.decide(ctx.state, diagnostics, ctx.config, ctx.trace)

        rollback_target = state.get("rollback_target", "COLLECT")
        if not rollback_target:
            rollback_target = "COLLECT" if decision.reason_codes else "WRITE"

        collect_rounds = state.get("collect_rounds", 1)
        max_rounds = state.get("max_collect_rounds", ctx.config.max_collect_rounds)

        if rollback_target == "COLLECT" and collect_rounds >= max_rounds:
            rollback_target = "WRITE"

        if rollback_target == "COLLECT":
            collect_rounds += 1

        history.append({
            "decision_id": decision.decision_id,
            "reason_codes": decision.reason_codes,
            "target": rollback_target,
            "confidence": decision.confidence,
        })

        completed = list(state.get("completed_stages", []))
        stage_order = ["PLAN", "COLLECT", "READ", "VERIFY", "WRITE", "QA"]
        if rollback_target in stage_order:
            idx = stage_order.index(rollback_target)
            completed = [s for s in completed if s in stage_order[:idx]]

        return {
            "rollback_target": rollback_target,
            "collect_rounds": collect_rounds,
            "decision_history": history,
            "diagnostics": diagnostics,
            "stage": rollback_target,
            "completed_stages": completed,
        }
    return supervisor_node


def _make_eval_node(app_spec: AppSpec):
    def eval_node(state: PipelineState) -> dict[str, Any]:
        run_dir = Path(state["run_dir"])
        config = app_spec.config_class.model_validate(state["config"])
        if app_spec.compute_eval is not None:
            result = app_spec.compute_eval(run_dir, config=config)
            return {"eval_result": result.model_dump(), "stage": "DONE"}
        return {"eval_result": {}, "stage": "DONE"}
    return eval_node


# ── Edge routing (app-agnostic) ────────────────────────────────────────

def _after_read(state: PipelineState) -> str:
    diagnostics = state.get("diagnostics", {})
    coverage = diagnostics.get("coverage", 1.0)
    config = state.get("config", {})
    threshold = config.get("acceptance_threshold", 0.7)
    collect_rounds = state.get("collect_rounds", 1)
    max_rounds = state.get("max_collect_rounds", 6)
    if coverage < threshold and collect_rounds < max_rounds:
        return "supervisor"
    return "verify"


def _after_write(state: PipelineState) -> str:
    rollback = state.get("rollback_target")
    write_rounds = state.get("write_rounds", 0)
    if write_rounds >= 2:
        return "qa"
    if rollback and rollback != "None":
        return "supervisor"
    return "qa"


def _after_qa(state: PipelineState) -> str:
    qa = state.get("qa_result", {})
    if qa.get("passed", False):
        return "evaluate"
    rollback = state.get("rollback_target")
    if rollback == "WRITE":
        return "write"
    return "supervisor"


def _after_supervisor(state: PipelineState) -> str:
    target = state.get("rollback_target", "collect")
    target_lower = target.lower() if target else "collect"
    valid = {"collect", "read", "write", "verify"}
    return target_lower if target_lower in valid else "collect"


# ── Graph builder ──────────────────────────────────────────────────────

def build_pipeline_graph(app_spec: AppSpec) -> Any:
    """Construct and compile the multi-agent pipeline for *any* registered app.

    The graph topology is identical for every app; only the node behaviour
    differs via the ``app_spec`` (custom planner, tool set, evaluator).
    """
    from langgraph.graph import END, StateGraph

    g: StateGraph = StateGraph(PipelineState)

    g.add_node("plan", _make_plan_node(app_spec))
    g.add_node("collect", _make_collect_node(app_spec))
    g.add_node("read", _make_read_node(app_spec))
    g.add_node("verify", _make_verify_node(app_spec))
    g.add_node("write", _make_write_node(app_spec))
    g.add_node("qa", _make_qa_node(app_spec))
    g.add_node("evaluate", _make_eval_node(app_spec))
    g.add_node("supervisor", _make_supervisor_node(app_spec))

    g.add_edge("plan", "collect")
    g.add_edge("collect", "read")
    g.add_conditional_edges("read", _after_read, {"verify": "verify", "supervisor": "supervisor"})
    g.add_edge("verify", "write")
    g.add_conditional_edges("write", _after_write, {"qa": "qa", "supervisor": "supervisor"})
    g.add_conditional_edges("qa", _after_qa, {"evaluate": "evaluate", "supervisor": "supervisor", "write": "write"})
    g.add_conditional_edges("supervisor", _after_supervisor, {"collect": "collect", "read": "read", "write": "write", "verify": "verify"})
    g.add_edge("evaluate", END)

    g.set_entry_point("plan")
    return g.compile()


def build_initial_state(
    run_id: str,
    run_dir: Path,
    topic: str,
    config: Any,
) -> dict[str, Any]:
    """Create the initial LangGraph state dict for any app."""
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "topic": topic,
        "config": config.model_dump(mode="json"),
        "plan": None,
        "sources": [],
        "claims": [],
        "report": "",
        "qa_result": None,
        "eval_result": None,
        "diagnostics": {},
        "stage": "PLAN",
        "rollback_target": None,
        "collect_rounds": 1,
        "max_collect_rounds": config.max_collect_rounds,
        "write_rounds": 0,
        "refinement_count": 0,
        "decision_history": [],
        "completed_stages": [],
        "memory": {},
        "last_error": None,
    }
