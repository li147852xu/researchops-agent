"""LangGraph node functions — each wraps an existing agent."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from researchops.agents.base import RunContext
from researchops.agents.collector import CollectorAgent
from researchops.agents.planner import PlannerAgent
from researchops.agents.qa import QAAgent
from researchops.agents.reader import ReaderAgent
from researchops.agents.verifier import VerifierAgent
from researchops.agents.writer import WriterAgent
from researchops.checkpoint import save_state
from researchops.graph.state import ResearchState
from researchops.supervisor import Supervisor
from researchops.utils import compute_coverage, load_claim_dicts

logger = logging.getLogger(__name__)


_CTX_CACHE: dict[str, Any] = {}


def clear_ctx_cache() -> None:
    """Reset the module-level context cache between graph runs."""
    _CTX_CACHE.clear()


def _build_ctx(state: ResearchState) -> RunContext:
    """Reconstruct a RunContext from the graph state.

    Heavy objects (registry, sandbox, reasoner, trace) are cached on the
    first call and reused on subsequent calls within the same graph run.
    """
    run_dir = Path(state["run_dir"])

    _cache = _CTX_CACHE
    if "ctx" not in _cache or _cache.get("_run_id") != state.get("run_id"):
        from researchops.checkpoint import load_state
        from researchops.config import RunConfig
        from researchops.reasoning import create_reasoner
        from researchops.registry.builtin import register_builtin_tools
        from researchops.registry.manager import ToolRegistry
        from researchops.sandbox.proc import SubprocessSandbox
        from researchops.trace import TraceLogger

        config = RunConfig.model_validate(state["config"])
        trace = TraceLogger(run_dir / "trace.jsonl")

        registry = ToolRegistry()
        register_builtin_tools(registry)
        registry.set_persistent_cache_path(run_dir / "cache.json")
        perms: set[str] = {"sandbox"}
        if config.allow_net:
            perms.add("net")
        registry.grant_permissions(perms)

        sandbox = SubprocessSandbox()
        reasoner = create_reasoner(config)

        st = load_state(run_dir)
        if st is None:
            from researchops.models import StateSnapshot
            st = StateSnapshot(
                run_id=state.get("run_id", run_dir.name),
                config_snapshot=config.model_dump(mode="json"),
            )
            save_state(st, run_dir)

        _cache["ctx"] = RunContext(
            run_dir=run_dir, config=config, state=st,
            registry=registry, trace=trace, sandbox=sandbox,
            reasoner=reasoner,
        )
        _cache["_run_id"] = state.get("run_id")

    ctx: RunContext = _cache["ctx"]
    # Sync mutable state fields back from graph state
    ctx.state.collect_rounds = state.get("collect_rounds", ctx.state.collect_rounds)
    ctx.state.write_rounds = state.get("write_rounds", ctx.state.write_rounds)
    ctx.state.refinement_count = state.get("refinement_count", ctx.state.refinement_count)
    ctx.state.incomplete_sections = []
    ctx.shared.update(state.get("memory", {}))
    return ctx


def _sync_state_back(ctx: RunContext, stage: str) -> dict[str, Any]:
    """Extract updated fields from ctx back into graph state updates."""
    save_state(ctx.state, ctx.run_dir)
    return {
        "stage": stage,
        "collect_rounds": ctx.state.collect_rounds,
        "write_rounds": ctx.state.write_rounds,
        "refinement_count": ctx.state.refinement_count,
        "memory": dict(ctx.shared),
    }


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def plan_node(state: ResearchState) -> dict[str, Any]:
    ctx = _build_ctx(state)
    for d in ["notes", "code", "code/logs", "artifacts", "downloads"]:
        (ctx.run_dir / d).mkdir(parents=True, exist_ok=True)

    agent = PlannerAgent()
    agent.execute(ctx)

    plan_data = None
    plan_path = ctx.run_dir / "plan.json"
    if plan_path.exists():
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))

    updates = _sync_state_back(ctx, "COLLECT")
    updates["plan"] = plan_data
    updates["completed_stages"] = ["PLAN"]
    return updates


def collect_node(state: ResearchState) -> dict[str, Any]:
    ctx = _build_ctx(state)
    agent = CollectorAgent()
    agent.execute(ctx)

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


def read_node(state: ResearchState) -> dict[str, Any]:
    ctx = _build_ctx(state)

    from researchops.retrieval import create_retriever

    retriever = create_retriever(
        ctx.config.retrieval.value, ctx.run_dir,
        embedder_model=ctx.config.embedder_model,
    )
    retriever.index(load_claim_dicts(ctx.run_dir))
    ctx.shared["retriever"] = retriever

    agent = ReaderAgent()
    agent.execute(ctx)

    all_claims_post = load_claim_dicts(ctx.run_dir)
    retriever.index(all_claims_post)
    ctx.shared["retriever"] = retriever

    rq_counts, coverage = compute_coverage(ctx.run_dir)

    updates = _sync_state_back(ctx, "VERIFY")
    updates["claims"] = all_claims_post
    updates["diagnostics"] = {
        "coverage": coverage,
        "rq_claim_counts": rq_counts,
    }
    completed = list(state.get("completed_stages", []))
    if "READ" not in completed:
        completed.append("READ")
    updates["completed_stages"] = completed
    return updates


def verify_node(state: ResearchState) -> dict[str, Any]:
    ctx = _build_ctx(state)
    agent = VerifierAgent()
    agent.execute(ctx)

    updates = _sync_state_back(ctx, "WRITE")
    completed = list(state.get("completed_stages", []))
    if "VERIFY" not in completed:
        completed.append("VERIFY")
    updates["completed_stages"] = completed
    return updates


def write_node(state: ResearchState) -> dict[str, Any]:
    ctx = _build_ctx(state)
    ctx.state.write_rounds += 1
    agent = WriterAgent()
    result = agent.execute(ctx)

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


def qa_node(state: ResearchState) -> dict[str, Any]:
    ctx = _build_ctx(state)
    agent = QAAgent()
    result = agent.execute(ctx)

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


def supervisor_node(state: ResearchState) -> dict[str, Any]:
    """Supervisor analyses diagnostics and decides the rollback target."""
    ctx = _build_ctx(state)
    supervisor = Supervisor(ctx.run_dir, ctx.reasoner)

    diagnostics = state.get("diagnostics", {})
    # Load QA report diagnostics
    qa_report_path = ctx.run_dir / "qa_report.json"
    if qa_report_path.exists():
        try:
            qa_data = json.loads(qa_report_path.read_text(encoding="utf-8"))
            diagnostics["bucket_coverage_rate"] = qa_data.get("bucket_coverage_rate", 1.0)
            diagnostics["relevance_avg"] = qa_data.get("relevance_avg", 1.0)
        except Exception:
            pass

    diagnostics["coverage_vector"] = ctx.state.coverage_vector

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

    history = list(state.get("decision_history", []))
    history.append({
        "decision_id": decision.decision_id,
        "reason_codes": decision.reason_codes,
        "target": rollback_target,
        "confidence": decision.confidence,
    })

    # Clear completed stages from rollback target onwards
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


def eval_node(state: ResearchState) -> dict[str, Any]:
    """Compute evaluation metrics."""
    from researchops.config import RunConfig
    from researchops.evaluator import compute_eval

    run_dir = Path(state["run_dir"])
    config = RunConfig.model_validate(state["config"])
    result = compute_eval(run_dir, config=config)

    return {
        "eval_result": result.model_dump(),
        "stage": "DONE",
    }
