"""Market Intelligence pipeline — LangGraph-based financial research pipeline.

Demonstrates core kernel reusability: same orchestration pattern, same agents layer,
same tool governance, same checkpoint/state infrastructure — different app config,
prompts, and schemas.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# ── Graph state (mirrors research pipeline structure) ──────────────────

class QuantWorkingMemory(TypedDict, total=False):
    retriever: Any
    low_quality_sources: list[str]
    rq_claim_counts: dict[str, int]
    rq_source_counts: dict[str, int]
    evidence_limited: bool
    has_conflicts: bool
    conflict_count: int


class QuantState(TypedDict, total=False):
    run_id: str
    run_dir: str
    query: str
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
    memory: QuantWorkingMemory
    last_error: str | None


# ── Context cache ──────────────────────────────────────────────────────

_CTX_CACHE: dict[str, Any] = {}


def clear_ctx_cache() -> None:
    _CTX_CACHE.clear()


def _build_ctx(state: QuantState):
    """Build RunContext for quant pipeline — reuses core infrastructure."""
    from researchops.apps.market.adapters import register_market_tools
    from researchops.apps.market.config import MarketConfig
    from researchops.core.checkpoint import load_state, save_state
    from researchops.core.context import RunContext
    from researchops.core.sandbox.proc import SubprocessSandbox
    from researchops.core.state import StateSnapshot
    from researchops.core.tools.registry import ToolRegistry
    from researchops.core.tracing import TraceLogger
    from researchops.reasoning import create_reasoner

    run_dir = Path(state["run_dir"])
    _cache = _CTX_CACHE

    if "ctx" not in _cache or _cache.get("_run_id") != state.get("run_id"):
        config = MarketConfig.model_validate(state["config"])
        trace = TraceLogger(run_dir / "trace.jsonl")

        registry = ToolRegistry()
        register_market_tools(registry)
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


# ── Quant-specific planner ─────────────────────────────────────────────

def _quant_plan(ctx) -> None:
    """Generate quant research plan using LLM or rule-based fallback."""
    from researchops.apps.market.prompts import (
        QUANT_PLANNER_BUCKETS,
        QUANT_PLANNER_HEADINGS,
        QUANT_PLANNER_RQS,
    )
    from researchops.apps.market.schemas import OutlineSection, QuantPlanOutput, QuantQuestion
    from researchops.apps.research.prompts import parse_json_response

    query = ctx.config.query if hasattr(ctx.config, "query") else ctx.config.topic
    ticker = getattr(ctx.config, "ticker", "")
    analysis_type = getattr(ctx.config, "analysis_type", "fundamental")
    mode = ctx.config.mode.value
    num_rqs = 5 if mode == "deep" else 4

    rqs: list[QuantQuestion] = []
    if ctx.reasoner.is_llm:
        sys_msg, user_msg = QUANT_PLANNER_RQS.render(num_rqs=str(num_rqs), topic=query)
        try:
            raw = ctx.reasoner.complete_text(user_msg, context=sys_msg, trace=ctx.trace)
            data = parse_json_response(raw)
            for q in data.get("questions", [])[:num_rqs]:
                rqs.append(QuantQuestion(
                    rq_id=q.get("rq_id", f"rq_{len(rqs)}"),
                    text=q.get("text", ""),
                    priority=q.get("priority", 1),
                    needs_verification=q.get("needs_verification", False),
                    dimension=q.get("dimension", ""),
                ))
        except Exception as exc:
            logger.warning("LLM quant planning failed: %s", exc)

    if not rqs:
        rqs = _rule_based_quant_rqs(query, ticker, analysis_type)

    # Build outline
    outline = [OutlineSection(heading="Executive Summary", rq_refs=[])]
    heading_map: dict[str, str] = {}
    if ctx.reasoner.is_llm:
        rq_list = "\n".join(f"{rq.rq_id}: {rq.text}" for rq in rqs)
        sys_msg, user_msg = QUANT_PLANNER_HEADINGS.render(rq_list=rq_list)
        try:
            raw = ctx.reasoner.complete_text(user_msg, context=sys_msg, trace=ctx.trace)
            data = parse_json_response(raw)
            for h in data.get("headings", []):
                if h.get("rq_id") and h.get("heading"):
                    heading_map[h["rq_id"]] = h["heading"]
        except Exception:
            pass

    for rq in rqs:
        heading = heading_map.get(rq.rq_id, rq.text[:60])
        outline.append(OutlineSection(heading=heading, rq_refs=[rq.rq_id]))
    outline.append(OutlineSection(heading="Conclusion", rq_refs=[r.rq_id for r in rqs]))

    # Build coverage checklist
    checklist: list[dict] = []
    if ctx.reasoner.is_llm:
        sys_msg, user_msg = QUANT_PLANNER_BUCKETS.render(topic=query)
        try:
            raw = ctx.reasoner.complete_text(user_msg, context=sys_msg, trace=ctx.trace)
            data = parse_json_response(raw)
            buckets = data.get("buckets", [])
            if len(buckets) >= 3:
                checklist = buckets[:8]
        except Exception:
            pass

    if not checklist:
        checklist = _rule_based_quant_buckets(query, analysis_type)

    for b in checklist:
        b.setdefault("min_sources", 1)
        b.setdefault("min_claims", 2)

    plan = QuantPlanOutput(
        query=query, ticker=ticker, analysis_type=analysis_type if isinstance(analysis_type, str) else analysis_type.value,
        research_questions=rqs, outline=outline,
        acceptance_threshold=0.7, coverage_checklist=checklist,
    )
    plan_path = ctx.run_dir / "plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")


def _rule_based_quant_rqs(query: str, ticker: str, analysis_type) -> list:
    from researchops.apps.market.schemas import QuantQuestion

    at = analysis_type if isinstance(analysis_type, str) else analysis_type.value
    subject = ticker or query.split()[:3]
    subject = subject if isinstance(subject, str) else " ".join(subject)

    rqs = [
        QuantQuestion(rq_id="rq_fundamentals", text=f"What are the key financial metrics and fundamentals of {subject}?",
                      priority=1, needs_verification=True, dimension="fundamentals"),
        QuantQuestion(rq_id="rq_competitive", text=f"How does {subject} compare to its main competitors?",
                      priority=1, dimension="competitive"),
        QuantQuestion(rq_id="rq_risks", text=f"What are the primary risks facing {subject}?",
                      priority=2, dimension="risk"),
    ]
    if at in ("fundamental", "competitive"):
        rqs.append(QuantQuestion(rq_id="rq_growth", text=f"What is the growth outlook for {subject}?",
                                 priority=2, needs_verification=True, dimension="growth"))
    return rqs


def _rule_based_quant_buckets(query: str, analysis_type) -> list[dict]:
    return [
        {"bucket_id": "bkt_financials", "bucket_name": "financials", "description": f"Revenue, profit, margins for {query}"},
        {"bucket_id": "bkt_competitive", "bucket_name": "competitive", "description": f"Market position and competitors of {query}"},
        {"bucket_id": "bkt_risks", "bucket_name": "risks", "description": f"Key risks and challenges for {query}"},
        {"bucket_id": "bkt_outlook", "bucket_name": "outlook", "description": f"Growth outlook and future prospects for {query}"},
    ]


# ── Node functions (reusing agents layer) ──────────────────────────────

def plan_node(state: QuantState) -> dict[str, Any]:
    ctx = _build_ctx(state)
    for d in ["notes", "code", "code/logs", "artifacts", "downloads"]:
        (ctx.run_dir / d).mkdir(parents=True, exist_ok=True)

    _quant_plan(ctx)
    ctx.trace.log(stage="PLAN", agent="quant_planner", action="complete")

    plan_data = None
    plan_path = ctx.run_dir / "plan.json"
    if plan_path.exists():
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))

    updates = _sync_state_back(ctx, "COLLECT")
    updates["plan"] = plan_data
    updates["completed_stages"] = ["PLAN"]
    return updates


def collect_node(state: QuantState) -> dict[str, Any]:
    """Reuses CollectorAgent — same collection logic, quant config injected via context."""
    from researchops.agents.collection import CollectorAgent

    ctx = _build_ctx(state)
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


def read_node(state: QuantState) -> dict[str, Any]:
    """Reuses ReaderAgent — same extraction logic, quant context."""
    from researchops.agents.reading import ReaderAgent
    from researchops.retrieval import create_retriever
    from researchops.utils import compute_coverage, load_claim_dicts

    ctx = _build_ctx(state)

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


def verify_node(state: QuantState) -> dict[str, Any]:
    """Reuses VerifierAgent — same sandbox verification."""
    from researchops.agents.verification import VerifierAgent

    ctx = _build_ctx(state)
    VerifierAgent().execute(ctx)

    updates = _sync_state_back(ctx, "WRITE")
    completed = list(state.get("completed_stages", []))
    if "VERIFY" not in completed:
        completed.append("VERIFY")
    updates["completed_stages"] = completed
    return updates


def write_node(state: QuantState) -> dict[str, Any]:
    """Reuses WriterAgent — same writing logic with quant context."""
    from researchops.agents.writing import WriterAgent

    ctx = _build_ctx(state)
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


def qa_node(state: QuantState) -> dict[str, Any]:
    """Reuses QAAgent — same quality checks."""
    from researchops.agents.qa import QAAgent

    ctx = _build_ctx(state)
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


def supervisor_node(state: QuantState) -> dict[str, Any]:
    from researchops.core.orchestration.supervisor import Supervisor

    ctx = _build_ctx(state)
    supervisor = Supervisor(ctx.run_dir, ctx.reasoner)

    diagnostics = state.get("diagnostics", {})
    diagnostics["coverage_vector"] = ctx.state.coverage_vector

    history = list(state.get("decision_history", []))

    if len(history) >= 5:
        logger.warning("Quant supervisor: max decision loops reached (%d), forcing progression", len(history))
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
        rollback_target = "COLLECT"

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


def eval_node(state: QuantState) -> dict[str, Any]:
    from researchops.apps.market.config import MarketConfig
    from researchops.apps.market.evaluators import compute_quant_eval

    run_dir = Path(state["run_dir"])
    config = MarketConfig.model_validate(state["config"])
    result = compute_quant_eval(run_dir, config=config)

    return {"eval_result": result.model_dump(), "stage": "DONE"}


# ── Edge routing ───────────────────────────────────────────────────────

def after_read(state: QuantState) -> str:
    diagnostics = state.get("diagnostics", {})
    coverage = diagnostics.get("coverage", 1.0)
    config = state.get("config", {})
    threshold = config.get("acceptance_threshold", 0.7)
    collect_rounds = state.get("collect_rounds", 1)
    max_rounds = state.get("max_collect_rounds", 4)
    if coverage < threshold and collect_rounds < max_rounds:
        return "supervisor"
    return "verify"


def after_write(state: QuantState) -> str:
    rollback = state.get("rollback_target")
    write_rounds = state.get("write_rounds", 0)
    if write_rounds >= 2:
        return "qa"
    if rollback and rollback != "None":
        return "supervisor"
    return "qa"


def after_qa(state: QuantState) -> str:
    qa = state.get("qa_result", {})
    if qa.get("passed", False):
        return "evaluate"
    rollback = state.get("rollback_target")
    if rollback == "WRITE":
        return "write"
    return "supervisor"


def after_supervisor(state: QuantState) -> str:
    target = state.get("rollback_target", "collect")
    target_lower = target.lower() if target else "collect"
    valid = {"collect", "read", "write", "verify"}
    return target_lower if target_lower in valid else "collect"


# ── Graph builder ──────────────────────────────────────────────────────

def build_quant_graph() -> Any:
    """Construct and compile the quant research LangGraph pipeline.

    Same topology as the research pipeline — demonstrating core reusability.
    """
    from langgraph.graph import END, StateGraph

    g: StateGraph = StateGraph(QuantState)

    g.add_node("plan", plan_node)
    g.add_node("collect", collect_node)
    g.add_node("read", read_node)
    g.add_node("verify", verify_node)
    g.add_node("write", write_node)
    g.add_node("qa", qa_node)
    g.add_node("evaluate", eval_node)
    g.add_node("supervisor", supervisor_node)

    g.add_edge("plan", "collect")
    g.add_edge("collect", "read")
    g.add_conditional_edges("read", after_read, {"verify": "verify", "supervisor": "supervisor"})
    g.add_edge("verify", "write")
    g.add_conditional_edges("write", after_write, {"qa": "qa", "supervisor": "supervisor"})
    g.add_conditional_edges("qa", after_qa, {"evaluate": "evaluate", "supervisor": "supervisor", "write": "write"})
    g.add_conditional_edges("supervisor", after_supervisor, {"collect": "collect", "read": "read", "write": "write", "verify": "verify"})
    g.add_edge("evaluate", END)

    g.set_entry_point("plan")
    return g.compile()
