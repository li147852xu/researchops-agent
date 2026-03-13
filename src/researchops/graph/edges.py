"""Conditional edge functions for LangGraph routing decisions."""

from __future__ import annotations

from researchops.graph.state import ResearchState


def after_read(state: ResearchState) -> str:
    """Route after READ: check coverage, decide verify vs. supervisor."""
    diagnostics = state.get("diagnostics", {})
    coverage = diagnostics.get("coverage", 1.0)
    config = state.get("config", {})
    threshold = config.get("acceptance_threshold", 0.7)

    collect_rounds = state.get("collect_rounds", 1)
    max_rounds = state.get("max_collect_rounds", 6)

    if coverage < threshold and collect_rounds < max_rounds:
        return "supervisor"
    return "verify"


def after_write(state: ResearchState) -> str:
    """Route after WRITE: if rollback requested, go to supervisor."""
    rollback = state.get("rollback_target")
    if rollback and rollback != "None":
        return "supervisor"
    return "qa"


def after_qa(state: ResearchState) -> str:
    """Route after QA: pass -> evaluate, fail -> supervisor."""
    qa = state.get("qa_result", {})
    if qa.get("passed", False):
        return "evaluate"
    rollback = state.get("rollback_target")
    if rollback == "WRITE":
        return "write"
    return "supervisor"


def after_supervisor(state: ResearchState) -> str:
    """Route from supervisor to the decided rollback target node."""
    target = state.get("rollback_target", "collect")
    target_lower = target.lower() if target else "collect"
    valid = {"collect", "read", "write", "verify"}
    if target_lower in valid:
        return target_lower
    return "collect"
