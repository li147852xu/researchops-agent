"""Typed graph state shared across all LangGraph nodes."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class WorkingMemory(TypedDict, total=False):
    """Typed sub-state for inter-agent data (replaces ctx.shared dict)."""
    retriever: Any
    low_quality_sources: list[str]
    rq_claim_counts: dict[str, int]
    rq_source_counts: dict[str, int]
    evidence_limited: bool
    has_conflicts: bool
    conflict_count: int


class ResearchState(TypedDict, total=False):
    """Top-level state flowing through the LangGraph research pipeline."""

    # --- Identifiers ---
    run_id: str
    run_dir: str
    topic: str

    # --- Configuration (serialised) ---
    config: dict[str, Any]

    # --- Pipeline artefacts ---
    plan: dict[str, Any] | None
    sources: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    report: str
    qa_result: dict[str, Any] | None
    eval_result: dict[str, Any] | None

    # --- Diagnostics for supervisor routing ---
    diagnostics: dict[str, Any]

    # --- Control flow ---
    stage: str
    rollback_target: str | None
    collect_rounds: int
    max_collect_rounds: int
    write_rounds: int
    refinement_count: int
    decision_history: list[dict[str, Any]]
    completed_stages: list[str]

    # --- Working memory (inter-agent shared data) ---
    memory: WorkingMemory

    # --- Error tracking ---
    last_error: str | None
