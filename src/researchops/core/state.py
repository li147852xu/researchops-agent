"""Core state models — stage lifecycle, run snapshots, decisions, trace events."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Stage(str, Enum):
    PLAN = "PLAN"
    COLLECT = "COLLECT"
    READ = "READ"
    VERIFY = "VERIFY"
    WRITE = "WRITE"
    QA = "QA"
    DONE = "DONE"


STAGE_ORDER: list[Stage] = [
    Stage.PLAN, Stage.COLLECT, Stage.READ,
    Stage.VERIFY, Stage.WRITE, Stage.QA, Stage.DONE,
]


class TaskRecord(BaseModel):
    task_id: str
    stage: Stage
    status: str = "pending"
    retries: int = 0
    error: str | None = None


class StateSnapshot(BaseModel):
    run_id: str
    stage: Stage = Stage.PLAN
    step: int = 0
    completed_stages: list[Stage] = Field(default_factory=list)
    tasks: list[TaskRecord] = Field(default_factory=list)
    tool_cache_keys: list[str] = Field(default_factory=list)
    retry_counts: dict[str, int] = Field(default_factory=dict)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    started_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    refinement_count: int = 0
    collect_rounds: int = 1
    sources_hash: str = ""
    claims_hash: str = ""
    coverage_vector: dict[str, int] = Field(default_factory=dict)
    rollback_history: list[dict[str, Any]] = Field(default_factory=list)
    incomplete_sections: list[str] = Field(default_factory=list)
    collect_strategy_level: int = 0
    no_progress_streak: int = 0
    write_rounds: int = 0
    bucket_coverage: dict[str, dict] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    ts: str = Field(default_factory=_now)
    stage: str = ""
    agent: str = ""
    action: str = ""
    tool: str = ""
    input_summary: str = ""
    output_summary: str = ""
    duration_ms: float = 0
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    decision_id: str = ""
    stage: str = ""
    reason_codes: list[str] = Field(default_factory=list)
    action_plan: dict[str, Any] = Field(default_factory=dict)
    suggested_queries: list[str] = Field(default_factory=list)
    suggested_neg_terms: list[str] = Field(default_factory=list)
    suggested_categories: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    policy_version: str = "1.0.0"
    ts: str = Field(default_factory=_now)


class AgentResult(BaseModel):
    success: bool = True
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    rollback_to: Stage | None = None
