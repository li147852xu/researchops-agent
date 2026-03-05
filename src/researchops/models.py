from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ── Plan ──────────────────────────────────────────────────────────────

class ResearchQuestion(BaseModel):
    rq_id: str
    text: str
    priority: int = 1
    needs_verification: bool = False


class OutlineSection(BaseModel):
    heading: str
    rq_refs: list[str] = Field(default_factory=list)


class PlanOutput(BaseModel):
    topic: str
    research_questions: list[ResearchQuestion]
    outline: list[OutlineSection]
    acceptance_threshold: float = 0.7
    coverage_checklist: list[dict] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)


# ── Source ────────────────────────────────────────────────────────────

class SourceType(str, Enum):
    HTML = "html"
    PDF = "pdf"
    TEXT = "text"
    API = "api"


class Source(BaseModel):
    source_id: str
    type: SourceType
    url: str = ""
    domain: str = ""
    title: str = ""
    retrieved_at: str = Field(default_factory=_now)
    local_path: str = ""
    hash: str = ""
    source_type_detail: str = ""
    collect_round: int = 1
    query_id: str = ""


# ── Claims / Notes ───────────────────────────────────────────────────

class Claim(BaseModel):
    claim_id: str
    text: str
    evidence_spans: list[str] = Field(default_factory=list)
    supports_rq: list[str] = Field(default_factory=list)
    category: str = ""
    evidence_location: str = ""
    claim_type: str = ""
    polarity: str = "neutral"


class SourceNotes(BaseModel):
    source_id: str
    claims: list[Claim] = Field(default_factory=list)
    contribution: str = ""
    method: str = ""
    limitations: str = ""
    bibliographic: dict = Field(default_factory=dict)
    quality: dict = Field(default_factory=dict)
    relevance_score: float = 0.0
    bucket_hits: list[str] = Field(default_factory=list)


# ── State / Checkpoint ───────────────────────────────────────────────

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


# ── Trace ─────────────────────────────────────────────────────────────

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


# ── Decision (Supervisor) ────────────────────────────────────────────

class Decision(BaseModel):
    decision_id: str = ""
    stage: str = ""
    reason_codes: list[str] = Field(default_factory=list)
    action_plan: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    policy_version: str = "1.1.0"
    ts: str = Field(default_factory=_now)


# ── Eval ──────────────────────────────────────────────────────────────

class EvalResult(BaseModel):
    citation_coverage: float = 0.0
    source_diversity: dict[str, Any] = Field(default_factory=dict)
    reproduction_rate: float = 0.0
    tool_calls: int = 0
    latency_sec: float = 0.0
    steps: int = 0
    unsupported_claim_rate: float = 0.0
    cache_hit_rate: float = 0.0
    llm_enabled: bool = False
    estimated_tokens: int = 0
    estimated_cost_usd: float = 0.0
    estimate_method: str = "none"
    conflict_count: int = 0
    plan_refinement_count: int = 0
    collect_rounds: int = 1
    artifacts_count: int = 0
    llm_provider_label: str = ""
    papers_per_rq: float = 0.0
    low_quality_source_rate: float = 0.0
    section_nonempty_rate: float = 0.0
    incomplete_sections: list[str] = Field(default_factory=list)
    collect_rounds_total: int = 1
    sources_per_rq: dict[str, int] = Field(default_factory=dict)
    max_rollback_used: bool = False
    bucket_coverage_rate: float = 0.0
    relevance_avg: float = 0.0
    decision_count: int = 0


# ── Agent Result ──────────────────────────────────────────────────────

class AgentResult(BaseModel):
    success: bool = True
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    rollback_to: Stage | None = None
