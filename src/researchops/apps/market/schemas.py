"""Market Intelligence app domain models — financial questions, sources, claims, evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(UTC).isoformat()


class QuantQuestion(BaseModel):
    rq_id: str
    text: str
    priority: int = 1
    needs_verification: bool = False
    dimension: str = ""


class OutlineSection(BaseModel):
    heading: str
    rq_refs: list[str] = Field(default_factory=list)


class QuantPlanOutput(BaseModel):
    query: str
    ticker: str = ""
    analysis_type: str = "fundamental"
    topic: str = ""
    research_questions: list[QuantQuestion]
    outline: list[OutlineSection]
    acceptance_threshold: float = 0.7
    coverage_checklist: list[dict] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        if not self.topic:
            self.topic = self.query


class SourceType(str, Enum):
    HTML = "html"
    PDF = "pdf"
    TEXT = "text"
    API = "api"


class MarketSource(BaseModel):
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


class FinancialClaim(BaseModel):
    claim_id: str
    text: str
    evidence_spans: list[str] = Field(default_factory=list)
    supports_rq: list[str] = Field(default_factory=list)
    category: str = ""
    evidence_location: str = ""
    claim_type: str = ""
    polarity: str = "neutral"
    has_numerical: bool = False


class QuantSourceNotes(BaseModel):
    source_id: str
    claims: list[FinancialClaim] = Field(default_factory=list)
    contribution: str = ""
    method: str = ""
    limitations: str = ""
    bibliographic: dict = Field(default_factory=dict)
    quality: dict = Field(default_factory=dict)
    relevance_score: float = 0.0
    bucket_hits: list[str] = Field(default_factory=list)


class QuantEvalResult(BaseModel):
    citation_coverage: float = 0.0
    source_diversity: dict[str, Any] = Field(default_factory=dict)
    reproduction_rate: float = 0.0
    tool_calls: int = 0
    latency_sec: float = 0.0
    steps: int = 0
    unsupported_claim_rate: float = 0.0
    cache_hit_rate: float = 0.0
    llm_enabled: bool = False
    numerical_claim_rate: float = 0.0
    data_freshness_score: float = 0.0
    source_type_diversity: dict[str, int] = Field(default_factory=dict)
    conflict_count: int = 0
    collect_rounds: int = 1
    artifacts_count: int = 0
    bucket_coverage_rate: float = 0.0
    relevance_avg: float = 0.0
    section_nonempty_rate: float = 0.0
