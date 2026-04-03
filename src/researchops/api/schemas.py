"""FastAPI request/response schemas — Pydantic v2 models for the unified REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "openai_compat"
    model: str = ""
    base_url: str = ""
    api_key: str = ""


class RunRequest(BaseModel):
    """Generic run request — works for any registered app."""
    topic: str
    mode: str = "fast"
    sources: str = "hybrid"
    llm: LLMConfig | None = Field(default_factory=LLMConfig)
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="App-specific fields (e.g. ticker, analysis_type for market)",
    )


class AppInfo(BaseModel):
    name: str
    display_name: str
    description: str


class RunStatus(BaseModel):
    run_id: str
    app_name: str
    status: str = "pending"
    stage: str = ""
    run_dir: str = ""


class RunSummary(BaseModel):
    run_id: str
    app_name: str
    status: str
    topic: str = ""
    created_at: str = ""
    eval_summary: dict[str, Any] = Field(default_factory=dict)


class ArtifactResponse(BaseModel):
    run_id: str
    artifact_type: str
    content: Any = None
    error: str = ""


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = ""
