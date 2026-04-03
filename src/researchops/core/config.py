"""Base app configuration — shared enums, LLM config, and computed properties.

Every app on the platform extends BaseAppConfig with domain-specific fields.
Shared infrastructure (pipeline, agents, orchestrator) works through this base.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class RunMode(str, Enum):
    FAST = "fast"
    DEEP = "deep"


class SandboxBackend(str, Enum):
    SUBPROCESS = "subprocess"
    DOCKER = "docker"


class RetrievalMode(str, Enum):
    NONE = "none"
    BM25 = "bm25"
    HYBRID = "hybrid"


class BaseAppConfig(BaseModel):
    """Configuration base for any app running on the multi-agent platform.

    App-specific configs inherit from this and add only the fields unique
    to their domain (e.g. ``ticker`` for market intelligence).
    """

    app_name: str = "generic"
    topic: str
    mode: RunMode = RunMode.FAST
    checkpoint: Path | None = None
    budget: float = 10.0
    max_steps: int = 50
    allow_net: bool = True
    net_allowlist: list[str] = Field(default_factory=list)
    sandbox: SandboxBackend = SandboxBackend.SUBPROCESS
    run_dir: Path = Field(default=Path("runs"))
    sources: str = "hybrid"
    retrieval: RetrievalMode = RetrievalMode.BM25
    embedder: str = "none"
    embedder_model: str = "all-MiniLM-L6-v2"
    use_graph: bool = True
    judge: bool = False
    llm: Literal["none", "openai", "openai_compat", "anthropic"] = "openai_compat"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_provider_label: str = ""
    llm_headers: str = ""
    seed: int = 42

    _SENSITIVE_FIELDS = frozenset({"llm_api_key", "llm_headers"})

    def safe_dump(self) -> dict:
        data = self.model_dump(mode="json")
        for field in self._SENSITIVE_FIELDS:
            if field in data and data[field]:
                data[field] = "***REDACTED***"
        return data

    @property
    def max_collect(self) -> int:
        return 10 if self.mode == RunMode.DEEP else 3

    @property
    def max_retries(self) -> int:
        return 3 if self.mode == RunMode.DEEP else 2

    @property
    def verify_timeout(self) -> int:
        return 120 if self.mode == RunMode.DEEP else 30

    @property
    def max_refinements(self) -> int:
        return 2 if self.mode == RunMode.DEEP else 1

    @property
    def min_sources_per_rq(self) -> int:
        return 3 if self.mode == RunMode.DEEP else 1

    @property
    def min_claims_per_rq(self) -> int:
        return 5 if self.mode == RunMode.DEEP else 2

    @property
    def max_collect_rounds(self) -> int:
        return 6 if self.mode == RunMode.DEEP else 3

    @property
    def target_sources_per_rq(self) -> int:
        return 6 if self.mode == RunMode.DEEP else 2

    @property
    def max_total_sources(self) -> int:
        return 40 if self.mode == RunMode.DEEP else 12

    @property
    def target_claims_per_rq(self) -> int:
        return 10 if self.mode == RunMode.DEEP else 5

    @property
    def target_sources_per_bucket(self) -> int:
        return 3 if self.mode == RunMode.DEEP else 1

    @property
    def relevance_threshold(self) -> float:
        return 0.5 if self.mode == RunMode.DEEP else 0.3

    @property
    def bucket_coverage_threshold(self) -> float:
        return 0.8 if self.mode == RunMode.DEEP else 0.6
