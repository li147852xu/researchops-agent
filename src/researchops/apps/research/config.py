"""Research app configuration — extends BaseAppConfig with research-specific fields."""

from __future__ import annotations

from enum import Enum

from researchops.core.config import BaseAppConfig, RunMode


class SourceStrategy(str, Enum):
    DEMO = "demo"
    ARXIV = "arxiv"
    WEB = "web"
    HYBRID = "hybrid"


class RunConfig(BaseAppConfig):
    """Configuration for a general research run."""

    app_name: str = "research"
    sources: SourceStrategy = SourceStrategy.HYBRID  # type: ignore[assignment]
