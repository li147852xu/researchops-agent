"""Market Intelligence app configuration — extends BaseAppConfig with finance-specific fields."""

from __future__ import annotations

from enum import Enum

from researchops.core.config import BaseAppConfig, RunMode


class AnalysisType(str, Enum):
    FUNDAMENTAL = "fundamental"
    COMPETITIVE = "competitive"
    RISK = "risk"
    TECHNICAL = "technical"


class MarketSourceStrategy(str, Enum):
    WEB = "web"
    HYBRID = "hybrid"


class MarketConfig(BaseAppConfig):
    """Configuration for a market intelligence research run."""

    app_name: str = "market"
    ticker: str = ""
    analysis_type: AnalysisType = AnalysisType.FUNDAMENTAL
    sources: MarketSourceStrategy = MarketSourceStrategy.HYBRID  # type: ignore[assignment]

    @property
    def query(self) -> str:
        """Backward-compat alias — market pipeline uses 'query' in some places."""
        return self.topic

    @property
    def max_collect_rounds(self) -> int:
        return 4 if self.mode == RunMode.DEEP else 2

    @property
    def max_collect(self) -> int:
        return 8 if self.mode == RunMode.DEEP else 3

    @property
    def target_sources_per_rq(self) -> int:
        return 5 if self.mode == RunMode.DEEP else 2

    @property
    def max_total_sources(self) -> int:
        return 30 if self.mode == RunMode.DEEP else 10

    @property
    def target_claims_per_rq(self) -> int:
        return 8 if self.mode == RunMode.DEEP else 4

    @property
    def target_sources_per_bucket(self) -> int:
        return 2 if self.mode == RunMode.DEEP else 1

    @property
    def relevance_threshold(self) -> float:
        return 0.4

    @property
    def bucket_coverage_threshold(self) -> float:
        return 0.7 if self.mode == RunMode.DEEP else 0.5
