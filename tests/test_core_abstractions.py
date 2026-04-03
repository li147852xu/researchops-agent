"""Tests for the core platform abstractions — BaseAppConfig, AppSpec, registry, pipeline."""

from __future__ import annotations

import pytest


class TestBaseAppConfig:
    def test_base_config_creation(self):
        from researchops.core.config import BaseAppConfig, RunMode

        config = BaseAppConfig(topic="test topic")
        assert config.topic == "test topic"
        assert config.mode == RunMode.FAST
        assert config.app_name == "generic"

    def test_safe_dump_redacts_api_key(self):
        from researchops.core.config import BaseAppConfig

        config = BaseAppConfig(topic="test", llm_api_key="sk-secret")
        dump = config.safe_dump()
        assert dump["llm_api_key"] == "***REDACTED***"
        assert dump["topic"] == "test"

    def test_deep_mode_properties(self):
        from researchops.core.config import BaseAppConfig, RunMode

        config = BaseAppConfig(topic="test", mode=RunMode.DEEP)
        assert config.max_collect > 3
        assert config.max_retries == 3
        assert config.max_collect_rounds == 6

    def test_fast_mode_properties(self):
        from researchops.core.config import BaseAppConfig, RunMode

        config = BaseAppConfig(topic="test", mode=RunMode.FAST)
        assert config.max_collect <= 3
        assert config.max_retries == 2
        assert config.max_collect_rounds == 3

    def test_shared_enums(self):
        from researchops.core.config import RetrievalMode, RunMode, SandboxBackend

        assert RunMode.FAST.value == "fast"
        assert RunMode.DEEP.value == "deep"
        assert SandboxBackend.SUBPROCESS.value == "subprocess"
        assert RetrievalMode.BM25.value == "bm25"
        assert RetrievalMode.HYBRID.value == "hybrid"


class TestResearchConfig:
    def test_inherits_base(self):
        from researchops.apps.research.config import RunConfig
        from researchops.core.config import BaseAppConfig

        assert issubclass(RunConfig, BaseAppConfig)

    def test_defaults(self):
        from researchops.apps.research.config import RunConfig

        config = RunConfig(topic="deep learning")
        assert config.app_name == "research"
        assert config.topic == "deep learning"
        assert config.sources.value == "hybrid"

    def test_source_strategy_enum(self):
        from researchops.apps.research.config import SourceStrategy

        assert SourceStrategy.DEMO.value == "demo"
        assert SourceStrategy.ARXIV.value == "arxiv"


class TestMarketConfig:
    def test_inherits_base(self):
        from researchops.apps.market.config import MarketConfig
        from researchops.core.config import BaseAppConfig

        assert issubclass(MarketConfig, BaseAppConfig)

    def test_defaults(self):
        from researchops.apps.market.config import MarketConfig

        config = MarketConfig(topic="NVDA competitive analysis")
        assert config.app_name == "market"
        assert config.ticker == ""
        assert config.analysis_type.value == "fundamental"

    def test_query_alias(self):
        from researchops.apps.market.config import MarketConfig

        config = MarketConfig(topic="NVDA analysis", ticker="NVDA")
        assert config.query == "NVDA analysis"

    def test_analysis_type_enum(self):
        from researchops.apps.market.config import AnalysisType

        assert AnalysisType.FUNDAMENTAL.value == "fundamental"
        assert AnalysisType.COMPETITIVE.value == "competitive"
        assert AnalysisType.RISK.value == "risk"


class TestAppRegistry:
    def test_list_apps_returns_both(self):
        from researchops.apps.registry import list_apps

        apps = list_apps()
        names = [a.name for a in apps]
        assert "research" in names
        assert "market" in names

    def test_get_app_research(self):
        from researchops.apps.registry import get_app

        spec = get_app("research")
        assert spec.name == "research"
        assert spec.display_name == "General Research"
        assert spec.config_class is not None
        assert spec.register_tools is not None

    def test_get_app_market(self):
        from researchops.apps.registry import get_app

        spec = get_app("market")
        assert spec.name == "market"
        assert spec.display_name == "Market Intelligence"
        assert spec.custom_planner is not None

    def test_get_app_unknown_raises(self):
        from researchops.apps.registry import get_app

        with pytest.raises(KeyError, match="Unknown app"):
            get_app("nonexistent_app")

    def test_app_spec_config_class_is_base_subclass(self):
        from researchops.apps.registry import list_apps
        from researchops.core.config import BaseAppConfig

        for spec in list_apps():
            assert issubclass(spec.config_class, BaseAppConfig), (
                f"{spec.name} config_class should extend BaseAppConfig"
            )


class TestPipelineState:
    def test_state_creation(self):
        from researchops.core.pipeline import PipelineState

        state: PipelineState = {
            "run_id": "test",
            "run_dir": "/tmp/test",
            "topic": "test topic",
            "config": {},
            "stage": "PLAN",
        }
        assert state["run_id"] == "test"

    def test_build_initial_state(self):
        from pathlib import Path

        from researchops.apps.research.config import RunConfig
        from researchops.core.pipeline import build_initial_state

        config = RunConfig(topic="test")
        state = build_initial_state("run_001", Path("/tmp/run_001"), "test", config)
        assert state["run_id"] == "run_001"
        assert state["topic"] == "test"
        assert state["stage"] == "PLAN"
        assert state["collect_rounds"] == 1
        assert isinstance(state["config"], dict)
