"""Market Intelligence app — financial research powered by the multi-agent core."""

from researchops.apps.market.adapters import register_market_tools
from researchops.apps.market.config import MarketConfig
from researchops.apps.market.evaluators import compute_quant_eval
from researchops.apps.registry import AppSpec, register_app


def _market_planner(ctx) -> None:
    """Custom planner for market intelligence — generates finance-focused research plan."""
    from researchops.apps.market.pipeline import _quant_plan
    _quant_plan(ctx)


MARKET_APP_SPEC = AppSpec(
    name="market",
    display_name="Market Intelligence",
    description="Company analysis, sector intelligence, competitive landscape, financial memos",
    config_class=MarketConfig,
    register_tools=register_market_tools,
    custom_planner=_market_planner,
    compute_eval=compute_quant_eval,
    extra_cli_options={
        "ticker": {"default": "", "help": "Stock ticker symbol"},
        "analysis_type": {"default": "fundamental", "help": "fundamental/competitive/risk/technical"},
    },
)

register_app(MARKET_APP_SPEC)
