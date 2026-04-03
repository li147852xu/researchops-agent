"""General Research app — topic research powered by the multi-agent core."""

from researchops.apps.registry import AppSpec, register_app
from researchops.apps.research.adapters import register_research_tools
from researchops.apps.research.config import RunConfig
from researchops.apps.research.evaluators import compute_eval

RESEARCH_APP_SPEC = AppSpec(
    name="research",
    display_name="General Research",
    description="Topic research, literature surveys, technology analysis, policy briefs",
    config_class=RunConfig,
    register_tools=register_research_tools,
    custom_planner=None,
    compute_eval=compute_eval,
)

register_app(RESEARCH_APP_SPEC)
