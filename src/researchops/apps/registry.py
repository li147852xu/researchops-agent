"""App registry — discover, register, and retrieve app specifications.

Each app on the platform is described by an ``AppSpec``.  The interface layer
(CLI, Web UI, API) queries the registry to enumerate available apps and build
pipelines dynamically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from researchops.core.config import BaseAppConfig
from researchops.core.tools.registry import ToolRegistry


@dataclass
class AppSpec:
    """Declarative specification for a thin app layer.

    Attributes:
        name: Machine-readable identifier (used in CLI ``--app`` flag).
        display_name: Human-readable name shown in the UI.
        description: One-line description of this app.
        config_class: Pydantic model that extends :class:`BaseAppConfig`.
        register_tools: Callable that populates a :class:`ToolRegistry`.
        custom_planner: Optional override for the default PlannerAgent.
                        Signature: ``(ctx: RunContext) -> None``.
        compute_eval: Optional evaluation function.
                      Signature: ``(run_dir, *, config) -> BaseModel``.
        extra_cli_options: Additional Typer Option kwargs surfaced in ``run``.
    """

    name: str
    display_name: str
    description: str
    config_class: type[BaseAppConfig]
    register_tools: Callable[[ToolRegistry], None]
    custom_planner: Callable | None = None
    compute_eval: Callable | None = None
    extra_cli_options: dict[str, Any] = field(default_factory=dict)


# ── Global registry ────────────────────────────────────────────────────

_REGISTRY: dict[str, AppSpec] = {}


def register_app(spec: AppSpec) -> None:
    """Register an app spec.  Overwrites if already registered."""
    _REGISTRY[spec.name] = spec


def get_app(name: str) -> AppSpec:
    """Retrieve an app spec by name.  Raises ``KeyError`` if unknown."""
    _ensure_loaded()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown app {name!r}. Available: {available}")
    return _REGISTRY[name]


def list_apps() -> list[AppSpec]:
    """Return all registered app specs, ordered by name."""
    _ensure_loaded()
    return sorted(_REGISTRY.values(), key=lambda s: s.name)


def _ensure_loaded() -> None:
    """Trigger discovery of built-in apps on first access."""
    if _REGISTRY:
        return
    import researchops.apps.research  # noqa: F401
    import researchops.apps.market  # noqa: F401
