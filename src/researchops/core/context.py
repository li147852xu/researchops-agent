"""RunContext — shared execution context passed through agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from researchops.core.state import StateSnapshot

if TYPE_CHECKING:
    from researchops.core.sandbox.base import SandboxBase
    from researchops.core.tools.registry import ToolRegistry
    from researchops.core.tracing import TraceLogger
    from researchops.reasoning.base import ReasonerBase


@dataclass
class RunContext:
    run_dir: Path
    config: object
    state: StateSnapshot
    registry: ToolRegistry
    trace: TraceLogger
    sandbox: SandboxBase
    reasoner: ReasonerBase
    shared: dict = field(default_factory=dict)
