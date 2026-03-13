from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from researchops.models import AgentResult, StateSnapshot

if TYPE_CHECKING:
    from researchops.config import RunConfig
    from researchops.reasoning.base import ReasonerBase
    from researchops.registry.manager import ToolRegistry
    from researchops.sandbox.base import SandboxBase
    from researchops.trace import TraceLogger


@dataclass
class RunContext:
    run_dir: Path
    config: RunConfig
    state: StateSnapshot
    registry: ToolRegistry
    trace: TraceLogger
    sandbox: SandboxBase
    reasoner: ReasonerBase
    shared: dict = field(default_factory=dict)


class AgentBase(ABC):
    name: str = "base"

    @abstractmethod
    def execute(self, ctx: RunContext) -> AgentResult:
        ...

    def can_retry(self, error: Exception) -> bool:
        return True
