"""AgentBase — abstract base class for all pipeline agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

from researchops.core.context import RunContext
from researchops.core.state import AgentResult


class AgentBase(ABC):
    name: str = "base"

    @abstractmethod
    def execute(self, ctx: RunContext) -> AgentResult:
        ...

    def can_retry(self, error: Exception) -> bool:
        return True
