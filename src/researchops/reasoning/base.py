from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from researchops.core.tracing import TraceLogger


class ReasonerBase(ABC):
    def __init__(self) -> None:
        self.token_count: int = 0
        self.total_latency_ms: float = 0.0
        self.call_count: int = 0

    @abstractmethod
    def complete_json(
        self,
        schema: type[BaseModel],
        prompt: str,
        *,
        context: str = "",
        trace: TraceLogger | None = None,
    ) -> BaseModel:
        ...

    @abstractmethod
    def complete_text(
        self,
        prompt: str,
        *,
        context: str = "",
        trace: TraceLogger | None = None,
    ) -> str:
        ...

    @property
    def is_llm(self) -> bool:
        return False

    def _record_call(self, tokens: int, latency_ms: float) -> None:
        self.token_count += tokens
        self.total_latency_ms += latency_ms
        self.call_count += 1

    def get_stats(self) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "call_count": self.call_count,
        }
