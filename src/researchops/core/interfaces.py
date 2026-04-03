"""Core protocol definitions — architectural contracts for the ResearchOps platform."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from researchops.core.artifacts import SourceRecord


@runtime_checkable
class Agent(Protocol):
    """A discrete pipeline capability (plan, collect, read, verify, write, qa)."""

    name: str

    def execute(self, ctx: Any) -> Any: ...

    def can_retry(self, error: Exception) -> bool: ...


@runtime_checkable
class SourceAdapter(Protocol):
    """Pluggable source collection backend (arxiv, web, market data, etc.)."""

    def collect(self, ctx: Any, plan: Any) -> list[SourceRecord]: ...


@runtime_checkable
class Evaluator(Protocol):
    """Computes metrics for a completed run."""

    def evaluate(self, run_dir: Path, **kwargs: Any) -> dict[str, Any]: ...


@runtime_checkable
class Reporter(Protocol):
    """Generates a final report from evidence and plan artifacts."""

    def generate(self, run_dir: Path, **kwargs: Any) -> str: ...


@runtime_checkable
class SandboxRunner(Protocol):
    """Isolated execution environment for verification tasks."""

    def execute(
        self, *, script_path: Path, work_dir: Path, timeout: int, allow_net: bool,
    ) -> Any: ...


@runtime_checkable
class Workflow(Protocol):
    """Top-level pipeline execution contract."""

    def run(self) -> None: ...
