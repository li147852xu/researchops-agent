from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class RunMode(str, Enum):
    FAST = "fast"
    DEEP = "deep"


class SandboxBackend(str, Enum):
    SUBPROCESS = "subprocess"
    DOCKER = "docker"


class RunConfig(BaseModel):
    topic: str
    mode: RunMode = RunMode.FAST
    checkpoint: Path | None = None
    budget: float = 10.0
    max_steps: int = 50
    allow_net: bool = True
    net_allowlist: list[str] = Field(default_factory=list)
    sandbox: SandboxBackend = SandboxBackend.SUBPROCESS
    run_dir: Path = Field(default=Path("runs"))
    llm: Literal["none", "openai", "openai_compat", "anthropic"] = "none"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_provider_label: str = ""
    llm_headers: str = ""
    seed: int = 42

    @property
    def max_collect(self) -> int:
        return 10 if self.mode == RunMode.DEEP else 3

    @property
    def max_retries(self) -> int:
        return 3 if self.mode == RunMode.DEEP else 2

    @property
    def verify_timeout(self) -> int:
        return 120 if self.mode == RunMode.DEEP else 30

    @property
    def max_refinements(self) -> int:
        return 2 if self.mode == RunMode.DEEP else 1
