from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel


class SandboxResult(BaseModel):
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    resource_limited: bool = False


class SandboxBase(ABC):
    @abstractmethod
    def execute(
        self,
        script_path: Path,
        work_dir: Path,
        timeout: int = 30,
        allow_net: bool = False,
        env_extra: dict[str, str] | None = None,
    ) -> SandboxResult: ...
