from __future__ import annotations

from pathlib import Path

from researchops.core.sandbox.base import SandboxBase, SandboxResult


class DockerSandbox(SandboxBase):
    """Placeholder for Docker-based sandbox execution."""

    def execute(
        self,
        script_path: Path,
        work_dir: Path,
        timeout: int = 30,
        allow_net: bool = False,
        env_extra: dict[str, str] | None = None,
    ) -> SandboxResult:
        raise NotImplementedError(
            "Docker sandbox is not yet implemented. Use --sandbox subprocess."
        )
