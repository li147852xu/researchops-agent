from __future__ import annotations

from pathlib import Path

from researchops.sandbox.base import SandboxBase, SandboxResult


class DockerSandbox(SandboxBase):
    """Placeholder for Docker-based sandbox execution.

    Full implementation would use docker SDK to run scripts in
    isolated containers with cgroup limits and network namespace control.
    """

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
