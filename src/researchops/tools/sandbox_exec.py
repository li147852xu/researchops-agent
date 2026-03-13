from __future__ import annotations

from pathlib import Path

from researchops.sandbox.proc import SubprocessSandbox


def sandbox_exec(
    script_path: str,
    timeout: int = 30,
    allow_net: bool = False,
    work_dir: str = ".",
) -> dict:
    """Execute a Python script via SubprocessSandbox and return results."""
    sb = SubprocessSandbox()
    result = sb.execute(
        script_path=Path(script_path),
        work_dir=Path(work_dir),
        timeout=timeout,
        allow_net=allow_net,
    )
    return result.model_dump()
