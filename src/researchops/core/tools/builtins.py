"""Core tool implementations — cite and sandbox_exec."""

from __future__ import annotations

from pathlib import Path

from researchops.core.sandbox.proc import SubprocessSandbox

_COUNTER: dict[str, int] = {}


def cite(source_id: str = "", claim_id: str = "") -> dict:
    """Map source_id / claim_id to a citation marker like [1], [2], etc."""
    key = source_id or claim_id
    if not key:
        return {"marker": ""}
    if key not in _COUNTER:
        _COUNTER[key] = len(_COUNTER) + 1
    return {"marker": f"[{_COUNTER[key]}]"}


def reset_citations() -> None:
    _COUNTER.clear()


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
