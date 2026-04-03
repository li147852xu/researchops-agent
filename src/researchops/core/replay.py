"""Replay — trace-based run replay for inspection and debugging."""

from __future__ import annotations

import json
from pathlib import Path

from researchops.core.tracing import TraceLogger

try:
    from rich.console import Console
    _console = Console()
    def _print(msg: str) -> None:
        _console.print(msg)
except ImportError:
    import re as _re
    def _print(msg: str) -> None:
        print(_re.sub(r"\[/?[a-z ]+\]", "", msg))


def replay_run(
    run_dir: Path, from_step: int = 0, no_tools: bool = False, json_output: bool = False,
) -> None:
    trace_path = run_dir / "trace.jsonl"
    if not trace_path.exists():
        _print(f"[red]No trace.jsonl found in {run_dir}[/]")
        raise SystemExit(1)
    tl = TraceLogger(trace_path)
    events = tl.read_all()

    if json_output:
        output = []
        for i, ev in enumerate(events):
            if i < from_step:
                continue
            entry: dict = {
                "step": i, "stage": ev.stage, "agent": ev.agent,
                "action": ev.action, "tool": ev.tool,
                "outcome": "error" if ev.error else "ok",
                "latency_ms": ev.duration_ms,
                "cache_hit": ev.action == "cache_hit",
            }
            if no_tools and ev.tool:
                entry["dry_run"] = True
            output.append(entry)
        print(json.dumps(output, indent=2, default=str))
        return

    _print(f"[bold]Replaying {len(events)} events from {run_dir}[/]")
    for i, ev in enumerate(events):
        if i < from_step:
            continue
        prefix = f"[{i:04d}]"
        parts = [prefix]
        if ev.stage:
            parts.append(f"[cyan]{ev.stage}[/]")
        if ev.agent:
            parts.append(f"[magenta]{ev.agent}[/]")
        parts.append(ev.action)
        if ev.tool:
            parts.append(f"tool={ev.tool}")
        if ev.error:
            parts.append(f"[red]ERR: {ev.error}[/]")
        _print(" ".join(parts))
        if ev.output_summary:
            _print(f"       → {ev.output_summary[:120]}")
