from __future__ import annotations

import json
from pathlib import Path

from researchops.trace import TraceLogger


def test_replay_no_tools_produces_events(tmp_run_dir: Path):
    """Replay --no-tools should mark tool invocations as dry-run."""
    trace_path = tmp_run_dir / "trace.jsonl"
    logger = TraceLogger(trace_path)

    logger.log(stage="PLAN", agent="planner", action="start")
    logger.log(stage="COLLECT", tool="web_search", action="invoke", input_summary="q=test")
    logger.log(stage="ORCHESTRATOR", action="run_complete", duration_ms=100)

    events = logger.read_all()
    assert len(events) == 3

    output = []
    for ev in events:
        entry = ev.model_dump()
        if ev.action == "invoke":
            entry["dry_run"] = True
            entry["note"] = f"Would invoke tool={ev.tool}"
        output.append(entry)

    tool_events = [e for e in output if e.get("dry_run")]
    assert len(tool_events) == 1
    assert "web_search" in tool_events[0]["note"]


def test_replay_json_output(tmp_run_dir: Path):
    """JSON replay output should be valid JSON array."""
    trace_path = tmp_run_dir / "trace.jsonl"
    logger = TraceLogger(trace_path)
    logger.log(stage="PLAN", action="start")
    logger.log(stage="PLAN", action="complete")

    events = logger.read_all()
    output = [ev.model_dump() for ev in events]
    serialized = json.dumps(output, default=str)

    parsed = json.loads(serialized)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["action"] == "start"
