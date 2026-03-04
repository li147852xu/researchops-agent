from __future__ import annotations

import json
from pathlib import Path

from researchops.trace import TraceLogger


def test_emit_and_read(tmp_run_dir: Path):
    trace_path = tmp_run_dir / "trace.jsonl"
    logger = TraceLogger(trace_path)

    logger.log(stage="PLAN", agent="planner", action="start")
    logger.log(stage="PLAN", agent="planner", action="complete", output_summary="3 RQs")

    events = logger.read_all()
    assert len(events) == 2
    assert events[0].stage == "PLAN"
    assert events[0].action == "start"
    assert events[1].output_summary == "3 RQs"


def test_jsonl_format(tmp_run_dir: Path):
    trace_path = tmp_run_dir / "trace.jsonl"
    logger = TraceLogger(trace_path)
    logger.log(stage="COLLECT", action="fetch", tool="web_search")

    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["stage"] == "COLLECT"
    assert data["tool"] == "web_search"


def test_timed_context(tmp_run_dir: Path):
    trace_path = tmp_run_dir / "trace.jsonl"
    logger = TraceLogger(trace_path)

    with logger.timed(stage="TEST", action="timed_op"):
        sum(range(1000))  # noqa: F841

    events = logger.read_all()
    assert len(events) == 1
    assert events[0].duration_ms > 0
    assert events[0].error is None


def test_read_empty(tmp_run_dir: Path):
    trace_path = tmp_run_dir / "trace_empty.jsonl"
    logger = TraceLogger(trace_path)
    assert logger.read_all() == []
