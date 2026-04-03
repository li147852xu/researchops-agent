"""Checkpoint — persist, reload, and advance pipeline state."""

from __future__ import annotations

import json
from pathlib import Path

from researchops.core.state import Stage, StateSnapshot, _now


def save_state(state: StateSnapshot, run_dir: Path) -> Path:
    state.updated_at = _now()
    path = run_dir / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_state(run_dir: Path) -> StateSnapshot | None:
    path = run_dir / "state.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return StateSnapshot.model_validate(data)


def advance_stage(state: StateSnapshot, completed: Stage, next_stage: Stage) -> None:
    if completed not in state.completed_stages:
        state.completed_stages.append(completed)
    state.stage = next_stage
    state.step += 1


def should_skip(state: StateSnapshot, stage: Stage) -> bool:
    return stage in state.completed_stages
