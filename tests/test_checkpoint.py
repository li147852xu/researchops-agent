from __future__ import annotations

from pathlib import Path

from researchops.core.checkpoint import advance_stage, load_state, save_state, should_skip
from researchops.core.state import Stage, StateSnapshot


def test_save_and_load(tmp_run_dir: Path):
    state = StateSnapshot(run_id="test_001", stage=Stage.COLLECT, step=2)
    save_state(state, tmp_run_dir)

    loaded = load_state(tmp_run_dir)
    assert loaded is not None
    assert loaded.run_id == "test_001"
    assert loaded.stage == Stage.COLLECT
    assert loaded.step == 2


def test_load_missing(tmp_path: Path):
    assert load_state(tmp_path / "nonexistent") is None


def test_advance_stage():
    state = StateSnapshot(run_id="test_002")
    advance_stage(state, Stage.PLAN, Stage.COLLECT)
    assert Stage.PLAN in state.completed_stages
    assert state.stage == Stage.COLLECT
    assert state.step == 1


def test_should_skip():
    state = StateSnapshot(
        run_id="test_003",
        completed_stages=[Stage.PLAN, Stage.COLLECT],
    )
    assert should_skip(state, Stage.PLAN) is True
    assert should_skip(state, Stage.COLLECT) is True
    assert should_skip(state, Stage.READ) is False


def test_resume_from_mid_stage(tmp_run_dir: Path):
    state = StateSnapshot(
        run_id="resume_test",
        stage=Stage.READ,
        step=3,
        completed_stages=[Stage.PLAN, Stage.COLLECT],
    )
    save_state(state, tmp_run_dir)

    loaded = load_state(tmp_run_dir)
    assert loaded is not None
    assert loaded.stage == Stage.READ
    assert should_skip(loaded, Stage.PLAN) is True
    assert should_skip(loaded, Stage.COLLECT) is True
    assert should_skip(loaded, Stage.READ) is False
