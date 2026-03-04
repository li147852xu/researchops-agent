from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    for sub in ["notes", "code", "code/logs", "artifacts"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir
