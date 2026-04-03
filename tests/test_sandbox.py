from __future__ import annotations

from pathlib import Path

import pytest

from researchops.core.sandbox.proc import SubprocessSandbox


@pytest.fixture
def sandbox():
    return SubprocessSandbox()


def test_basic_execution(sandbox: SubprocessSandbox, tmp_run_dir: Path):
    code_dir = tmp_run_dir / "code"
    script = code_dir / "hello.py"
    script.write_text('print("hello world")', encoding="utf-8")

    result = sandbox.execute(script, code_dir, timeout=10, allow_net=True)
    assert result.exit_code == 0
    assert "hello world" in result.stdout


def test_stderr_capture(sandbox: SubprocessSandbox, tmp_run_dir: Path):
    code_dir = tmp_run_dir / "code"
    script = code_dir / "err.py"
    script.write_text('import sys; sys.stderr.write("oops\\n"); sys.exit(1)', encoding="utf-8")

    result = sandbox.execute(script, code_dir, timeout=10, allow_net=True)
    assert result.exit_code == 1
    assert "oops" in result.stderr


def test_timeout(sandbox: SubprocessSandbox, tmp_run_dir: Path):
    code_dir = tmp_run_dir / "code"
    script = code_dir / "slow.py"
    script.write_text('import time; time.sleep(60)', encoding="utf-8")

    result = sandbox.execute(script, code_dir, timeout=2, allow_net=True)
    assert result.timed_out is True
    assert result.exit_code == -1


def test_log_files_created(sandbox: SubprocessSandbox, tmp_run_dir: Path):
    code_dir = tmp_run_dir / "code"
    script = code_dir / "logtest.py"
    script.write_text('print("logged")', encoding="utf-8")

    sandbox.execute(script, code_dir, timeout=10, allow_net=True)

    logs_dir = code_dir / "logs"
    assert (logs_dir / "logtest.out").exists()
    assert (logs_dir / "logtest.err").exists()
    assert "logged" in (logs_dir / "logtest.out").read_text(encoding="utf-8")


def test_net_blocked_socket(sandbox: SubprocessSandbox, tmp_run_dir: Path):
    code_dir = tmp_run_dir / "code"
    script = code_dir / "nettest.py"
    script.write_text(
        'import socket; s = socket.socket(); s.connect(("1.1.1.1", 80))',
        encoding="utf-8",
    )

    result = sandbox.execute(script, code_dir, timeout=10, allow_net=False)
    assert result.exit_code != 0


def test_net_blocked_http_client(sandbox: SubprocessSandbox, tmp_run_dir: Path):
    """http.client should also be blocked when allow_net=False."""
    code_dir = tmp_run_dir / "code"
    script = code_dir / "httptest.py"
    script.write_text(
        'import http.client; c = http.client.HTTPConnection("example.com"); c.request("GET", "/")',
        encoding="utf-8",
    )

    result = sandbox.execute(script, code_dir, timeout=10, allow_net=False)
    assert result.exit_code != 0
    assert "blocked" in result.stderr.lower() or "oserror" in result.stderr.lower() or result.exit_code != 0
