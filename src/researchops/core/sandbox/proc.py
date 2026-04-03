from __future__ import annotations

import os
import platform
import subprocess
import sys
import textwrap
from pathlib import Path

from researchops.core.sandbox.base import SandboxBase, SandboxResult

_NETBLOCK_PREAMBLE = textwrap.dedent("""\
    # ── network blocking preamble (injected by ResearchOps sandbox) ──
    import os as _os
    if _os.environ.get("RESEARCHOPS_NO_NET") == "1":
        import socket as _socket
        _orig_connect = _socket.socket.connect
        def _blocked_connect(self, *a, **kw):
            raise OSError("Network access blocked by ResearchOps sandbox")
        _socket.socket.connect = _blocked_connect

        try:
            import urllib.request
            _orig_urlopen = urllib.request.urlopen
            def _blocked_urlopen(*a, **kw):
                raise OSError("Network access blocked by ResearchOps sandbox")
            urllib.request.urlopen = _blocked_urlopen
        except ImportError:
            pass

        try:
            import http.client as _hc
            _orig_request = _hc.HTTPConnection.request
            def _blocked_request(self, *a, **kw):
                raise OSError("Network access blocked by ResearchOps sandbox")
            _hc.HTTPConnection.request = _blocked_request
            _hc.HTTPSConnection.request = _blocked_request
        except Exception:
            pass

        try:
            import requests as _req
            _orig_get = _req.Session.send
            def _blocked_send(self, *a, **kw):
                raise OSError("Network access blocked by ResearchOps sandbox")
            _req.Session.send = _blocked_send
        except ImportError:
            pass

        try:
            import httpx as _hx
            _orig_hx_send = _hx.Client.send
            def _blocked_hx_send(self, *a, **kw):
                raise OSError("Network access blocked by ResearchOps sandbox")
            _hx.Client.send = _blocked_hx_send
        except ImportError:
            pass
    # ── end preamble ──
""")


def _make_preexec(memory_mb: int = 512, cpu_sec: int = 60):
    if platform.system() != "Linux":
        return None

    def _set_limits():
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec))

    return _set_limits


class SubprocessSandbox(SandboxBase):
    def execute(
        self,
        script_path: Path,
        work_dir: Path,
        timeout: int = 30,
        allow_net: bool = False,
        env_extra: dict[str, str] | None = None,
    ) -> SandboxResult:
        work_dir = work_dir.resolve()
        script_path = script_path.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = work_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        step_name = script_path.stem

        effective_script = script_path
        if not allow_net:
            wrapped = work_dir / f"_wrapped_{script_path.name}"
            original_code = script_path.read_text(encoding="utf-8")
            wrapped.write_text(_NETBLOCK_PREAMBLE + original_code, encoding="utf-8")
            effective_script = wrapped

        env = os.environ.copy()
        env["RESEARCHOPS_SANDBOX"] = "1"
        if not allow_net:
            env["RESEARCHOPS_NO_NET"] = "1"
        if env_extra:
            env.update(env_extra)

        preexec = _make_preexec()
        resource_note = ""
        if platform.system() != "Linux":
            resource_note = f"resource_limits=unsupported on {platform.system()}"

        timed_out = False
        try:
            proc = subprocess.run(
                [sys.executable, str(effective_script)],
                cwd=str(work_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=preexec,
            )
            exit_code = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = -1
            stdout = exc.stdout or "" if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
            stderr = exc.stderr or "" if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")

        stdout_path = logs_dir / f"{step_name}.out"
        stderr_path = logs_dir / f"{step_name}.err"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        return SandboxResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            resource_limited=bool(resource_note),
        )
