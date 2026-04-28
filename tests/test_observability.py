"""Unit tests for the optional Langfuse observability layer.

These tests exercise the facade in isolation and as fanned-out from
:class:`TraceLogger`. They never depend on the real ``langfuse`` SDK being
installed: the no-op tests rely only on stdlib + ResearchOps; the SDK-version
dispatch test uses a stubbed ``langfuse`` module injected into ``sys.modules``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from researchops.core.observability import LangfuseFacade, get_client, reset_client
from researchops.core.observability.langfuse_client import (
    _NoopFacade,
    _NoopNode,
    _V2Facade,
    _V3Facade,
)
from researchops.core.tracing import TraceLogger


# ── Mocks ──────────────────────────────────────────────────────────────


class _RecordingNode:
    """LangfuseNode-shaped recorder for asserting the fanout dispatch table."""

    def __init__(self, recorder: list[tuple[str, dict]], name: str = "trace"):
        self._rec = recorder
        self._name = name

    def open_span(self, name, *, input="", metadata=None):
        self._rec.append(("open_span", {"name": name, "input": input,
                                          "metadata": metadata or {}}))
        return _RecordingNode(self._rec, name)

    def short_span(self, name, *, input="", output="", metadata=None):
        self._rec.append(("short_span", {"name": name, "input": input,
                                           "output": output,
                                           "metadata": metadata or {}}))

    def log_generation(self, *, name, model, input="", output="",
                       usage_input=0, usage_output=0, metadata=None):
        self._rec.append(("log_generation", {
            "name": name, "model": model, "input": input, "output": output,
            "usage_input": usage_input, "usage_output": usage_output,
            "metadata": metadata or {},
        }))

    def event(self, name, *, metadata=None):
        self._rec.append(("event", {"name": name, "metadata": metadata or {}}))

    def end(self, *, output="", error=None):
        self._rec.append(("end", {"name": self._name, "output": output, "error": error}))


class _RecordingFacade(LangfuseFacade):
    enabled = True

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def trace(self, *, id, name, metadata=None):
        self.calls.append(("trace", {"id": id, "name": name, "metadata": metadata or {}}))
        return _RecordingNode(self.calls, name)

    def flush(self):
        self.calls.append(("flush", {}))


class _ExplodingNode(_NoopNode):
    def open_span(self, name, *, input="", metadata=None):
        raise RuntimeError("BOOM in open_span")


class _ExplodingFacade(LangfuseFacade):
    enabled = True

    def trace(self, *, id, name, metadata=None):
        return _ExplodingNode()

    def flush(self):
        pass


# ── Tests ──────────────────────────────────────────────────────────────


def test_noop_facade_when_env_missing(monkeypatch):
    """No env vars -> NoopFacade; every method is harmless and returns no side effects."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    reset_client()
    client = get_client()
    assert isinstance(client, _NoopFacade)
    assert client.enabled is False
    node = client.trace(id="r1", name="r1")
    node.open_span("any").log_generation(
        name="g", model="m", usage_input=1, usage_output=2,
    )
    node.short_span("tool")
    node.event("misc")
    node.end()
    client.flush()


def test_tracelogger_works_without_langfuse(tmp_path: Path, monkeypatch):
    """TraceLogger keeps writing trace.jsonl when no LANGFUSE_* env is set."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    reset_client()

    run_dir = tmp_path / "research_test"
    trace = TraceLogger(run_dir / "trace.jsonl")
    trace.log(stage="PLAN", agent="planner", action="start", input_summary="topic")
    trace.log(stage="PLAN", agent="planner", action="complete",
              output_summary="ok")
    trace.close()

    events = trace.read_all()
    assert [e.action for e in events] == ["start", "complete"]


def test_full_run_sequence_dispatches_correctly(tmp_path: Path):
    """A real run sequence translates into the expected facade calls."""
    facade = _RecordingFacade()
    run_dir = tmp_path / "research_xyz"
    trace = TraceLogger(run_dir / "trace.jsonl", langfuse_client=facade)

    trace.log(stage="PLAN", agent="planner", action="start", input_summary="topic")
    trace.log(action="llm.call", input_summary="prompt-1",
              meta={"model": "deepseek-chat", "estimated_prompt_tokens": 320})
    trace.log(action="llm.result", output_summary="answer-1", duration_ms=1234.0,
              meta={"estimated_prompt_tokens": 320,
                    "estimated_completion_tokens": 128, "tokens": 448})
    trace.log(tool="web_search", action="invoke",
              input_summary="{'q':'foo'}", output_summary="[...]",
              duration_ms=42.0)
    trace.log(stage="PLAN", agent="planner", action="complete",
              output_summary="planner ok")
    trace.close()

    kinds = [c[0] for c in facade.calls]
    assert kinds[0] == "trace"                        # one trace per run
    assert "open_span" in kinds                       # PLAN/planner span opened
    assert "log_generation" in kinds                  # llm.call+llm.result merged
    assert "short_span" in kinds                      # tool invocation
    assert kinds.count("end") >= 1                    # span closed on complete
    assert kinds[-1] == "flush"                       # close() flushed at the end

    gen_call = next(c for c in facade.calls if c[0] == "log_generation")[1]
    assert gen_call["model"] == "deepseek-chat"
    assert gen_call["usage_input"] == 320
    assert gen_call["usage_output"] == 128
    assert gen_call["input"] == "prompt-1"
    assert gen_call["output"] == "answer-1"

    tool_call = next(c for c in facade.calls if c[0] == "short_span")[1]
    assert tool_call["name"] == "tool:web_search"
    assert tool_call["metadata"]["tool"] == "web_search"
    assert tool_call["metadata"]["cache_hit"] is False


def test_langfuse_exception_is_swallowed(tmp_path: Path):
    """An exception in the Langfuse path must not break the JSONL write."""
    run_dir = tmp_path / "research_explode"
    trace = TraceLogger(run_dir / "trace.jsonl", langfuse_client=_ExplodingFacade())
    trace.log(stage="PLAN", agent="planner", action="start")  # would raise
    trace.log(stage="PLAN", agent="planner", action="complete")

    events = trace.read_all()
    assert [e.action for e in events] == ["start", "complete"]
    trace.close()


def test_v2_v3_version_dispatch(monkeypatch):
    """SDK version probe routes to _V2Facade vs _V3Facade."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")

    constructed: dict[str, Any] = {}

    class _StubLangfuse:
        def __init__(self, **kwargs):
            constructed["kwargs"] = kwargs

        def flush(self):  # exercised by atexit at interpreter shutdown
            return None

    def install_stub(version: str) -> None:
        mod = types.ModuleType("langfuse")
        mod.__version__ = version  # type: ignore[attr-defined]
        mod.Langfuse = _StubLangfuse  # type: ignore[attr-defined]
        sys.modules["langfuse"] = mod

    try:
        install_stub("2.50.0")
        reset_client()
        c2 = get_client()
        assert isinstance(c2, _V2Facade)
        assert constructed["kwargs"]["public_key"] == "pk-lf-test"
        assert constructed["kwargs"]["secret_key"] == "sk-lf-test"
        assert constructed["kwargs"]["host"] == "http://localhost:3000"

        install_stub("3.1.0")
        reset_client()
        c3 = get_client()
        assert isinstance(c3, _V3Facade)
    finally:
        sys.modules.pop("langfuse", None)
        reset_client()


def test_get_client_caches(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    reset_client()
    a = get_client()
    b = get_client()
    assert a is b


@pytest.fixture(autouse=True)
def _clean_singleton():
    reset_client()
    yield
    reset_client()
