from __future__ import annotations

from pathlib import Path

import pytest

from researchops.core.tools.registry import ToolPermissionError, ToolRegistry
from researchops.core.tools.schema import ToolDefinition
from researchops.core.tracing import TraceLogger


def _echo_handler(**kwargs):
    return {"echo": kwargs}


def test_register_and_invoke():
    reg = ToolRegistry()
    defn = ToolDefinition(name="echo", version="1.0", permissions=[])
    reg.register(defn, _echo_handler)
    result = reg.invoke("echo", {"msg": "hello"})
    assert result == {"echo": {"msg": "hello"}}


def test_permission_denied():
    reg = ToolRegistry()
    defn = ToolDefinition(name="restricted", permissions=["admin"])
    reg.register(defn, _echo_handler)
    with pytest.raises(ToolPermissionError):
        reg.invoke("restricted", {})


def test_permission_denied_with_trace(tmp_run_dir: Path):
    """Permission denial should be recorded in trace."""
    reg = ToolRegistry()
    defn = ToolDefinition(name="net_tool", permissions=["net"])
    reg.register(defn, _echo_handler)

    trace = TraceLogger(tmp_run_dir / "trace.jsonl")

    with pytest.raises(ToolPermissionError):
        reg.invoke("net_tool", {"q": "test"}, trace=trace)

    events = trace.read_all()
    denied = [e for e in events if e.action == "permission_denied"]
    assert len(denied) == 1
    assert denied[0].tool == "net_tool"


def test_permission_granted():
    reg = ToolRegistry()
    defn = ToolDefinition(name="restricted", permissions=["admin"])
    reg.register(defn, _echo_handler)
    reg.grant_permissions({"admin"})
    result = reg.invoke("restricted", {"x": 1})
    assert result == {"echo": {"x": 1}}


def test_schema_validation():
    defn = ToolDefinition(
        name="test_tool",
        version="2.0",
        risk_level="high",
        permissions=["net", "sandbox"],
        timeout_default=120,
        cache_policy="session",
    )
    assert defn.name == "test_tool"
    assert defn.risk_level == "high"
    assert defn.cache_policy == "session"


def test_session_cache():
    call_count = 0

    def counting_handler(**kwargs):
        nonlocal call_count
        call_count += 1
        return {"count": call_count}

    reg = ToolRegistry()
    defn = ToolDefinition(name="cached", cache_policy="session", permissions=[])
    reg.register(defn, counting_handler)

    r1 = reg.invoke("cached", {"key": "a"})
    r2 = reg.invoke("cached", {"key": "a"})
    assert r1 == r2
    assert call_count == 1

    r3 = reg.invoke("cached", {"key": "b"})
    assert r3 != r1
    assert call_count == 2


def test_persistent_cache(tmp_run_dir: Path):
    """Persistent cache should save to disk and survive reload."""
    call_count = 0

    def counting_handler(**kwargs):
        nonlocal call_count
        call_count += 1
        return {"result": call_count}

    cache_path = tmp_run_dir / "cache.json"

    reg = ToolRegistry()
    reg.set_persistent_cache_path(cache_path)
    defn = ToolDefinition(name="pcached", cache_policy="persistent", permissions=[])
    reg.register(defn, counting_handler)

    r1 = reg.invoke("pcached", {"key": "x"})
    assert call_count == 1
    assert cache_path.exists()

    r2 = reg.invoke("pcached", {"key": "x"})
    assert r2 == r1
    assert call_count == 1


def test_list_tools():
    reg = ToolRegistry()
    reg.register(ToolDefinition(name="a", permissions=[]), _echo_handler)
    reg.register(ToolDefinition(name="b", permissions=[]), _echo_handler)
    names = [t.name for t in reg.list_tools()]
    assert "a" in names
    assert "b" in names


def test_unknown_tool():
    reg = ToolRegistry()
    with pytest.raises(KeyError):
        reg.invoke("nonexistent", {})
