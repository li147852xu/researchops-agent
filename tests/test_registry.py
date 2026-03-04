from __future__ import annotations

import pytest

from researchops.registry.manager import ToolPermissionError, ToolRegistry
from researchops.registry.schema import ToolDefinition


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
