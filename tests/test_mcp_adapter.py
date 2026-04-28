"""Unit tests for the ResearchOps MCP adapter.

Focus is on the schema-conversion layer (which is pure Python with no MCP SDK
dependency) plus a couple of round-trip tests that gracefully skip when the
optional ``mcp`` extra is not installed. This keeps the existing test suite
green even without ``pip install -e ".[mcp]"``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from researchops.core.tools.registry import ToolRegistry
from researchops.core.tools.schema import ToolDefinition
from researchops.mcp.adapter import (
    _build_description,
    shorthand_to_json_schema,
)

# ── Pure schema conversion (no mcp dep) ────────────────────────────────

def test_shorthand_to_json_schema_primitives():
    """Every primitive type shorthand maps to the right JSON Schema type and is required."""
    schema = shorthand_to_json_schema(
        {"q": "str", "n": "int", "ratio": "float", "flag": "bool"}
    )
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["properties"] == {
        "q": {"type": "string"},
        "n": {"type": "integer"},
        "ratio": {"type": "number"},
        "flag": {"type": "boolean"},
    }
    assert set(schema["required"]) == {"q", "n", "ratio", "flag"}


def test_shorthand_to_json_schema_list_and_dict():
    """Container shorthands (list[dict], dict) map to array/object."""
    schema = shorthand_to_json_schema(
        {"items": "list[dict]", "tags": "list[str]", "extra": "dict"}
    )
    assert schema["properties"]["items"] == {
        "type": "array",
        "items": {"type": "object"},
    }
    assert schema["properties"]["tags"] == {
        "type": "array",
        "items": {"type": "string"},
    }
    assert schema["properties"]["extra"] == {"type": "object"}
    assert set(schema["required"]) == {"items", "tags", "extra"}


def test_shorthand_to_json_schema_empty_and_unknown():
    """Empty input yields empty properties; unknown types fall back to no type."""
    empty = shorthand_to_json_schema({})
    assert empty["type"] == "object"
    assert empty["properties"] == {}
    assert "required" not in empty

    none_schema = shorthand_to_json_schema(None)
    assert none_schema["properties"] == {}

    unknown = shorthand_to_json_schema({"weird": "SomeCustomType"})
    # Unknown shorthand → no type constraint (so client can still pass anything).
    assert unknown["properties"]["weird"] == {}
    assert unknown["required"] == ["weird"]


def test_real_tool_definition_schema_round_trip():
    """A real ResearchOps ToolDefinition produces a usable JSON Schema object."""
    defn = ToolDefinition(
        name="web_search",
        description="Search the web",
        input_schema={"query": "str", "max_results": "int"},
        output_schema={"results": "list[dict]"},
        risk_level="medium",
        permissions=["net"],
        cache_policy="session",
    )
    schema = shorthand_to_json_schema(defn.input_schema)
    assert schema["properties"] == {
        "query": {"type": "string"},
        "max_results": {"type": "integer"},
    }
    assert set(schema["required"]) == {"query", "max_results"}

    # Description carries risk + permissions so MCP clients can see governance metadata.
    desc = _build_description(defn)
    assert "Search the web" in desc
    assert "risk=medium" in desc
    assert "perms=net" in desc
    assert "cache=session" in desc


# ── MCP SDK round-trip (skipped if extra missing) ──────────────────────

def test_tool_definition_to_mcp_when_sdk_present():
    """If the mcp extra is installed, ToolDefinition → mcp.types.Tool works."""
    pytest.importorskip("mcp")
    from researchops.mcp.adapter import tool_definition_to_mcp

    defn = ToolDefinition(
        name="web_search",
        description="Search the web",
        input_schema={"query": "str", "max_results": "int"},
        permissions=["net"],
        cache_policy="session",
    )
    mcp_tool = tool_definition_to_mcp(defn)
    assert mcp_tool.name == "web_search"
    assert "Search the web" in (mcp_tool.description or "")
    assert mcp_tool.inputSchema["properties"]["query"] == {"type": "string"}
    assert "query" in mcp_tool.inputSchema["required"]


def test_register_tools_on_server_dispatches_through_registry():
    """Calling the MCP-wrapped tool flows through ToolRegistry.invoke and returns JSON text."""
    pytest.importorskip("mcp")
    from mcp.server import Server

    from researchops.mcp.adapter import register_tools_on_server

    captured: dict = {}

    def echo_handler(**kwargs):
        captured.update(kwargs)
        return {"echoed": kwargs}

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo input back as JSON",
            input_schema={"msg": "str"},
            permissions=[],
        ),
        echo_handler,
    )

    server: Server = Server("test-researchops")
    register_tools_on_server(server, registry)

    # Pull the registered handlers out of the Server's request_handlers map.
    from mcp.types import CallToolRequest, ListToolsRequest

    list_handler = server.request_handlers[ListToolsRequest]
    call_handler = server.request_handlers[CallToolRequest]

    list_req = ListToolsRequest(method="tools/list", params=None)
    list_result = asyncio.run(list_handler(list_req))
    tool_names = [t.name for t in list_result.root.tools]
    assert "echo" in tool_names

    call_req = CallToolRequest(
        method="tools/call",
        params={"name": "echo", "arguments": {"msg": "hello"}},
    )
    call_result = asyncio.run(call_handler(call_req))
    contents = call_result.root.content
    assert contents and contents[0].type == "text"
    payload = json.loads(contents[0].text)
    assert payload == {"echoed": {"msg": "hello"}}
    assert captured == {"msg": "hello"}


# ── Resource layer (no SDK needed) ─────────────────────────────────────

def test_list_run_resources_filters_existing_files(tmp_path):
    """Only the three exposed filenames are surfaced, and only if they exist."""
    from researchops.mcp.resources import list_run_resources

    runs = tmp_path / "runs"
    full = runs / "research_abc123"
    full.mkdir(parents=True)
    (full / "plan.json").write_text("{}", encoding="utf-8")
    (full / "sources.jsonl").write_text('{"a":1}\n', encoding="utf-8")
    (full / "report.md").write_text("# r", encoding="utf-8")
    (full / "ignored.txt").write_text("x", encoding="utf-8")

    partial = runs / "market_xyz"
    partial.mkdir(parents=True)
    (partial / "plan.json").write_text("{}", encoding="utf-8")

    items = list_run_resources(runs)
    uris = {item["uri"] for item in items}
    assert "researchops://runs/research_abc123/plan.json" in uris
    assert "researchops://runs/research_abc123/sources.jsonl" in uris
    assert "researchops://runs/research_abc123/report.md" in uris
    assert "researchops://runs/market_xyz/plan.json" in uris
    # Partial run only contributed its plan.json.
    assert "researchops://runs/market_xyz/sources.jsonl" not in uris
    # Non-exposed files are never listed.
    assert not any(uri.endswith("ignored.txt") for uri in uris)


def test_read_run_resource_rejects_traversal_and_unknown(tmp_path):
    from researchops.mcp.resources import read_run_resource

    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "r1").mkdir()
    (runs / "r1" / "plan.json").write_text('{"ok": true}', encoding="utf-8")

    assert read_run_resource(runs, "researchops://runs/r1/plan.json") == '{"ok": true}'

    with pytest.raises(ValueError):
        read_run_resource(runs, "https://example.com/plan.json")
    with pytest.raises(ValueError):
        read_run_resource(runs, "researchops://runs/r1/secrets.env")
    with pytest.raises(FileNotFoundError):
        read_run_resource(runs, "researchops://runs/r2/plan.json")
