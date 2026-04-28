"""Adapter from ResearchOps :class:`ToolRegistry` to an MCP server.

This module is the single bridge between the existing tool protocol
(:class:`researchops.core.tools.schema.ToolDefinition`) and the Model Context
Protocol. It does **not** modify the core tool registry; it only reads tool
definitions and forwards calls to ``ToolRegistry.invoke``.

Two public layers
~~~~~~~~~~~~~~~~~

1. :func:`shorthand_to_json_schema` — pure helper that promotes the existing
   ``{"query": "str", "max_results": "int"}`` shorthand into a real JSON
   Schema object. No dependency on the ``mcp`` SDK, so the conversion is
   independently testable.
2. :func:`tool_definition_to_mcp` and :func:`register_tools_on_server` — wire
   each :class:`ToolDefinition` into an ``mcp.server.Server`` as a real MCP
   tool. These require the optional ``mcp`` extra.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from researchops.core.tools.registry import ToolRegistry
from researchops.core.tools.schema import ToolDefinition

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.server import Server
    from mcp.types import Tool


# ── Shorthand → JSON Schema ────────────────────────────────────────────

_PRIMITIVE_MAP: dict[str, str] = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
}

_LIST_RE = re.compile(r"^list\s*\[\s*([^\]]+)\s*\]$", re.IGNORECASE)


def _shorthand_type_to_schema(shorthand: str) -> dict[str, Any]:
    """Map a single shorthand type string to a JSON-schema fragment."""
    s = shorthand.strip().lower()
    if s in _PRIMITIVE_MAP:
        return {"type": _PRIMITIVE_MAP[s]}
    if s in ("dict", "object"):
        return {"type": "object"}
    if s in ("any", ""):
        return {}
    m = _LIST_RE.match(s)
    if m:
        inner = m.group(1).strip()
        if inner in _PRIMITIVE_MAP:
            return {"type": "array", "items": {"type": _PRIMITIVE_MAP[inner]}}
        if inner in ("dict", "object"):
            return {"type": "array", "items": {"type": "object"}}
        return {"type": "array"}
    if s.startswith("list"):
        return {"type": "array"}
    # Unknown — leave un-typed so the client can still send anything.
    return {}


def shorthand_to_json_schema(shorthand: dict[str, Any] | None) -> dict[str, Any]:
    """Convert a ToolDefinition.input_schema shorthand to a real JSON Schema.

    The existing convention in ``register_research_tools`` / ``register_market_tools``
    is to pass a flat ``{"param_name": "type-string"}`` mapping. This helper
    promotes that into the JSON-Schema object that MCP clients expect.

    All listed parameters are marked ``required`` because the underlying
    handlers accept them as positional/keyword arguments without defaults
    in the registry layer.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    if shorthand:
        for name, type_str in shorthand.items():
            properties[name] = _shorthand_type_to_schema(str(type_str))
            required.append(name)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


# ── ToolDefinition → MCP Tool ──────────────────────────────────────────

def _build_description(defn: ToolDefinition) -> str:
    """Compose a description that surfaces risk + permissions to MCP clients."""
    parts: list[str] = []
    if defn.description:
        parts.append(defn.description)
    meta_bits = [f"risk={defn.risk_level}"]
    if defn.permissions:
        meta_bits.append(f"perms={','.join(defn.permissions)}")
    if defn.cache_policy != "none":
        meta_bits.append(f"cache={defn.cache_policy}")
    parts.append(f"[{' | '.join(meta_bits)}]")
    return " ".join(parts).strip()


def tool_definition_to_mcp(defn: ToolDefinition) -> Tool:
    """Convert a :class:`ToolDefinition` into an ``mcp.types.Tool``.

    Requires the ``mcp`` extra to be installed.
    """
    from mcp.types import Tool

    return Tool(
        name=defn.name,
        description=_build_description(defn),
        inputSchema=shorthand_to_json_schema(defn.input_schema),
    )


# ── Registry → mcp.server.Server wiring ────────────────────────────────

def register_tools_on_server(server: Server, registry: ToolRegistry) -> None:
    """Wire every tool from ``registry`` onto an existing ``mcp.server.Server``.

    Adds two handlers:

    * ``list_tools`` — returns the static list of MCP ``Tool`` descriptors,
      computed once at registration time from the snapshot of
      ``registry.list_tools()``.
    * ``call_tool`` — dispatches incoming tool calls to
      ``registry.invoke(name, params)`` and JSON-serialises the return value
      into a single ``TextContent`` payload.
    """
    from mcp.types import TextContent

    tools_snapshot = [tool_definition_to_mcp(d) for d in registry.list_tools()]

    @server.list_tools()
    async def _list_tools() -> list[Any]:
        return list(tools_snapshot)

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[Any]:
        params = dict(arguments or {})
        try:
            result = registry.invoke(name, params)
        except Exception as exc:  # surface as MCP error text rather than crash
            return [TextContent(type="text", text=f"ERROR: {type(exc).__name__}: {exc}")]
        try:
            text = json.dumps(result, indent=2, default=str, ensure_ascii=False)
        except Exception:
            text = repr(result)
        return [TextContent(type="text", text=text)]
