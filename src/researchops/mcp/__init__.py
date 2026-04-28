"""MCP (Model Context Protocol) adapter for ResearchOps.

This sub-package wraps the existing :class:`ToolRegistry` and ``runs/`` artifacts
behind an MCP server, so any MCP client (Claude Desktop, Cursor, Cline, ...)
can call ResearchOps tools and read past run outputs without any change to
``researchops.core.tools``.

Public entry points:

- :func:`researchops.mcp.server.run` — start a stdio MCP server
- :func:`researchops.mcp.adapter.tool_definition_to_mcp` — schema conversion
- :func:`researchops.mcp.adapter.shorthand_to_json_schema` — type shorthand helper
"""

from __future__ import annotations

from researchops.mcp.adapter import (
    shorthand_to_json_schema,
    tool_definition_to_mcp,
)

__all__ = [
    "shorthand_to_json_schema",
    "tool_definition_to_mcp",
]
