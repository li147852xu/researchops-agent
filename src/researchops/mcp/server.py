"""ResearchOps MCP stdio server.

Boots an :class:`mcp.server.Server` that exposes:

* every tool registered by ``register_research_tools`` and
  ``register_market_tools`` (deduplicated by name — the registry overwrites
  same-named tools, which is fine because both apps share most tools);
* every ``runs/<run_id>/{plan.json,sources.jsonl,report.md}`` artifact as
  an MCP resource.

Permissions ``net`` and ``sandbox`` are auto-granted in the MCP server
process: the MCP client itself is the trust boundary, so we let the client
decide what to call. This can be tightened later via CLI flags.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from researchops.core.tools.registry import ToolRegistry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.server import Server


def build_registry() -> ToolRegistry:
    """Build a fresh ToolRegistry with both research and market tools registered.

    Both apps register tools with overlapping names (``web_search``, ``fetch``,
    ``parse``, ``sandbox_exec``, ``cite``); ``ToolRegistry.register`` simply
    overwrites, so registering both gives the union of unique tool names.
    """
    from researchops.apps.market.adapters import register_market_tools
    from researchops.apps.research.adapters import register_research_tools

    registry = ToolRegistry()
    register_research_tools(registry)
    register_market_tools(registry)
    registry.grant_permissions({"net", "sandbox"})
    return registry


def build_server(runs_dir: Path | None = None) -> Server:
    """Construct the MCP server with tools and resources wired up."""
    from mcp.server import Server

    from researchops.mcp.adapter import register_tools_on_server
    from researchops.mcp.resources import register_resources_on_server

    server: Server = Server("researchops")

    registry = build_registry()
    register_tools_on_server(server, registry)
    register_resources_on_server(server, runs_dir or Path("runs"))

    return server


async def _run_stdio(server: Server) -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run(runs_dir: Path | None = None) -> None:
    """Synchronous entry point: build the server and serve it over stdio."""
    server = build_server(runs_dir=runs_dir)
    asyncio.run(_run_stdio(server))


if __name__ == "__main__":  # pragma: no cover - manual invocation
    run()
