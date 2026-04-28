# MCP Integration Guide

ResearchOps ships with a [Model Context Protocol](https://modelcontextprotocol.io)
stdio server that exposes its `ToolRegistry` and historical run artifacts to any
MCP client (Claude Desktop, Cursor, Cline, Continue, ...).

The adapter is non-invasive — it only **reads** `ToolDefinition`s and forwards
calls to `ToolRegistry.invoke`. No core APIs were modified.

## Install

```bash
pip install -e ".[mcp]"
```

This pulls in the official Anthropic [`mcp`](https://pypi.org/project/mcp/)
Python SDK as an optional dependency.

## Run the server

```bash
researchops mcp                         # serves over stdio, runs/ is the default resource root
researchops mcp --runs-dir /custom/runs # point at a different runs directory
```

The process speaks JSON-RPC on stdin/stdout per the MCP stdio transport. There
is no port to expose; the MCP client is responsible for spawning it.

## Smoke test

Confirm the binary boots without errors (it will block waiting for stdio
input — kill it with `Ctrl-C`):

```bash
researchops mcp < /dev/null
```

Or pipe in a `tools/list` request:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}' | researchops mcp
```

You should see a JSON response on stdout containing `"serverInfo":{"name":"researchops",...}`.

## Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS),
`%APPDATA%\Claude\claude_desktop_config.json` (Windows), or the equivalent on
Linux:

```json
{
  "mcpServers": {
    "researchops": {
      "command": "researchops",
      "args": ["mcp"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

If you installed ResearchOps inside a virtualenv, point `command` at the
absolute path to that env's `researchops` binary, e.g.
`/Users/me/.virtualenvs/researchops/bin/researchops`.

Restart Claude Desktop. In any new chat, the hammer icon should now list at
least the following ResearchOps tools:

- `web_search`
- `arxiv_search`
- `semantic_scholar_search`
- `wikipedia_search`
- `fetch`
- `parse`
- `sandbox_exec`
- `cite`
- `arxiv_download_pdf`

## Cursor

Edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "researchops": {
      "command": "researchops",
      "args": ["mcp", "--runs-dir", "/abs/path/to/researchops-ai/runs"]
    }
  }
}
```

## Cline / Continue / others

Any MCP-compatible client that supports stdio transport will work — just point
the client at `researchops mcp`.

## What gets exposed

### Tools

Every tool registered by `register_research_tools` and `register_market_tools`
becomes an MCP tool. The shorthand input schema (`{"query": "str", "max_results": "int"}`)
is automatically promoted to a real JSON Schema object. Tool descriptions
include risk level, required permissions, and cache policy so the client can
display governance metadata.

### Resources

For each `runs/<run_id>/` directory the server publishes the three most useful
artifacts as MCP resources:

| URI                                              | Description                       | MIME                    |
|--------------------------------------------------|-----------------------------------|-------------------------|
| `researchops://runs/<run_id>/plan.json`          | Plan, outline, coverage checklist | `application/json`      |
| `researchops://runs/<run_id>/sources.jsonl`      | Collected sources with provenance | `application/x-ndjson`  |
| `researchops://runs/<run_id>/report.md`          | Final report with citations       | `text/markdown`         |

The list is computed dynamically per request, so runs created after the server
boots are visible without restarting it.

## Security notes

- The MCP server auto-grants `net` and `sandbox` permissions in the registry,
  because the MCP client is itself the trust boundary. Do not expose this
  server to untrusted clients.
- Resource reads validate that the run id and filename are on a fixed
  allowlist and reject path traversal.
- Tools that hit the network (DuckDuckGo, arXiv, Semantic Scholar, Wikipedia)
  are subject to those services' rate limits — same as the CLI / Web UI.
