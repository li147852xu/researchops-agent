"""Expose ``runs/<run_id>/`` artifacts as MCP resources.

Each completed run produces a workspace under ``runs/<run_id>/``. We surface
three of the most useful files as MCP resources so any MCP client can pull
historical run outputs into the model's context:

* ``plan.json``       — research plan / outline / coverage checklist
* ``sources.jsonl``   — collected sources with provenance metadata
* ``report.md``       — final report with citation markers

URI scheme: ``researchops://runs/{run_id}/{filename}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.server import Server


_EXPOSED_FILES: dict[str, str] = {
    "plan.json": "application/json",
    "sources.jsonl": "application/x-ndjson",
    "report.md": "text/markdown",
}

_URI_SCHEME = "researchops"
_URI_PREFIX = f"{_URI_SCHEME}://runs/"


def _build_uri(run_id: str, filename: str) -> str:
    return f"{_URI_PREFIX}{run_id}/{filename}"


def _parse_uri(uri: str) -> tuple[str, str] | None:
    """Return ``(run_id, filename)`` for a valid researchops URI, else ``None``."""
    if not uri.startswith(_URI_PREFIX):
        return None
    rest = uri[len(_URI_PREFIX):]
    parts = rest.split("/", 1)
    if len(parts) != 2:
        return None
    run_id, filename = parts
    if not run_id or not filename:
        return None
    return run_id, filename


def list_run_resources(runs_dir: Path) -> list[dict[str, Any]]:
    """Enumerate exposed run artifacts as plain dicts (SDK-agnostic, testable)."""
    out: list[dict[str, Any]] = []
    if not runs_dir.exists():
        return out
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        run_id = run_dir.name
        for filename, mime in _EXPOSED_FILES.items():
            f = run_dir / filename
            if not f.exists():
                continue
            out.append(
                {
                    "uri": _build_uri(run_id, filename),
                    "name": f"{run_id}/{filename}",
                    "description": f"ResearchOps run artifact ({filename}) from {run_id}",
                    "mimeType": mime,
                }
            )
    return out


def read_run_resource(runs_dir: Path, uri: str) -> str:
    """Return the file text for a researchops resource URI.

    Raises ``ValueError`` for malformed URIs and ``FileNotFoundError`` when
    the run or file does not exist. Path traversal is rejected.
    """
    parsed = _parse_uri(uri)
    if parsed is None:
        raise ValueError(f"Not a researchops resource URI: {uri!r}")
    run_id, filename = parsed
    if filename not in _EXPOSED_FILES:
        raise ValueError(f"Resource {filename!r} is not exposed")
    if "/" in run_id or ".." in run_id or "/" in filename:
        raise ValueError(f"Refusing path traversal in URI: {uri!r}")
    target = runs_dir / run_id / filename
    if not target.exists():
        raise FileNotFoundError(f"Run artifact not found: {target}")
    return target.read_text(encoding="utf-8")


def register_resources_on_server(server: Server, runs_dir: Path) -> None:
    """Wire run-artifact resources onto an ``mcp.server.Server``.

    Resources are listed dynamically per call so that runs created after the
    server has started are visible without restarting it.
    """
    from mcp.types import Resource

    @server.list_resources()
    async def _list_resources() -> list[Resource]:
        items = list_run_resources(runs_dir)
        return [
            Resource(
                uri=item["uri"],
                name=item["name"],
                description=item["description"],
                mimeType=item["mimeType"],
            )
            for item in items
        ]

    @server.read_resource()
    async def _read_resource(uri: Any) -> str:
        return read_run_resource(runs_dir, str(uri))
