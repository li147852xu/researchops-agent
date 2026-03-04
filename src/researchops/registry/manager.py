from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from researchops.registry.schema import ToolDefinition

if TYPE_CHECKING:
    from researchops.trace import TraceLogger


class ToolPermissionError(Exception):
    pass


ToolHandler = Callable[..., Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}
        self._session_cache: dict[str, Any] = {}
        self._granted_permissions: set[str] = set()

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        self._tools[definition.name] = definition
        self._handlers[definition.name] = handler

    def get_definition(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def grant_permissions(self, perms: set[str]) -> None:
        self._granted_permissions |= perms

    def _check_permissions(self, defn: ToolDefinition) -> None:
        missing = set(defn.permissions) - self._granted_permissions
        if missing:
            raise ToolPermissionError(
                f"Tool '{defn.name}' requires permissions: {missing}"
            )

    def _cache_key(self, name: str, params: dict[str, Any]) -> str:
        raw = json.dumps({"tool": name, **params}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def invoke(
        self,
        name: str,
        params: dict[str, Any],
        trace: TraceLogger | None = None,
    ) -> Any:
        defn = self._tools.get(name)
        if defn is None:
            raise KeyError(f"Tool '{name}' not registered")
        self._check_permissions(defn)

        if defn.cache_policy == "session":
            ck = self._cache_key(name, params)
            if ck in self._session_cache:
                if trace:
                    trace.log(tool=name, action="cache_hit", input_summary=str(params)[:200])
                return self._session_cache[ck]

        t0 = time.monotonic()
        error_msg: str | None = None
        result: Any = None
        try:
            result = self._handlers[name](**params)
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000
            if trace:
                trace.log(
                    tool=name,
                    action="invoke",
                    input_summary=str(params)[:200],
                    output_summary=str(result)[:200] if result is not None else "",
                    duration_ms=elapsed_ms,
                    error=error_msg,
                )

        if defn.cache_policy == "session":
            self._session_cache[self._cache_key(name, params)] = result

        return result

    def clear_cache(self) -> None:
        self._session_cache.clear()
