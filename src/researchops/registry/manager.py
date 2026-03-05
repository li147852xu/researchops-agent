from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
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
        self._persistent_cache: dict[str, Any] = {}
        self._persistent_cache_path: Path | None = None
        self._granted_permissions: set[str] = set()

    def set_persistent_cache_path(self, path: Path) -> None:
        self._persistent_cache_path = path
        if path.exists():
            try:
                self._persistent_cache = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._persistent_cache = {}

    def _save_persistent_cache(self) -> None:
        if self._persistent_cache_path is not None:
            self._persistent_cache_path.write_text(
                json.dumps(self._persistent_cache, indent=2, default=str), encoding="utf-8"
            )

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        self._tools[definition.name] = definition
        self._handlers[definition.name] = handler

    def get_definition(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def grant_permissions(self, perms: set[str]) -> None:
        self._granted_permissions |= perms

    def _check_permissions(self, defn: ToolDefinition, trace: TraceLogger | None) -> None:
        missing = set(defn.permissions) - self._granted_permissions
        if missing:
            if trace:
                trace.log(
                    tool=defn.name,
                    action="permission_denied",
                    error=f"Missing permissions: {missing}",
                    meta={"required": list(defn.permissions), "granted": list(self._granted_permissions)},
                )
            raise ToolPermissionError(
                f"Tool '{defn.name}' requires permissions: {missing}"
            )

    def _cache_key(self, name: str, params: dict[str, Any]) -> str:
        raw = json.dumps({"tool": name, **params}, sort_keys=True, default=str)
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
        self._check_permissions(defn, trace)

        ck = self._cache_key(name, params)

        if defn.cache_policy == "session" and ck in self._session_cache:
            if trace:
                trace.log(tool=name, action="cache_hit", input_summary=str(params)[:200],
                          meta={"cache_type": "session"})
            return self._session_cache[ck]

        if defn.cache_policy == "persistent" and ck in self._persistent_cache:
            if trace:
                trace.log(tool=name, action="cache_hit", input_summary=str(params)[:200],
                          meta={"cache_type": "persistent"})
            return self._persistent_cache[ck]

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
            self._session_cache[ck] = result
        elif defn.cache_policy == "persistent":
            self._persistent_cache[ck] = result
            self._save_persistent_cache()

        return result

    def clear_cache(self) -> None:
        self._session_cache.clear()
