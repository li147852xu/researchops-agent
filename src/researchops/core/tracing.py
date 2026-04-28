"""Trace logger — append-only JSONL event stream for audit and replay.

Optionally fans every event out to Langfuse when ``LANGFUSE_PUBLIC_KEY`` and
``LANGFUSE_SECRET_KEY`` are configured. The Langfuse path is best-effort: any
exception is downgraded to a warning log and the JSONL write is never affected.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from researchops.core.state import TraceEvent, _now

if TYPE_CHECKING:  # pragma: no cover - typing only
    from researchops.core.observability import LangfuseFacade, LangfuseNode

logger = logging.getLogger(__name__)


class TraceLogger:
    def __init__(
        self,
        trace_path: Path,
        *,
        langfuse_client: LangfuseFacade | None = None,
    ) -> None:
        self._path = trace_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Langfuse fan-out (no-op when env / SDK not present).
        from researchops.core.observability import get_client as _get_lf

        self._lf: LangfuseFacade = langfuse_client or _get_lf()
        self._run_id: str = trace_path.parent.name or "researchops_run"
        try:
            self._lf_trace: LangfuseNode = self._lf.trace(
                id=self._run_id, name=self._run_id, metadata={"run_id": self._run_id}
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.warning("Langfuse trace init failed: %s; disabling fan-out.", exc)
            from researchops.core.observability.langfuse_client import _NoopFacade

            self._lf = _NoopFacade()
            self._lf_trace = self._lf.trace(id=self._run_id, name=self._run_id)
        self._stack: list[tuple[str, str, LangfuseNode]] = []
        self._pending_llm: dict[str, Any] | None = None

    def emit(self, event: TraceEvent) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
        # Langfuse fan-out happens AFTER the JSONL write so any failure here
        # cannot corrupt or skip the on-disk audit log.
        try:
            self._fanout(event)
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.warning("Langfuse fanout failed for %s: %s", event.action, exc)

    def log(
        self,
        *,
        stage: str = "",
        agent: str = "",
        action: str = "",
        tool: str = "",
        input_summary: str = "",
        output_summary: str = "",
        duration_ms: float = 0,
        error: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        ev = TraceEvent(
            ts=_now(),
            stage=stage,
            agent=agent,
            action=action,
            tool=tool,
            input_summary=input_summary,
            output_summary=output_summary,
            duration_ms=duration_ms,
            error=error,
            meta=meta or {},
        )
        self.emit(ev)

    @contextmanager
    def timed(self, **kwargs: Any) -> Iterator[dict[str, Any]]:
        ctx: dict[str, Any] = {}
        t0 = time.monotonic()
        try:
            yield ctx
        except Exception as exc:
            ctx["error"] = str(exc)
            raise
        finally:
            elapsed = (time.monotonic() - t0) * 1000
            self.log(
                duration_ms=elapsed,
                error=ctx.get("error"),
                **kwargs,
            )

    def read_all(self) -> list[TraceEvent]:
        if not self._path.exists():
            return []
        events = []
        for line in self._path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                events.append(TraceEvent.model_validate(json.loads(line)))
        return events

    # ── Langfuse fan-out ────────────────────────────────────────────────

    def _top(self) -> LangfuseNode:
        return self._stack[-1][2] if self._stack else self._lf_trace

    def _pop_match(self, stage: str, agent: str) -> LangfuseNode | None:
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == stage and self._stack[i][1] == agent:
                _, _, node = self._stack.pop(i)
                return node
        return None

    def _fanout(self, ev: TraceEvent) -> None:
        action = ev.action
        if action == "start" and ev.stage and ev.agent:
            node = self._top().open_span(
                f"{ev.stage}/{ev.agent}",
                input=ev.input_summary,
                metadata={"stage": ev.stage, "agent": ev.agent},
            )
            self._stack.append((ev.stage, ev.agent, node))
            return
        if action in ("complete", "rollback") and ev.stage and ev.agent:
            node = self._pop_match(ev.stage, ev.agent)
            if node is not None:
                node.end(
                    output=ev.output_summary,
                    error=ev.error if action == "rollback" or ev.error else None,
                )
            return
        if action == "llm.call":
            self._pending_llm = {
                "input": ev.input_summary,
                "meta": dict(ev.meta or {}),
            }
            return
        if action == "llm.result":
            pending = self._pending_llm or {"input": "", "meta": {}}
            self._pending_llm = None
            meta = {**pending.get("meta", {}), **(ev.meta or {})}
            self._top().log_generation(
                name="llm",
                model=str(meta.get("model", "")),
                input=str(pending.get("input", "")),
                output=ev.output_summary,
                usage_input=int(meta.get("estimated_prompt_tokens") or 0),
                usage_output=int(meta.get("estimated_completion_tokens") or 0),
                metadata={k: v for k, v in meta.items()
                          if k not in {"model", "estimated_prompt_tokens",
                                       "estimated_completion_tokens"}},
            )
            return
        if action == "llm.error":
            # Surface the failure as a marker event under the current span.
            self._top().event(
                "llm.error",
                metadata={"error": ev.error or "", **(ev.meta or {})},
            )
            self._pending_llm = None
            return
        if action in ("invoke", "cache_hit") and ev.tool:
            self._top().short_span(
                f"tool:{ev.tool}",
                input=ev.input_summary,
                output=ev.output_summary,
                metadata={
                    "tool": ev.tool,
                    "duration_ms": ev.duration_ms,
                    "cache_hit": action == "cache_hit",
                    **(ev.meta or {}),
                },
            )
            return
        # Catch-all — record as an event so it's visible in the UI.
        self._top().event(
            action or "event",
            metadata={
                "stage": ev.stage,
                "agent": ev.agent,
                "tool": ev.tool,
                "duration_ms": ev.duration_ms,
                "error": ev.error,
                **(ev.meta or {}),
            },
        )

    def close(self) -> None:
        """Close any still-open spans and flush the Langfuse buffer."""
        while self._stack:
            _, _, node = self._stack.pop()
            try:
                node.end(error="run terminated with span still open")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Langfuse close failed: %s", exc)
        try:
            self._lf.flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse flush failed: %s", exc)
