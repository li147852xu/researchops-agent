from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from researchops.models import TraceEvent, _now


class TraceLogger:
    def __init__(self, trace_path: Path):
        self._path = trace_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: TraceEvent) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")

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
