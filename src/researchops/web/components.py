"""Shared UI components — pipeline progress, trace parsing, API-key resolution."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

_BJT = timezone(timedelta(hours=8))

STAGES = ["PLAN", "COLLECT", "READ", "VERIFY", "WRITE", "QA", "EVAL"]
_TRACE_STAGE_ORDER = ["PLAN", "COLLECT", "READ", "VERIFY", "WRITE", "QA"]


def resolve_api_key(llm: str, api_key: str) -> str:
    if api_key:
        return api_key
    if llm in ("openai", "openai_compat"):
        for env in ("OPENAI_API_KEY", "LLM_API_KEY", "DEEPSEEK_API_KEY"):
            val = os.environ.get(env, "")
            if val:
                return val
    elif llm == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("LLM_API_KEY", "")
    return ""


def stage_html(completed: list[str], current: str = "") -> str:
    parts: list[str] = []
    for s in STAGES:
        if s in completed:
            cls = "stage-done"
            icon = "&#10003;"
        elif s == current:
            cls = "stage-active"
            icon = "&#9654;"
        else:
            cls = "stage-pending"
            icon = "&#9679;"
        parts.append(f'<span class="stage-indicator {cls}">{icon} {s}</span>')
    return " ".join(parts)


def _utc_to_bjt(iso_ts: str) -> str:
    if not iso_ts or len(iso_ts) < 19:
        return iso_ts[:8] if len(iso_ts) >= 8 else iso_ts
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        bjt = dt.astimezone(_BJT)
        return bjt.strftime("%H:%M:%S")
    except Exception:
        return iso_ts[11:19]


def detect_stages_from_trace(run_dir: Path) -> tuple[list[str], str, list[str]]:
    """Parse trace.jsonl to determine completed stages, active stage, and log lines."""
    trace_path = run_dir / "trace.jsonl"
    if not trace_path.exists():
        return [], "PLAN", []

    completed: list[str] = []
    current = "PLAN"
    log_lines: list[str] = []
    first_ts_raw: str = ""
    last_ts_raw: str = ""
    try:
        lines = trace_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        for line in lines:
            if not line.strip():
                continue
            try:
                evt = json.loads(line)
            except Exception:
                continue
            stage = evt.get("stage", "")
            action = evt.get("action", "")
            agent = evt.get("agent", "")
            tool = evt.get("tool", "")
            dur = evt.get("duration_ms", 0)
            error = evt.get("error")
            meta = evt.get("meta", {})
            raw_ts = evt.get("ts", "")
            ts = _utc_to_bjt(raw_ts)
            output = evt.get("output_summary", "")

            if raw_ts:
                if not first_ts_raw:
                    first_ts_raw = raw_ts
                last_ts_raw = raw_ts

            if stage and stage in _TRACE_STAGE_ORDER:
                if action == "complete" and stage not in completed:
                    completed.append(stage)
                elif action == "start":
                    current = stage

            if action == "start" and stage:
                label = f"[{agent}]" if agent else f"[{stage}]"
                log_lines.append(f"[{ts}] {label} Starting {stage}...")
            elif action == "complete" and stage:
                label = f"[{agent}]" if agent else f"[{stage}]"
                msg = output or "done"
                log_lines.append(f"[{ts}] {label} {stage} complete — {msg}")
            elif action == "decide":
                codes = meta.get("reason_codes", [])
                conf = meta.get("confidence", 0)
                log_lines.append(f"[{ts}] [supervisor] Decision: {codes} (confidence={conf:.2f})")
            elif action == "invoke" and tool:
                dur_s = f"{dur / 1000:.1f}s" if dur > 0 else ""
                if error:
                    log_lines.append(f"[{ts}]   {tool} FAILED {dur_s} — {error[:80]}")
                else:
                    log_lines.append(f"[{ts}]   {tool} ok {dur_s}")
            elif action == "cache_hit" and tool:
                pass
            elif action == "llm.result":
                tokens = meta.get("tokens", 0)
                lat = meta.get("latency_ms", 0)
                log_lines.append(f"[{ts}]   LLM response: {tokens} tokens ({lat / 1000:.1f}s)")
            elif action in ("parse.low_quality", "parse.skipped"):
                sid = meta.get("source_id", "")
                reason = meta.get("reason", "")
                log_lines.append(f"[{ts}]   parse skip: {sid} — {reason}")
            elif action == "diversity_check":
                arxiv_n = meta.get("arxiv_count", 0)
                total = meta.get("total", 0)
                log_lines.append(f"[{ts}]   Sources: {total} total, {arxiv_n} arxiv")
    except Exception:
        pass

    if first_ts_raw and last_ts_raw:
        try:
            t0 = datetime.fromisoformat(first_ts_raw)
            t1 = datetime.fromisoformat(last_ts_raw)
            elapsed = (t1 - t0).total_seconds()
            log_lines.append(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
        except Exception:
            pass

    if current in completed:
        idx = _TRACE_STAGE_ORDER.index(current)
        if idx + 1 < len(_TRACE_STAGE_ORDER):
            current = _TRACE_STAGE_ORDER[idx + 1]

    return completed, current, log_lines
