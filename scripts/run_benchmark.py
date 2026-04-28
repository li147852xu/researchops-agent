#!/usr/bin/env python3
"""ResearchOps benchmark harness.

Drives the full multi-agent pipeline over a configurable topic set and
emits machine-readable JSONL plus a Markdown report with three matplotlib
charts. Designed to produce a "production-grade" performance / cost /
quality readout suitable for sharing externally.

Usage
-----

    python scripts/run_benchmark.py --max-topics 3 --backend none
    python scripts/run_benchmark.py --topics-file scripts/benchmark_topics.json
    python scripts/run_benchmark.py --backend openai_compat --mode fast

If an LLM backend is requested but no API key is available in the
environment, the harness falls back to ``--llm none`` and labels every
record with ``mock_llm=true``. The summary will then include a banner
calling out that the numbers come from the rule-based pipeline only.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Pricing (DeepSeek public list price) ────────────────────────────────
INPUT_USD_PER_M = 0.14
OUTPUT_USD_PER_M = 0.28
USD_TO_CNY = 7.2

# ── Defaults ────────────────────────────────────────────────────────────
DEFAULT_TOPICS_FILE = Path("scripts/benchmark_topics.json")
DEFAULT_OUT_DIR = Path("experiments/benchmark")
RUN_TIMEOUT_SEC = 600

# Agents we surface in the per-agent breakdown (fixed display order).
AGENT_ORDER = ["planner", "collector", "reader", "verifier", "writer", "qa", "supervisor"]


# ── Backend resolution ──────────────────────────────────────────────────

def _resolve_backend(requested: str) -> tuple[str, bool, str]:
    """Return ``(actual_backend, fell_back, reason)``.

    Mirrors the env-var lookup in ``researchops.cli._resolve_api_key`` so we
    only fall back when the CLI itself would refuse to start.
    """
    if requested == "none":
        return "none", False, ""
    env_keys: tuple[str, ...]
    if requested in ("openai", "openai_compat"):
        env_keys = ("OPENAI_API_KEY", "LLM_API_KEY", "DEEPSEEK_API_KEY")
    elif requested == "anthropic":
        env_keys = ("ANTHROPIC_API_KEY", "LLM_API_KEY")
    else:
        return requested, False, ""
    for env in env_keys:
        if os.environ.get(env, ""):
            return requested, False, ""
    return "none", True, f"no API key for backend={requested!r} in {env_keys}"


# ── Run-dir discovery ───────────────────────────────────────────────────

def _latest_run_dir(app: str, runs_dir: Path, since_ts: float) -> Path | None:
    """Pick the freshest ``runs/<app>_*`` directory created after ``since_ts``."""
    candidates: list[tuple[float, Path]] = []
    if not runs_dir.exists():
        return None
    for p in runs_dir.glob(f"{app}_*"):
        if not p.is_dir():
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if mtime + 0.5 < since_ts:
            continue
        candidates.append((mtime, p))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


# ── Trace parsers ───────────────────────────────────────────────────────

def _read_trace(run_dir: Path) -> list[dict]:
    path = run_dir / "trace.jsonl"
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _extract_tokens(events: list[dict]) -> tuple[int, int]:
    """Sum (input, output) token estimates across all ``llm.result`` events."""
    tin = tout = 0
    for e in events:
        if e.get("action") != "llm.result":
            continue
        meta = e.get("meta") or {}
        tin += int(meta.get("estimated_prompt_tokens") or 0)
        tout += int(meta.get("estimated_completion_tokens") or 0)
    return tin, tout


def _estimate_cost(tin: int, tout: int) -> tuple[float, float]:
    usd = (tin * INPUT_USD_PER_M + tout * OUTPUT_USD_PER_M) / 1_000_000
    return round(usd, 6), round(usd * USD_TO_CNY, 6)


def _count_rollbacks(events: list[dict]) -> int:
    return sum(1 for e in events if e.get("action") == "rollback")


def _parse_ts(ts: str) -> float:
    """ISO 8601 → Unix seconds."""
    return datetime.fromisoformat(ts).timestamp()


def _per_agent_durations(events: list[dict]) -> dict[str, float]:
    """Sum elapsed wall-clock time per agent by pairing start→complete events.

    The supervisor decision/rollback events live under stage=SUPERVISOR with no
    explicit start/complete pair, so they're approximated by summing their
    ``duration_ms`` (almost always 0 in practice).
    """
    pending: dict[tuple[str, str], float] = {}
    totals: dict[str, float] = defaultdict(float)
    for e in events:
        agent = (e.get("agent") or "").strip()
        stage = (e.get("stage") or "").strip()
        action = e.get("action") or ""
        if not agent or not stage:
            if agent == "" and stage == "SUPERVISOR" and e.get("duration_ms"):
                totals["supervisor"] += float(e["duration_ms"]) / 1000.0
            continue
        ts = e.get("ts")
        if not ts:
            continue
        if action == "start":
            pending[(stage, agent)] = _parse_ts(ts)
        elif action in ("complete", "rollback"):
            t0 = pending.pop((stage, agent), None)
            if t0 is not None:
                totals[agent] += max(0.0, _parse_ts(ts) - t0)

    for agent in AGENT_ORDER:
        totals.setdefault(agent, 0.0)
    return {a: round(totals[a], 3) for a in AGENT_ORDER}


# ── QA / eval helpers ──────────────────────────────────────────────────

def _qa_passed(run_dir: Path, events: list[dict]) -> tuple[bool, int, int]:
    """Return ``(passed, high_severity_issues, total_issue_count)``.

    Aligns with the pipeline's own gate (``researchops.agents.qa``): a run
    "passes" QA when the QA agent emits a ``(stage=QA, action=complete)`` event,
    which happens either when there are no high-severity issues or the run
    has exhausted its rollback budget. We still surface the high-severity
    issue count so the report can flag fragile passes.
    """
    qa_path = run_dir / "qa_report.json"
    high = 0
    total = 0
    if qa_path.exists():
        try:
            report = json.loads(qa_path.read_text(encoding="utf-8"))
            issues = report.get("issues") or []
            high = sum(1 for i in issues if i.get("severity") == "high")
            total = len(issues)
        except json.JSONDecodeError:
            pass
    qa_complete = any(
        (e.get("stage") == "QA" and e.get("action") == "complete") for e in events
    )
    return qa_complete, high, total


def _read_eval(run_dir: Path) -> dict:
    eval_path = run_dir / "eval.json"
    if not eval_path.exists():
        return {}
    try:
        return json.loads(eval_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _wall_clock_latency(events: list[dict]) -> float:
    """First event ts → last event ts, in seconds."""
    if not events:
        return 0.0
    try:
        return round(_parse_ts(events[-1]["ts"]) - _parse_ts(events[0]["ts"]), 3)
    except (KeyError, ValueError):
        return 0.0


# ── Single run driver ───────────────────────────────────────────────────

def _run_one(
    *,
    topic_entry: dict,
    app: str,
    backend: str,
    mode: str,
    mock_llm: bool,
    runs_dir: Path,
    allow_net: bool = False,
) -> dict[str, Any]:
    topic = topic_entry["topic"]
    sources = "hybrid" if app == "market" else ("hybrid" if allow_net else "demo")
    cmd = [
        "researchops", "run", topic,
        "--app", app,
        "--mode", mode,
        "--allow-net", "true" if allow_net else "false",
        "--llm", backend,
        "--sources", sources,
    ]
    if backend != "none":
        model = os.environ.get("LLM_MODEL", "")
        base_url = os.environ.get("LLM_BASE_URL", "")
        if model:
            cmd += ["--llm-model", model]
        if base_url:
            cmd += ["--llm-base-url", base_url]
    if app == "market":
        ticker = topic_entry.get("ticker", "")
        analysis = topic_entry.get("analysis_type", "fundamental")
        if ticker:
            cmd += ["--ticker", ticker]
        cmd += ["--analysis-type", analysis]

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT_SEC)
    walltime = time.time() - started

    record: dict[str, Any] = {
        "topic": topic,
        "app": app,
        "ticker": topic_entry.get("ticker"),
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "wall_time_sec": round(walltime, 3),
        "mock_llm": mock_llm,
        "error": None,
    }

    if proc.returncode != 0:
        record["error"] = (proc.stderr or proc.stdout)[-2000:]
        return record

    run_dir = _latest_run_dir(app, runs_dir, started)
    if run_dir is None:
        record["ok"] = False
        record["error"] = "no run dir produced"
        return record

    events = _read_trace(run_dir)
    eval_data = _read_eval(run_dir)
    qa_pass, qa_high, qa_total = _qa_passed(run_dir, events)
    tin, tout = _extract_tokens(events)
    usd, cny = _estimate_cost(tin, tout)
    latency = float(eval_data.get("latency_sec") or _wall_clock_latency(events))

    record.update({
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "latency_sec": round(latency, 3),
        "tokens_input": tin,
        "tokens_output": tout,
        "tokens_total": tin + tout,
        "cost_usd": usd,
        "cost_cny": cny,
        "rollback_count": _count_rollbacks(events),
        "qa_passed": qa_pass,
        "qa_high_issues": qa_high,
        "qa_issue_count": qa_total,
        "citation_coverage": float(eval_data.get("citation_coverage") or 0.0),
        "bucket_coverage_rate": float(eval_data.get("bucket_coverage_rate") or 0.0),
        "relevance_avg": float(eval_data.get("relevance_avg") or 0.0),
        "tool_calls": int(eval_data.get("tool_calls") or 0),
        "agent_durations_sec": _per_agent_durations(events),
    })
    return record


# ── Aggregation ─────────────────────────────────────────────────────────

def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sv = sorted(values)
    k = (len(sv) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sv) - 1)
    frac = k - lo
    return float(sv[lo] * (1 - frac) + sv[hi] * frac)


def _aggregate(records: list[dict]) -> dict[str, Any]:
    ok = [r for r in records if r.get("ok")]
    latencies = [r["latency_sec"] for r in ok if "latency_sec" in r]
    tokens = [r["tokens_total"] for r in ok if "tokens_total" in r]
    costs_usd = [r["cost_usd"] for r in ok if "cost_usd" in r]
    costs_cny = [r["cost_cny"] for r in ok if "cost_cny" in r]
    rollbacks = [r["rollback_count"] for r in ok if "rollback_count" in r]
    qa_pass = [bool(r.get("qa_passed")) for r in ok]
    coverage = [r.get("citation_coverage", 0.0) for r in ok]

    agent_means: dict[str, float] = {}
    if ok:
        for agent in AGENT_ORDER:
            vals = [r["agent_durations_sec"].get(agent, 0.0) for r in ok if "agent_durations_sec" in r]
            agent_means[agent] = round(statistics.mean(vals) if vals else 0.0, 3)

    rollback_triggered = [r for r in ok if (r.get("rollback_count") or 0) > 0]
    rollback_rate = (len(rollback_triggered) / len(ok)) if ok else 0.0
    post_rollback_pass_rate = (
        sum(1 for r in rollback_triggered if r.get("qa_passed")) / len(rollback_triggered)
        if rollback_triggered else 0.0
    )

    return {
        "n_total": len(records),
        "n_ok": len(ok),
        "n_failed": len(records) - len(ok),
        "any_mock_llm": any(r.get("mock_llm") for r in records),
        "all_mock_llm": all(r.get("mock_llm") for r in records) if records else False,
        "latency": {
            "p50": round(_percentile(latencies, 50), 3),
            "p95": round(_percentile(latencies, 95), 3),
            "p99": round(_percentile(latencies, 99), 3),
            "mean": round(statistics.mean(latencies), 3) if latencies else 0.0,
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "tokens": {
            "mean": round(statistics.mean(tokens), 1) if tokens else 0.0,
            "max": max(tokens) if tokens else 0,
            "total": sum(tokens),
        },
        "cost_usd": {
            "mean": round(statistics.mean(costs_usd), 4) if costs_usd else 0.0,
            "max": round(max(costs_usd), 4) if costs_usd else 0.0,
            "total": round(sum(costs_usd), 4),
        },
        "cost_cny": {
            "mean": round(statistics.mean(costs_cny), 4) if costs_cny else 0.0,
            "max": round(max(costs_cny), 4) if costs_cny else 0.0,
            "total": round(sum(costs_cny), 4),
        },
        "rollback": {
            "trigger_rate": round(rollback_rate, 3),
            "max_rounds": max(rollbacks) if rollbacks else 0,
            "post_rollback_pass_rate": round(post_rollback_pass_rate, 3),
        },
        "qa_pass_rate": round(sum(qa_pass) / len(qa_pass), 3) if qa_pass else 0.0,
        "citation_coverage_mean": round(statistics.mean(coverage), 3) if coverage else 0.0,
        "agent_means_sec": agent_means,
    }


# ── Charts ──────────────────────────────────────────────────────────────

def _render_charts(records: list[dict], agg: dict, charts_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    charts_dir.mkdir(parents=True, exist_ok=True)
    ok = [r for r in records if r.get("ok")]

    # Chart 1 — latency histogram
    latencies = [r["latency_sec"] for r in ok if "latency_sec" in r] or [0.0]
    p50 = agg["latency"]["p50"]
    p95 = agg["latency"]["p95"]
    p99 = agg["latency"]["p99"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = max(5, min(20, len(latencies)))
    ax.hist(latencies, bins=bins, edgecolor="black", alpha=0.7)
    for value, label, style in (
        (p50, f"P50 = {p50:.2f}s", "--"),
        (p95, f"P95 = {p95:.2f}s", "-."),
        (p99, f"P99 = {p99:.2f}s", ":"),
    ):
        ax.axvline(value, linestyle=style, linewidth=2, label=label)
    ax.set_xlabel("End-to-end latency (s)")
    ax.set_ylabel("Run count")
    ax.set_title("ResearchOps end-to-end latency distribution")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(charts_dir / "latency_hist.png", dpi=130)
    plt.close(fig)

    # Chart 2 — agent-share pie
    agent_means = agg["agent_means_sec"]
    labels = [a for a in AGENT_ORDER if agent_means.get(a, 0.0) > 0]
    sizes = [agent_means[a] for a in labels]
    if not sizes:
        labels, sizes = ["(no data)"], [1.0]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Mean per-agent wall-clock share")
    fig.tight_layout()
    fig.savefig(charts_dir / "agent_share.png", dpi=130)
    plt.close(fig)

    # Chart 3 — rollback rounds vs final QA pass rate
    buckets = ["0", "1", "2", ">=3"]
    bucket_records: dict[str, list[dict]] = {b: [] for b in buckets}
    for r in ok:
        rc = r.get("rollback_count") or 0
        key = str(rc) if rc < 3 else ">=3"
        bucket_records[key].append(r)
    counts = [len(bucket_records[b]) for b in buckets]
    pass_rates = [
        (sum(1 for r in bucket_records[b] if r.get("qa_passed")) / len(bucket_records[b]) * 100.0)
        if bucket_records[b] else 0.0
        for b in buckets
    ]
    x = list(range(len(buckets)))
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    bars = ax1.bar(x, pass_rates, alpha=0.7, label="QA pass rate (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"rollback {b}" for b in buckets])
    ax1.set_ylabel("QA pass rate (%)")
    ax1.set_ylim(0, 110)
    for rect, value in zip(bars, pass_rates, strict=False):
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 2,
            f"{value:.0f}%",
            ha="center", va="bottom", fontsize=9,
        )
    ax2 = ax1.twinx()
    ax2.plot(x, counts, marker="o", linewidth=2, label="Run count")
    ax2.set_ylabel("Run count")
    ax2.set_ylim(0, max(counts) * 1.4 if counts and max(counts) > 0 else 1)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Rollback rounds vs final QA pass rate")
    fig.tight_layout()
    fig.savefig(charts_dir / "rollback_pass.png", dpi=130)
    plt.close(fig)


# ── Markdown summary ───────────────────────────────────────────────────

def _render_summary(records: list[dict], agg: dict, out_dir: Path) -> str:
    lat = agg["latency"]
    cny = agg["cost_cny"]
    usd = agg["cost_usd"]
    rb = agg["rollback"]

    headline = (
        f"P95 latency: {lat['p95']:.1f}s | "
        f"平均成本: ¥{cny['mean']:.2f}/报告 | "
        f"rollback 触发率: {rb['trigger_rate'] * 100:.0f}% | "
        f"rollback 后 QA pass 率: {rb['post_rollback_pass_rate'] * 100:.0f}%"
    )

    lines: list[str] = []
    lines.append("# ResearchOps Performance & Cost Benchmark\n")
    if agg["any_mock_llm"]:
        lines.append(
            "> 注意：以下数字来自 mock LLM (`LLM_BACKEND=none`)，仅用于演示 pipeline 健壮性。\n"
        )
    lines.append(f"**{headline}**\n")
    lines.append(
        f"- Topics run: {agg['n_total']} (ok: {agg['n_ok']}, failed: {agg['n_failed']})  "
    )
    lines.append(
        f"- Mock LLM: any={str(agg['any_mock_llm']).lower()}, all={str(agg['all_mock_llm']).lower()}  "
    )
    lines.append(
        f"- Pricing: DeepSeek list (input ${INPUT_USD_PER_M}/1M, output ${OUTPUT_USD_PER_M}/1M); "
        f"USD→CNY = {USD_TO_CNY}\n"
    )

    lines.append("## Latency / Cost / Quality\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Latency P50 / P95 / P99 | {lat['p50']:.2f}s / {lat['p95']:.2f}s / {lat['p99']:.2f}s |")
    lines.append(f"| Latency mean / max | {lat['mean']:.2f}s / {lat['max']:.2f}s |")
    lines.append(f"| Tokens mean / max / total | {agg['tokens']['mean']:.0f} / {agg['tokens']['max']} / {agg['tokens']['total']} |")
    lines.append(
        f"| Cost mean / max / total (USD) | ${usd['mean']:.4f} / ${usd['max']:.4f} / ${usd['total']:.4f} |"
    )
    lines.append(
        f"| Cost mean / max / total (CNY) | ¥{cny['mean']:.4f} / ¥{cny['max']:.4f} / ¥{cny['total']:.4f} |"
    )
    lines.append(f"| Rollback trigger rate | {rb['trigger_rate'] * 100:.1f}% |")
    lines.append(f"| Post-rollback QA pass rate | {rb['post_rollback_pass_rate'] * 100:.1f}% |")
    lines.append(f"| Final QA pass rate (overall) | {agg['qa_pass_rate'] * 100:.1f}% |")
    lines.append(f"| Citation coverage (mean) | {agg['citation_coverage_mean'] * 100:.1f}% |")
    lines.append("")

    lines.append("## Charts\n")
    lines.append("![Latency histogram](charts/latency_hist.png)\n")
    lines.append("![Per-agent wall-clock share](charts/agent_share.png)\n")
    lines.append("![Rollback rounds vs QA pass rate](charts/rollback_pass.png)\n")

    lines.append("## Per-agent mean wall-clock (seconds)\n")
    lines.append("| Agent | Mean seconds |")
    lines.append("|---|---|")
    for agent in AGENT_ORDER:
        lines.append(f"| {agent} | {agg['agent_means_sec'].get(agent, 0.0):.3f} |")
    lines.append("")

    lines.append("## Per-topic results (top 25)\n")
    lines.append("| # | App | Topic | Latency (s) | Tokens | Cost (¥) | Rollbacks | QA pass |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(records[:25], start=1):
        if not r.get("ok"):
            lines.append(
                f"| {i} | {r.get('app', '?')} | {r['topic'][:80]} | — | — | — | — | "
                f"FAILED ({(r.get('error') or 'unknown')[:60]}) |"
            )
            continue
        lines.append(
            f"| {i} | {r['app']} | {r['topic'][:80]} | "
            f"{r['latency_sec']:.2f} | {r['tokens_total']} | ¥{r['cost_cny']:.4f} | "
            f"{r['rollback_count']} | {'yes' if r['qa_passed'] else 'no'} |"
        )
    lines.append("")
    lines.append(
        "_Generated by `scripts/run_benchmark.py`. "
        "QA pass = pipeline emitted a `(stage=QA, action=complete)` event "
        "(no high-severity issues, or rollback budget exhausted). "
        "Per-agent times are summed across rollback rounds._\n"
    )

    summary_text = "\n".join(lines)
    (out_dir / "summary.md").write_text(summary_text, encoding="utf-8")
    return summary_text


# ── Main ────────────────────────────────────────────────────────────────

def _load_topics(path: Path, max_topics: int) -> list[tuple[str, dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs: list[tuple[str, dict]] = []
    for entry in data.get("research", []):
        pairs.append(("research", entry))
    for entry in data.get("market", []):
        pairs.append(("market", entry))
    if max_topics > 0:
        pairs = pairs[:max_topics]
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="ResearchOps benchmark harness")
    parser.add_argument("--topics-file", type=Path, default=DEFAULT_TOPICS_FILE)
    parser.add_argument("--max-topics", type=int, default=0,
                        help="Cap the number of topics (0 = no cap, default)")
    parser.add_argument("--backend", default="none",
                        choices=["none", "openai", "openai_compat", "anthropic"],
                        help="LLM backend (auto-falls back to none if no API key)")
    parser.add_argument("--mode", default="fast", choices=["fast", "deep"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--allow-net", action="store_true",
        help="Allow network access (arxiv/wiki/web). Default: hermetic demo sources.",
    )
    args = parser.parse_args()

    if not args.topics_file.exists():
        print(f"ERROR: topics file not found: {args.topics_file}", file=sys.stderr)
        return 2

    backend, fell_back, reason = _resolve_backend(args.backend)
    mock_llm = backend == "none"
    if fell_back:
        print(f"[warn] {reason}; falling back to backend=none", file=sys.stderr)

    topics = _load_topics(args.topics_file, args.max_topics)
    print(
        f"Benchmark: {len(topics)} topics, backend={backend}"
        f"{' (fallback)' if fell_back else ''}, mode={args.mode}, "
        f"allow_net={args.allow_net}"
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = out_dir / "charts"
    results_path = out_dir / "results.jsonl"
    results_path.write_text("", encoding="utf-8")

    records: list[dict] = []
    for i, (app, entry) in enumerate(topics, start=1):
        print(f"[{i}/{len(topics)}] {app}: {entry['topic'][:80]}", flush=True)
        try:
            rec = _run_one(
                topic_entry=entry, app=app, backend=backend, mode=args.mode,
                mock_llm=mock_llm, runs_dir=args.runs_dir,
                allow_net=args.allow_net,
            )
        except subprocess.TimeoutExpired as exc:
            rec = {
                "topic": entry["topic"], "app": app,
                "ticker": entry.get("ticker"),
                "ok": False, "exit_code": -1, "mock_llm": mock_llm,
                "error": f"timeout after {exc.timeout}s",
            }
        records.append(rec)
        with results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        if rec.get("ok"):
            print(
                f"  ok latency={rec.get('latency_sec', 0):.2f}s "
                f"tokens={rec.get('tokens_total', 0)} "
                f"cost=¥{rec.get('cost_cny', 0):.4f} "
                f"rollback={rec.get('rollback_count', 0)} "
                f"qa_pass={'yes' if rec.get('qa_passed') else 'no'}",
                flush=True,
            )
        else:
            print(f"  FAILED: {rec.get('error', 'unknown')[:120]}", flush=True)

    agg = _aggregate(records)
    _render_charts(records, agg, charts_dir)
    summary_text = _render_summary(records, agg, out_dir)
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    headline_line = next(
        (ln.strip("*") for ln in summary_text.splitlines() if ln.startswith("**P95")),
        "",
    )
    print()
    print(f"Headline: {headline_line}")
    print(f"Artifacts written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
