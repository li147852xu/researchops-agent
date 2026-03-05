from __future__ import annotations

import json
import re
from pathlib import Path

from researchops.models import EvalResult, Source, TraceEvent
from researchops.trace import TraceLogger


def compute_eval(run_dir: Path, *, llm_enabled: bool = False) -> EvalResult:
    report_path = run_dir / "report.md"
    sources = _load_sources(run_dir)
    trace = TraceLogger(run_dir / "trace.jsonl")
    events = trace.read_all()

    citation_cov = _citation_coverage(report_path)
    diversity = _source_diversity(sources)
    repro_rate = _reproduction_rate(events)
    tool_calls = _count_tool_calls(events)
    latency = _compute_latency(events)
    steps = _count_steps(events)
    unsupported = _unsupported_claim_rate(run_dir)
    cache_hit = _cache_hit_rate(events)

    result = EvalResult(
        citation_coverage=citation_cov,
        source_diversity=diversity,
        reproduction_rate=repro_rate,
        tool_calls=tool_calls,
        latency_sec=latency,
        steps=steps,
        unsupported_claim_rate=unsupported,
        cache_hit_rate=cache_hit,
        llm_enabled=llm_enabled,
        estimated_cost_usd=0.0,
    )

    eval_path = run_dir / "eval.json"
    eval_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def _citation_coverage(report_path: Path) -> float:
    if not report_path.exists():
        return 0.0
    text = report_path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    if not paragraphs:
        return 0.0
    cited = sum(1 for p in paragraphs if re.search(r"\[@\w+\]", p))
    return round(cited / len(paragraphs), 3)


def _source_diversity(sources: list[Source]) -> dict:
    domains = {s.domain for s in sources if s.domain}
    type_dist: dict[str, int] = {}
    for s in sources:
        type_dist[s.type.value] = type_dist.get(s.type.value, 0) + 1
    return {
        "unique_domains": len(domains),
        "domains": sorted(domains),
        "type_distribution": type_dist,
    }


def _reproduction_rate(events: list[TraceEvent]) -> float:
    execs = [e for e in events if e.action == "sandbox_exec"]
    if not execs:
        return 1.0
    successes = sum(1 for e in events if e.action == "sandbox_success")
    return round(successes / len(execs), 3) if execs else 1.0


def _count_tool_calls(events: list[TraceEvent]) -> int:
    return sum(1 for e in events if e.action == "invoke")


def _compute_latency(events: list[TraceEvent]) -> float:
    for e in events:
        if e.action == "run_complete" and e.duration_ms > 0:
            return round(e.duration_ms / 1000, 2)
    total = sum(e.duration_ms for e in events) / 1000
    return round(total, 2)


def _count_steps(events: list[TraceEvent]) -> int:
    stages_seen: set[str] = set()
    for e in events:
        if e.stage and e.action in ("start", "complete"):
            stages_seen.add(e.stage)
    return len(stages_seen)


def _unsupported_claim_rate(run_dir: Path) -> float:
    index_path = run_dir / "report_index.json"
    if not index_path.exists():
        return 0.0
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        entries = index.get("entries", [])
        if not entries:
            return 0.0
        unsupported = sum(1 for e in entries if not e.get("claim_ids"))
        return round(unsupported / len(entries), 3)
    except Exception:
        return 0.0


def _cache_hit_rate(events: list[TraceEvent]) -> float:
    invokes = sum(1 for e in events if e.action in ("invoke", "cache_hit"))
    hits = sum(1 for e in events if e.action == "cache_hit")
    if invokes == 0:
        return 0.0
    return round(hits / invokes, 3)


def _load_sources(run_dir: Path) -> list[Source]:
    path = run_dir / "sources.jsonl"
    if not path.exists():
        return []
    sources = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            sources.append(Source.model_validate(json.loads(line)))
    return sources
