"""Core evaluation — health metrics and evidence quality metrics.

App-specific metrics (bucket coverage, per-RQ, judge) live in apps/research/evaluators.py.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from researchops.core.artifacts import AssertionStatus
from researchops.core.state import TraceEvent
from researchops.core.tracing import TraceLogger

logger = logging.getLogger(__name__)


def compute_core_health(run_dir: Path) -> dict[str, Any]:
    """Core pipeline health metrics — reusable across all apps."""
    trace = TraceLogger(run_dir / "trace.jsonl")
    events = trace.read_all()

    return {
        "tool_calls": _count_tool_calls(events),
        "tool_failure_rate": _tool_failure_rate(events),
        "cache_hit_rate": _cache_hit_rate(events),
        "latency_sec": _compute_latency(events),
        "steps": _count_steps(events),
        "rollback_frequency": _rollback_frequency(events),
        "artifact_completeness": _artifact_completeness(run_dir),
    }


def compute_evidence_quality(run_dir: Path) -> dict[str, Any]:
    """Evidence quality metrics — reusable across evidence-grounded apps."""
    report_path = run_dir / "report.md"
    return {
        "citation_coverage": _citation_coverage(report_path),
        "unsupported_assertion_ratio": _unsupported_claim_rate(run_dir),
        "source_diversity": _source_diversity(run_dir),
        "conflict_count": _count_conflicts(run_dir),
    }


def classify_assertion(claim_ids: list[str], source_ids: list[str]) -> AssertionStatus:
    """Classify a report entry's assertion status based on evidence links."""
    if not claim_ids and not source_ids:
        return AssertionStatus.UNSUPPORTED
    if claim_ids and source_ids:
        return AssertionStatus.SUPPORTED
    if source_ids:
        return AssertionStatus.PARTIAL
    return AssertionStatus.INFERRED


def _count_tool_calls(events: list[TraceEvent]) -> int:
    return sum(1 for e in events if e.action == "invoke")


def _tool_failure_rate(events: list[TraceEvent]) -> float:
    invokes = [e for e in events if e.action == "invoke"]
    if not invokes:
        return 0.0
    failures = sum(1 for e in invokes if e.error)
    return round(failures / len(invokes), 3)


def _cache_hit_rate(events: list[TraceEvent]) -> float:
    invokes = sum(1 for e in events if e.action in ("invoke", "cache_hit"))
    hits = sum(1 for e in events if e.action == "cache_hit")
    if invokes == 0:
        return 0.0
    return round(hits / invokes, 3)


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


def _rollback_frequency(events: list[TraceEvent]) -> int:
    return sum(1 for e in events if e.action == "rollback")


def _artifact_completeness(run_dir: Path) -> float:
    expected = ["plan.json", "sources.jsonl", "report.md", "eval.json", "trace.jsonl"]
    found = sum(1 for f in expected if (run_dir / f).exists())
    return round(found / len(expected), 3)


def _citation_coverage(report_path: Path) -> float:
    if not report_path.exists():
        return 0.0
    text = report_path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    if not paragraphs:
        return 0.0
    cited = sum(1 for p in paragraphs if re.search(r"\[@\w+\]", p))
    return round(cited / len(paragraphs), 3)


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


def _source_diversity(run_dir: Path) -> dict:
    sources_path = run_dir / "sources.jsonl"
    if not sources_path.exists():
        return {"unique_domains": 0, "domains": [], "type_distribution": {}}
    domains: set[str] = set()
    type_dist: dict[str, int] = {}
    for line in sources_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        try:
            s = json.loads(line)
            d = s.get("domain", "")
            if d:
                domains.add(d)
            t = s.get("type", "")
            type_dist[t] = type_dist.get(t, 0) + 1
        except Exception:
            continue
    return {
        "unique_domains": len(domains),
        "domains": sorted(domains),
        "type_distribution": type_dist,
    }


def _count_conflicts(run_dir: Path) -> int:
    path = run_dir / "qa_conflicts.json"
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return len(data.get("conflicts", []))
    except Exception:
        return 0
