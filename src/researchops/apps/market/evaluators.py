"""Market Intelligence evaluation — financial metrics on top of core health + evidence quality."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from researchops.apps.market.schemas import MarketSource, QuantEvalResult
from researchops.core.evaluation.base import compute_core_health, compute_evidence_quality
from researchops.core.state import StateSnapshot, TraceEvent
from researchops.core.tracing import TraceLogger

logger = logging.getLogger(__name__)


def compute_quant_eval(run_dir: Path, *, config: Any | None = None) -> QuantEvalResult:
    trace = TraceLogger(run_dir / "trace.jsonl")
    events = trace.read_all()
    state = _load_state(run_dir)

    core = compute_core_health(run_dir)
    evidence = compute_evidence_quality(run_dir)

    sources = _load_quant_sources(run_dir)
    numerical_rate = _numerical_claim_rate(run_dir)
    freshness = _data_freshness_score(sources)
    type_div = _source_type_diversity(sources)
    sec_rate = _section_nonempty_rate(run_dir / "report.md")
    bucket_cov = _bucket_coverage_rate(run_dir)
    rel_avg = _relevance_avg(run_dir)
    is_llm = config is not None and config.llm != "none"

    result = QuantEvalResult(
        citation_coverage=evidence["citation_coverage"],
        source_diversity=evidence["source_diversity"],
        reproduction_rate=_reproduction_rate(events),
        tool_calls=core["tool_calls"],
        latency_sec=core["latency_sec"],
        steps=core["steps"],
        unsupported_claim_rate=evidence["unsupported_assertion_ratio"],
        cache_hit_rate=core["cache_hit_rate"],
        llm_enabled=is_llm,
        numerical_claim_rate=numerical_rate,
        data_freshness_score=freshness,
        source_type_diversity=type_div,
        conflict_count=evidence["conflict_count"],
        collect_rounds=state.collect_rounds if state else 1,
        artifacts_count=_count_artifacts(run_dir),
        bucket_coverage_rate=bucket_cov,
        relevance_avg=rel_avg,
        section_nonempty_rate=sec_rate,
    )

    eval_path = run_dir / "eval.json"
    eval_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def _load_quant_sources(run_dir: Path) -> list[MarketSource]:
    sources_path = run_dir / "sources.jsonl"
    if not sources_path.exists():
        return []
    results: list[MarketSource] = []
    for line in sources_path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            try:
                results.append(MarketSource.model_validate_json(line))
            except Exception:
                continue
    return results


def _numerical_claim_rate(run_dir: Path) -> float:
    notes_dir = run_dir / "notes"
    if not notes_dir.exists():
        return 0.0
    total, numerical = 0, 0
    for f in notes_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for c in data.get("claims", []):
                total += 1
                text = c.get("text", "")
                if re.search(r"\d+\.?\d*\s*[%$€£¥BMKbmk]|\$\d|revenue|profit|margin|growth", text):
                    numerical += 1
        except Exception:
            continue
    return round(numerical / max(1, total), 3)


def _data_freshness_score(sources: list[MarketSource]) -> float:
    if not sources:
        return 0.0
    web_sources = [s for s in sources if s.domain and s.domain != "builtin"]
    return round(len(web_sources) / max(1, len(sources)), 3)


def _source_type_diversity(sources: list[MarketSource]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in sources:
        key = s.source_type_detail or s.type.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def _reproduction_rate(events: list[TraceEvent]) -> float:
    execs = [e for e in events if e.action == "sandbox_exec"]
    if not execs:
        return 1.0
    successes = sum(1 for e in events if e.action == "sandbox_success")
    return round(successes / len(execs), 3)


def _count_artifacts(run_dir: Path) -> int:
    art_dir = run_dir / "artifacts"
    if not art_dir.exists():
        return 0
    return sum(1 for _ in art_dir.iterdir())


def _section_nonempty_rate(report_path: Path) -> float:
    if not report_path.exists():
        return 0.0
    text = report_path.read_text(encoding="utf-8")
    sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
    if len(sections) <= 1:
        return 0.0
    content_sections = sections[1:]
    nonempty = sum(
        1 for sec in content_sections
        if any(len(p.strip()) > 30 for p in sec.split("\n\n") if not p.strip().startswith("#"))
    )
    return round(nonempty / len(content_sections), 3)


def _bucket_coverage_rate(run_dir: Path) -> float:
    emap_path = run_dir / "evidence_map.json"
    if not emap_path.exists():
        return 0.0
    try:
        emap = json.loads(emap_path.read_text(encoding="utf-8"))
        if not emap:
            return 0.0
        with_claims = sum(1 for v in emap.values() if v.get("claims"))
        return round(with_claims / len(emap), 3)
    except Exception:
        return 0.0


def _relevance_avg(run_dir: Path) -> float:
    notes_dir = run_dir / "notes"
    if not notes_dir.exists():
        return 0.0
    scores: list[float] = []
    for f in notes_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            score = data.get("relevance_score", 0.0)
            if score > 0:
                scores.append(score)
        except Exception:
            continue
    return round(sum(scores) / max(1, len(scores)), 3) if scores else 0.0


def _load_state(run_dir: Path) -> StateSnapshot | None:
    path = run_dir / "state.json"
    if not path.exists():
        return None
    try:
        return StateSnapshot.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None
