"""Evaluation harness — unified eval runner, before/after comparison, metric reporting.

Provides a single entry point to evaluate any completed run (research or
market intelligence), compare two runs side by side, and produce a
structured report suitable for both CLI display and JSON export.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from researchops.core.evaluation.base import compute_core_health, compute_evidence_quality
from researchops.core.quality import (
    compute_evidence_density,
    detect_conflicts,
    find_citation_gaps,
    overall_evidence_density,
    score_claim_confidence,
)

logger = logging.getLogger(__name__)


# ── Structured Eval Report ─────────────────────────────────────────────

@dataclass
class MetricDelta:
    """Before/after delta for a single metric."""

    name: str = ""
    before: float = 0.0
    after: float = 0.0
    delta: float = 0.0
    improved: bool = False


@dataclass
class EvalReport:
    """Complete evaluation report for a single run."""

    run_id: str = ""
    app_type: str = ""
    core_health: dict[str, Any] = field(default_factory=dict)
    evidence_quality: dict[str, Any] = field(default_factory=dict)
    evidence_density: float = 0.0
    citation_gaps: int = 0
    claim_confidence_avg: float = 0.0
    conflict_count: int = 0
    quality_enhanced: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "app_type": self.app_type,
            "core_health": self.core_health,
            "evidence_quality": self.evidence_quality,
            "evidence_density": self.evidence_density,
            "citation_gaps": self.citation_gaps,
            "claim_confidence_avg": self.claim_confidence_avg,
            "conflict_count": self.conflict_count,
            "quality_enhanced": self.quality_enhanced,
        }


@dataclass
class ComparisonReport:
    """Side-by-side comparison of two evaluation runs."""

    run_a_id: str = ""
    run_b_id: str = ""
    deltas: list[MetricDelta] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_a": self.run_a_id,
            "run_b": self.run_b_id,
            "deltas": [
                {"name": d.name, "before": d.before, "after": d.after,
                 "delta": d.delta, "improved": d.improved}
                for d in self.deltas
            ],
            "summary": self.summary,
        }


# ── Unified Eval Runner ───────────────────────────────────────────────

def evaluate_run(run_dir: Path, app_type: str = "auto") -> EvalReport:
    """Evaluate a completed run and produce a structured report.

    Combines core health metrics, evidence quality, and the new quality
    enhancement signals into a single EvalReport.
    """
    run_id = run_dir.name

    if app_type == "auto":
        app_type = _detect_app_type(run_dir)

    core = compute_core_health(run_dir)
    evidence = compute_evidence_quality(run_dir)

    report_path = run_dir / "report.md"
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

    density = overall_evidence_density(report_text) if report_text else 0.0
    gaps = find_citation_gaps(report_text) if report_text else []

    claims = _load_all_claims(run_dir)
    confidence_scores = score_claim_confidence(claims)
    conf_avg = (
        sum(c.confidence for c in confidence_scores) / len(confidence_scores)
        if confidence_scores
        else 0.0
    )

    conflicts = detect_conflicts(claims)

    quality_enhanced = {
        "section_densities": [
            {"heading": s.heading, "density": s.density_score, "paragraphs": s.paragraph_count}
            for s in compute_evidence_density(report_text)
        ] if report_text else [],
        "citation_gap_count": len(gaps),
        "top_gaps": [
            {"section": g.section, "preview": g.text_preview[:80]}
            for g in gaps[:5]
        ],
        "claim_confidence_distribution": _confidence_distribution(confidence_scores),
        "conflicts": [
            {"type": c.conflict_type, "severity": c.severity,
             "a": c.claim_a_text[:80], "b": c.claim_b_text[:80]}
            for c in conflicts[:10]
        ],
    }

    report = EvalReport(
        run_id=run_id,
        app_type=app_type,
        core_health=core,
        evidence_quality=evidence,
        evidence_density=round(density, 3),
        citation_gaps=len(gaps),
        claim_confidence_avg=round(conf_avg, 3),
        conflict_count=len(conflicts),
        quality_enhanced=quality_enhanced,
    )

    eval_enhanced_path = run_dir / "eval_enhanced.json"
    eval_enhanced_path.write_text(
        json.dumps(report.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )

    return report


# ── Before/After Comparison ────────────────────────────────────────────

def compare_runs(run_a_dir: Path, run_b_dir: Path) -> ComparisonReport:
    """Compare two runs and produce a delta report.

    Useful for demonstrating improvement after tuning retrieval, prompts,
    or quality thresholds.
    """
    report_a = evaluate_run(run_a_dir)
    report_b = evaluate_run(run_b_dir)

    metric_pairs = [
        ("citation_coverage", report_a.evidence_quality.get("citation_coverage", 0),
         report_b.evidence_quality.get("citation_coverage", 0)),
        ("evidence_density", report_a.evidence_density, report_b.evidence_density),
        ("citation_gaps", report_a.citation_gaps, report_b.citation_gaps),
        ("claim_confidence_avg", report_a.claim_confidence_avg, report_b.claim_confidence_avg),
        ("conflict_count", report_a.conflict_count, report_b.conflict_count),
        ("tool_calls", report_a.core_health.get("tool_calls", 0),
         report_b.core_health.get("tool_calls", 0)),
        ("cache_hit_rate", report_a.core_health.get("cache_hit_rate", 0),
         report_b.core_health.get("cache_hit_rate", 0)),
    ]

    deltas: list[MetricDelta] = []
    improvements = 0
    for name, before, after in metric_pairs:
        delta = after - before
        lower_is_better = name in ("citation_gaps", "conflict_count")
        improved = delta < 0 if lower_is_better else delta > 0
        if improved:
            improvements += 1
        deltas.append(MetricDelta(
            name=name, before=round(before, 3), after=round(after, 3),
            delta=round(delta, 3), improved=improved,
        ))

    total = len(deltas)
    summary = (
        f"{improvements}/{total} metrics improved between "
        f"{report_a.run_id} and {report_b.run_id}"
    )

    return ComparisonReport(
        run_a_id=report_a.run_id,
        run_b_id=report_b.run_id,
        deltas=deltas,
        summary=summary,
    )


# ── Evalset Runner ─────────────────────────────────────────────────────

def run_evalset(
    evalset_path: Path,
    results_dir: Path | None = None,
) -> list[EvalReport]:
    """Run evaluation on a set of pre-completed runs listed in a JSONL file.

    Each line in the evalset file should have at minimum ``{"run_dir": "..."}``
    and optionally ``{"app_type": "research"|"market"}``.
    """
    results_dir = results_dir or evalset_path.parent / "eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    reports: list[EvalReport] = []
    for line in evalset_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping invalid evalset line: %s", line[:100])
            continue

        run_dir = Path(entry.get("run_dir", ""))
        if not run_dir.exists():
            logger.warning("Run dir not found: %s", run_dir)
            continue

        app_type = entry.get("app_type", "auto")
        report = evaluate_run(run_dir, app_type=app_type)
        reports.append(report)

    if reports:
        summary_path = results_dir / "evalset_summary.json"
        summary_path.write_text(
            json.dumps(
                {"total_runs": len(reports),
                 "results": [r.to_dict() for r in reports]},
                indent=2, default=str,
            ),
            encoding="utf-8",
        )

    return reports


# ── Internal Helpers ───────────────────────────────────────────────────

def _detect_app_type(run_dir: Path) -> str:
    plan_path = run_dir / "plan.json"
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            if "ticker" in plan or "analysis_type" in plan:
                return "market"
        except Exception:
            pass
    if "quant" in run_dir.name:
        return "market"
    return "research"


def _load_all_claims(run_dir: Path) -> list[dict[str, Any]]:
    notes_dir = run_dir / "notes"
    if not notes_dir.exists():
        return []
    claims: list[dict[str, Any]] = []
    for f in sorted(notes_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for c in data.get("claims", []):
                c.setdefault("source_id", data.get("source_id", ""))
                claims.append(c)
        except Exception:
            continue
    return claims


def _confidence_distribution(scores: list) -> dict[str, int]:
    buckets = {"high": 0, "medium": 0, "low": 0}
    for s in scores:
        if s.confidence >= 0.7:
            buckets["high"] += 1
        elif s.confidence >= 0.4:
            buckets["medium"] += 1
        else:
            buckets["low"] += 1
    return buckets
