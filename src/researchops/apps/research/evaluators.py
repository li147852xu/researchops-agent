"""Research-specific evaluation — app metrics + LLM-as-judge.

Core health and evidence quality metrics live in researchops.core.evaluation.base.
This module adds research-task-specific metrics (bucket coverage, per-RQ, judge scores).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from researchops.apps.research.schemas import (
    EvalResult,
    JudgeResult,
    JudgeScore,
    Source,
)
from researchops.core.evaluation.base import (
    compute_core_health,
    compute_evidence_quality,
)
from researchops.core.state import StateSnapshot, TraceEvent
from researchops.core.tracing import TraceLogger
from researchops.utils import load_all_notes, load_plan, load_sources

if TYPE_CHECKING:
    from researchops.apps.research.config import RunConfig
    from researchops.reasoning.base import ReasonerBase

logger = logging.getLogger(__name__)


def compute_eval(
    run_dir: Path,
    *,
    config: RunConfig | None = None,
    llm_enabled: bool = False,
    enable_judge: bool = False,
    reasoner: Any | None = None,
) -> EvalResult:
    sources = load_sources(run_dir)
    trace = TraceLogger(run_dir / "trace.jsonl")
    events = trace.read_all()
    state = _load_state(run_dir)

    core = compute_core_health(run_dir)
    evidence = compute_evidence_quality(run_dir)

    repro_rate = _reproduction_rate(events)
    is_llm = llm_enabled or (config is not None and config.llm != "none")
    has_llm_calls = any(e.action in ("llm.call", "llm.result") for e in events)
    is_llm = is_llm or has_llm_calls
    provider_label = config.llm_provider_label if config else ""
    if not provider_label:
        for e in events:
            if e.action == "llm.call" and e.meta.get("provider_label"):
                provider_label = e.meta["provider_label"]
                break

    est_tokens = _estimate_tokens(events)
    est_method = "token_count_from_api" if has_llm_calls else "none"
    papers_rq = _papers_per_rq(sources, run_dir)
    lq_rate = _low_quality_source_rate(events, sources)
    sec_rate = _section_nonempty_rate(run_dir / "report.md")
    incomplete = state.incomplete_sections if state else []
    collect_total = state.collect_rounds if state else 1
    max_rollback = collect_total >= (config.max_collect_rounds if config else 6) if state else False
    src_per_rq = _sources_per_rq(sources, run_dir)
    bucket_cov = _bucket_coverage_rate(run_dir)
    rel_avg = _relevance_avg(run_dir)
    dec_count = _decision_count(run_dir)
    artifacts_count = _count_artifacts(run_dir)

    result = EvalResult(
        citation_coverage=evidence["citation_coverage"],
        source_diversity=evidence["source_diversity"],
        reproduction_rate=repro_rate,
        tool_calls=core["tool_calls"],
        latency_sec=core["latency_sec"],
        steps=core["steps"],
        unsupported_claim_rate=evidence["unsupported_assertion_ratio"],
        cache_hit_rate=core["cache_hit_rate"],
        llm_enabled=is_llm,
        estimated_tokens=est_tokens,
        estimated_cost_usd=0.0,
        estimate_method=est_method,
        conflict_count=evidence["conflict_count"],
        plan_refinement_count=state.refinement_count if state else 0,
        collect_rounds=state.collect_rounds if state else 1,
        artifacts_count=artifacts_count,
        llm_provider_label=provider_label,
        papers_per_rq=papers_rq,
        low_quality_source_rate=lq_rate,
        section_nonempty_rate=sec_rate,
        incomplete_sections=incomplete,
        collect_rounds_total=collect_total,
        sources_per_rq=src_per_rq,
        max_rollback_used=max_rollback,
        bucket_coverage_rate=bucket_cov,
        relevance_avg=rel_avg,
        decision_count=dec_count,
        judge_scores={},
    )

    if enable_judge and reasoner is not None:
        try:
            judge = LLMJudge(reasoner)
            judge_result = judge.evaluate(run_dir)
            result.judge_scores = {
                "faithfulness": judge_result.faithfulness,
                "coverage": judge_result.coverage,
                "coherence": judge_result.coherence,
                "relevance": judge_result.relevance,
                "overall": judge_result.overall,
            }
        except Exception as exc:
            logger.warning("LLM judge evaluation failed: %s", exc)

    eval_path = run_dir / "eval.json"
    eval_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


# ── LLM-as-Judge ───────────────────────────────────────────────────────

class LLMJudge:
    def __init__(self, reasoner: ReasonerBase):
        self.reasoner = reasoner

    def evaluate(self, run_dir: Path) -> JudgeResult:
        from researchops.apps.research.prompts import (
            JUDGE_COHERENCE,
            JUDGE_COVERAGE,
            JUDGE_FAITHFULNESS,
            JUDGE_RELEVANCE,
        )

        if not self.reasoner.is_llm:
            return JudgeResult()
        report_path = run_dir / "report.md"
        if not report_path.exists():
            return JudgeResult()
        report = report_path.read_text(encoding="utf-8")

        plan = load_plan(run_dir)
        notes = load_all_notes(run_dir)
        claims_text = self._build_claims_text(notes)
        topic = plan.topic if plan else "unknown"
        rq_list = "\n".join(f"- {rq.rq_id}: {rq.text}" for rq in plan.research_questions) if plan else ""
        report_excerpt = report[:4000]

        faithfulness = self._judge_dimension(JUDGE_FAITHFULNESS, report=report_excerpt, claims=claims_text[:3000])
        coverage = self._judge_dimension(JUDGE_COVERAGE, topic=topic, rq_list=rq_list, report=report_excerpt)
        coherence = self._judge_dimension(JUDGE_COHERENCE, report=report_excerpt)
        relevance = self._judge_dimension(JUDGE_RELEVANCE, topic=topic, report=report_excerpt)

        overall = faithfulness.score * 0.3 + coverage.score * 0.3 + coherence.score * 0.2 + relevance.score * 0.2
        reasoning_parts = []
        if faithfulness.reasoning:
            reasoning_parts.append(f"Faithfulness: {faithfulness.reasoning}")
        if coverage.reasoning:
            reasoning_parts.append(f"Coverage: {coverage.reasoning}")
        if coherence.reasoning:
            reasoning_parts.append(f"Coherence: {coherence.reasoning}")
        if relevance.reasoning:
            reasoning_parts.append(f"Relevance: {relevance.reasoning}")

        result = JudgeResult(
            faithfulness=faithfulness.score, coverage=coverage.score,
            coherence=coherence.score, relevance=relevance.score,
            overall=round(overall, 3), reasoning=" | ".join(reasoning_parts),
        )
        judge_path = run_dir / "judge_result.json"
        judge_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    def _judge_dimension(self, template, **kwargs) -> JudgeScore:
        from researchops.apps.research.prompts import parse_json_response
        try:
            sys_msg, user_msg = template.render(**kwargs)
            raw = self.reasoner.complete_text(user_msg, context=sys_msg)
            data = parse_json_response(raw)
            score = max(0.0, min(1.0, float(data.get("score", 0.0))))
            return JudgeScore(score=score, reasoning=str(data.get("reasoning", "")))
        except Exception as exc:
            logger.warning("Judge dimension failed: %s", exc)
            return JudgeScore(score=0.0, reasoning=f"evaluation error: {exc}")

    def _build_claims_text(self, notes: dict) -> str:
        parts: list[str] = []
        for sid, n in notes.items():
            for c in n.claims[:5]:
                parts.append(f"[{sid}] {c.text[:150]}")
            if len(parts) > 30:
                break
        return "\n".join(parts)


# ── Research-specific metric helpers ───────────────────────────────────

def _reproduction_rate(events: list[TraceEvent]) -> float:
    execs = [e for e in events if e.action == "sandbox_exec"]
    if not execs:
        return 1.0
    successes = sum(1 for e in events if e.action == "sandbox_success")
    return round(successes / len(execs), 3)


def _estimate_tokens(events: list[TraceEvent]) -> int:
    return sum(e.meta.get("tokens", 0) for e in events if e.action == "llm.result")


def _count_artifacts(run_dir: Path) -> int:
    art_dir = run_dir / "artifacts"
    if not art_dir.exists():
        return 0
    return sum(1 for _ in art_dir.iterdir())


def _papers_per_rq(sources: list[Source], run_dir: Path) -> float:
    arxiv_sources = [s for s in sources if "arxiv" in s.source_type_detail]
    plan = load_plan(run_dir)
    if not plan:
        return 0.0
    rq_count = len(plan.research_questions)
    return round(len(arxiv_sources) / max(1, rq_count), 2)


def _low_quality_source_rate(events: list[TraceEvent], sources: list[Source]) -> float:
    if not sources:
        return 0.0
    lq_count = sum(1 for e in events if e.action == "parse.low_quality")
    return round(lq_count / len(sources), 3)


def _section_nonempty_rate(report_path: Path) -> float:
    if not report_path.exists():
        return 0.0
    text = report_path.read_text(encoding="utf-8")
    sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
    if len(sections) <= 1:
        return 0.0
    content_sections = sections[1:]
    nonempty = 0
    for sec in content_sections:
        paragraphs = [p.strip() for p in sec.split("\n\n") if len(p.strip()) > 30 and not p.strip().startswith("#")]
        if paragraphs:
            nonempty += 1
    return round(nonempty / len(content_sections), 3) if content_sections else 0.0


def _sources_per_rq(sources: list[Source], run_dir: Path) -> dict[str, int]:
    plan = load_plan(run_dir)
    if not plan:
        return {}
    result: dict[str, int] = {}
    for rq in plan.research_questions:
        rq_id = rq.rq_id
        count = sum(1 for s in sources if rq_id in s.source_id or rq_id in (s.query_id if hasattr(s, "query_id") else ""))
        result[rq_id] = count
    return result


def _bucket_coverage_rate(run_dir: Path) -> float:
    qa_path = run_dir / "qa_report.json"
    if qa_path.exists():
        try:
            data = json.loads(qa_path.read_text(encoding="utf-8"))
            return data.get("bucket_coverage_rate", 0.0)
        except Exception:
            pass
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
    qa_path = run_dir / "qa_report.json"
    if qa_path.exists():
        try:
            data = json.loads(qa_path.read_text(encoding="utf-8"))
            val = data.get("relevance_avg", 0.0)
            if val > 0:
                return val
        except Exception:
            pass
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


def _decision_count(run_dir: Path) -> int:
    path = run_dir / "decisions.jsonl"
    if not path.exists():
        return 0
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        return len([line for line in lines if line.strip()])
    except Exception:
        return 0


def _load_state(run_dir: Path) -> StateSnapshot | None:
    path = run_dir / "state.json"
    if not path.exists():
        return None
    try:
        return StateSnapshot.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None
