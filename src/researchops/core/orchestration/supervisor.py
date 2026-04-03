"""Supervisor — analyses diagnostics and decides rollback strategy."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from researchops.core.state import Decision, StateSnapshot
from researchops.utils import get_negative_terms, load_plan

if TYPE_CHECKING:
    from researchops.core.tracing import TraceLogger
    from researchops.reasoning.base import ReasonerBase


logger = logging.getLogger(__name__)


class Supervisor:
    """Evaluate pipeline health and produce a corrective Decision."""

    def __init__(self, run_dir: Path, reasoner: ReasonerBase):
        self.run_dir = run_dir
        self.reasoner = reasoner

    def decide(
        self,
        state: StateSnapshot,
        diagnostics: dict[str, Any],
        config: Any,
        trace: TraceLogger,
    ) -> Decision:
        reason_codes = self._compute_reason_codes(state, diagnostics, config)
        queries, neg_terms, categories = self._plan_remediation(
            state, diagnostics, config, reason_codes, trace,
        )
        confidence = self._compute_confidence(reason_codes, diagnostics)

        decision = Decision(
            decision_id=str(uuid.uuid4())[:8],
            reason_codes=reason_codes,
            suggested_queries=queries,
            suggested_neg_terms=neg_terms,
            suggested_categories=categories,
            confidence=confidence,
        )

        decision_path = self.run_dir / "last_decision.json"
        decision_path.write_text(decision.model_dump_json(indent=2), encoding="utf-8")

        trace.log(
            stage="SUPERVISOR", action="decide",
            output_summary=f"Decision: {len(reason_codes)} reason codes, confidence={confidence:.2f}",
            meta={
                "reason_codes": reason_codes,
                "suggested_queries": queries[:3],
                "confidence": confidence,
            },
        )
        return decision

    def _compute_reason_codes(
        self, state: StateSnapshot, diag: dict, config: Any,
    ) -> list[str]:
        codes: list[str] = []

        coverage_vec = diag.get("coverage_vector", state.coverage_vector)
        plan = load_plan(self.run_dir)
        if plan:
            for rq in plan.research_questions:
                claim_count = coverage_vec.get(rq.rq_id, 0)
                if claim_count < config.min_claims_per_rq:
                    codes.append("coverage_gap")
                    break

            threshold = plan.acceptance_threshold
            total_rqs = len(plan.research_questions)
            covered = sum(1 for rq in plan.research_questions if coverage_vec.get(rq.rq_id, 0) > 0)
            if total_rqs > 0 and covered / total_rqs < threshold:
                codes.append("coverage_below_threshold")

        if diag.get("bucket_coverage_rate", 1.0) < 0.5:
            codes.append("bucket_incomplete")

        if diag.get("relevance_avg", 1.0) < config.relevance_threshold:
            codes.append("low_relevance")

        return codes

    def _plan_remediation(
        self, state: StateSnapshot, diag: dict, config: Any,
        reason_codes: list[str], trace: TraceLogger,
    ) -> tuple[list[str], list[str], list[str]]:
        from researchops.apps.research.prompts import SUPERVISOR_PLAN, parse_json_response

        plan = load_plan(self.run_dir)
        topic = plan.topic if plan else config.topic

        neg_terms = get_negative_terms(topic)
        queries: list[str] = []
        categories: list[str] = []

        if self.reasoner.is_llm and reason_codes:
            try:
                coverage_summary = ""
                if plan:
                    coverage_vec = diag.get("coverage_vector", state.coverage_vector)
                    parts = []
                    for rq in plan.research_questions:
                        cnt = coverage_vec.get(rq.rq_id, 0)
                        parts.append(f"{rq.rq_id}: {cnt} claims")
                    coverage_summary = ", ".join(parts)

                sys_msg, user_msg = SUPERVISOR_PLAN.render(
                    topic=topic,
                    reason_codes=str(reason_codes),
                    coverage_summary=coverage_summary or "no data",
                )
                raw = self.reasoner.complete_text(user_msg, context=sys_msg, trace=trace)
                data = parse_json_response(raw)
                queries = data.get("queries", [])[:5]
                neg_terms = data.get("negative_terms", neg_terms)[:5]
                categories = data.get("categories", [])[:3]
            except Exception as exc:
                logger.warning("LLM supervisor planning failed: %s", exc)

        if not queries:
            queries = self._default_queries(topic, plan)
        return queries, neg_terms, categories

    def _default_queries(self, topic: str, plan) -> list[str]:
        queries = [f"{topic} survey", f"{topic} recent advances"]
        if plan:
            for rq in plan.research_questions[:2]:
                words = rq.text.split()[:5]
                queries.append(" ".join(words))
        return queries

    def _compute_confidence(self, reason_codes: list[str], diag: dict) -> float:
        if not reason_codes:
            return 1.0
        penalty = 0.15 * len(reason_codes)
        bcr = diag.get("bucket_coverage_rate", 1.0)
        rel = diag.get("relevance_avg", 1.0)
        return max(0.1, min(1.0, 1.0 - penalty + bcr * 0.1 + rel * 0.1))
