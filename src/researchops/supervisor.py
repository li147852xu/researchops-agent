from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from researchops.models import Decision, StateSnapshot

if TYPE_CHECKING:
    from researchops.config import RunConfig
    from researchops.reasoning.base import ReasonerBase
    from researchops.trace import TraceLogger

_VALID_REASON_CODES = {
    "coverage_gap",
    "relevance_drift",
    "source_quality_low",
    "domination",
    "unsupported_high",
    "conflict_found",
    "no_progress_loop",
    "bucket_incomplete",
}


class Supervisor:
    """Policy advisor that the orchestrator consults before rollback decisions."""

    def __init__(self, run_dir: Path, reasoner: ReasonerBase | None = None):
        self.run_dir = run_dir
        self.reasoner = reasoner
        self._decisions_path = run_dir / "decisions.jsonl"

    def decide(
        self,
        state: StateSnapshot,
        diagnostics: dict[str, Any],
        config: RunConfig,
        trace: TraceLogger,
    ) -> Decision:
        reason_codes = self._compute_reason_codes(state, diagnostics, config)
        action_plan = self._build_action_plan(
            reason_codes, diagnostics, config, state, trace,
        )
        confidence = self._compute_confidence(reason_codes, diagnostics)

        decision = Decision(
            decision_id=f"dec_{uuid.uuid4().hex[:8]}",
            stage=state.stage.value,
            reason_codes=reason_codes,
            action_plan=action_plan,
            confidence=confidence,
        )

        self._append_decision(decision)

        trace.log(
            stage="SUPERVISOR",
            action="decision",
            output_summary=f"reason_codes={reason_codes}, confidence={confidence:.2f}",
            meta={
                "decision_id": decision.decision_id,
                "reason_codes": reason_codes,
                "action_plan": action_plan,
                "confidence": confidence,
            },
        )

        return decision

    def _compute_reason_codes(
        self,
        state: StateSnapshot,
        diagnostics: dict[str, Any],
        config: RunConfig,
    ) -> list[str]:
        codes: list[str] = []

        bucket_cov = diagnostics.get("bucket_coverage_rate", 1.0)
        if bucket_cov < config.bucket_coverage_threshold:
            codes.append("bucket_incomplete")

        relevance_avg = diagnostics.get("relevance_avg", 1.0)
        if relevance_avg < config.relevance_threshold:
            codes.append("relevance_drift")

        coverage_vector = diagnostics.get("coverage_vector", state.coverage_vector)
        if coverage_vector:
            underfilled = sum(1 for v in coverage_vector.values() if v < config.min_claims_per_rq)
            if underfilled > 0:
                codes.append("coverage_gap")

        if diagnostics.get("low_quality_rate", 0) > 0.4:
            codes.append("source_quality_low")

        if diagnostics.get("domination_issues"):
            codes.append("domination")

        unsupported = diagnostics.get("unsupported_rate", 0)
        if unsupported > 0.15:
            codes.append("unsupported_high")

        if diagnostics.get("conflict_count", 0) > 0:
            codes.append("conflict_found")

        if state.no_progress_streak >= 2:
            codes.append("no_progress_loop")

        return codes

    def _build_action_plan(
        self,
        reason_codes: list[str],
        diagnostics: dict[str, Any],
        config: RunConfig,
        state: StateSnapshot,
        trace: TraceLogger,
    ) -> dict[str, Any]:
        queries: list[str] = []
        categories: list[str] = []
        negative_terms: list[str] = []
        target_per_bucket: dict[str, int] = {}

        topic = config.topic

        if "bucket_incomplete" in reason_codes:
            checklist = diagnostics.get("coverage_checklist", [])
            bucket_cov = diagnostics.get("bucket_status", {})
            for bucket in checklist:
                bid = bucket.get("bucket_id", "")
                status = bucket_cov.get(bid, {})
                if status.get("sources", 0) < config.target_sources_per_bucket:
                    bname = bucket.get("bucket_name", bid)
                    queries.append(f"{topic} {bname}")
                    target_per_bucket[bid] = config.target_sources_per_bucket

        if "relevance_drift" in reason_codes:
            negative_terms = self._infer_negative_terms(topic)
            queries.append(f"{topic} survey core concepts")

        if "coverage_gap" in reason_codes and not queries:
            queries.append(f"{topic} comprehensive overview")
            queries.append(f"{topic} recent advances")

        if "domination" in reason_codes:
            queries.append(f"{topic} alternative approaches comparison")

        if "conflict_found" in reason_codes and config.allow_net:
            queries.append(f"{topic} consensus review meta-analysis")

        if self.reasoner and self.reasoner.is_llm and reason_codes:
            llm_plan = self._llm_action_plan(reason_codes, topic, trace)
            if llm_plan:
                queries.extend(llm_plan.get("queries", []))
                negative_terms.extend(llm_plan.get("negative_terms", []))
                categories.extend(llm_plan.get("categories", []))

        return {
            "action": "rollback_collect" if queries else "proceed",
            "queries": queries[:10],
            "categories": categories[:5],
            "target_per_bucket": target_per_bucket,
            "negative_terms": negative_terms[:10],
        }

    def _llm_action_plan(
        self,
        reason_codes: list[str],
        topic: str,
        trace: TraceLogger,
    ) -> dict[str, Any] | None:
        prompt = (
            f"You are a research strategy advisor. Given the topic '{topic}' and "
            f"the following issues: {reason_codes}, suggest:\n"
            f"1. 3-5 search queries to fix coverage gaps\n"
            f"2. 3-5 negative terms to exclude off-topic results\n"
            f"3. 1-3 arXiv categories to search\n"
            f"Return JSON: {{\"queries\": [...], \"negative_terms\": [...], \"categories\": [...]}}"
        )
        try:
            raw = self.reasoner.complete_text(prompt, trace=trace)
            start = raw.find("{")
            if start >= 0:
                return json.loads(raw[start:])
        except Exception:
            pass
        return None

    def _infer_negative_terms(self, topic: str) -> list[str]:
        topic_lower = topic.lower()
        negatives: dict[str, list[str]] = {
            "deep learning": ["collider", "particle physics", "astronomy", "geology", "marine biology"],
            "machine learning": ["collider", "particle physics", "astronomy", "deep sea"],
            "quantum computing": ["quantum healing", "quantum mysticism", "spirituality"],
            "natural language processing": ["chemical processing", "food processing", "manufacturing"],
            "computer vision": ["ophthalmology", "optometry", "eye surgery"],
            "reinforcement learning": ["positive reinforcement parenting", "dog training"],
        }
        for key, terms in negatives.items():
            if key in topic_lower:
                return terms
        return []

    def _compute_confidence(
        self, reason_codes: list[str], diagnostics: dict[str, Any],
    ) -> float:
        if not reason_codes:
            return 1.0
        base = 0.9
        base -= 0.1 * len(reason_codes)
        if "no_progress_loop" in reason_codes:
            base -= 0.15
        return max(0.1, round(base, 2))

    def _append_decision(self, decision: Decision) -> None:
        self._decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with self._decisions_path.open("a", encoding="utf-8") as f:
            f.write(decision.model_dump_json() + "\n")
