"""LLM-as-Judge evaluator — RAGAS-inspired quality scoring."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from researchops.models import JudgeResult, JudgeScore
from researchops.prompts import (
    JUDGE_COHERENCE,
    JUDGE_COVERAGE,
    JUDGE_FAITHFULNESS,
    JUDGE_RELEVANCE,
    parse_json_response,
)
from researchops.utils import load_all_notes, load_plan

if TYPE_CHECKING:
    from researchops.reasoning.base import ReasonerBase

logger = logging.getLogger(__name__)


class LLMJudge:
    """Evaluate a research report using LLM-as-judge across four dimensions:

    - **Faithfulness**: Are claims grounded in source evidence?
    - **Coverage**: Are all research questions addressed?
    - **Coherence**: Is the report well-structured and non-redundant?
    - **Relevance**: Is every section on-topic?
    """

    def __init__(self, reasoner: ReasonerBase):
        self.reasoner = reasoner

    def evaluate(self, run_dir: Path) -> JudgeResult:
        if not self.reasoner.is_llm:
            logger.info("LLM judge skipped — reasoner is not an LLM")
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

        faithfulness = self._judge_dimension(
            JUDGE_FAITHFULNESS, report=report_excerpt, claims=claims_text[:3000],
        )
        coverage = self._judge_dimension(
            JUDGE_COVERAGE, topic=topic, rq_list=rq_list, report=report_excerpt,
        )
        coherence = self._judge_dimension(
            JUDGE_COHERENCE, report=report_excerpt,
        )
        relevance = self._judge_dimension(
            JUDGE_RELEVANCE, topic=topic, report=report_excerpt,
        )

        overall = (
            faithfulness.score * 0.3
            + coverage.score * 0.3
            + coherence.score * 0.2
            + relevance.score * 0.2
        )

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
            faithfulness=faithfulness.score,
            coverage=coverage.score,
            coherence=coherence.score,
            relevance=relevance.score,
            overall=round(overall, 3),
            reasoning=" | ".join(reasoning_parts),
        )

        judge_path = run_dir / "judge_result.json"
        judge_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

        logger.info(
            "LLM Judge: faithfulness=%.2f coverage=%.2f coherence=%.2f relevance=%.2f overall=%.2f",
            result.faithfulness, result.coverage, result.coherence, result.relevance, result.overall,
        )
        return result

    def _judge_dimension(self, template, **kwargs) -> JudgeScore:
        try:
            sys_msg, user_msg = template.render(**kwargs)
            raw = self.reasoner.complete_text(user_msg, context=sys_msg)
            data = parse_json_response(raw)
            score = float(data.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            reasoning = str(data.get("reasoning", ""))
            return JudgeScore(score=score, reasoning=reasoning)
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
