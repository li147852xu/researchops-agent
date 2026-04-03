"""Tests for LLM-as-Judge evaluator."""

from __future__ import annotations

import json
from pathlib import Path

from researchops.apps.research.evaluators import LLMJudge
from researchops.apps.research.schemas import JudgeResult


class MockReasoner:
    is_llm = True
    _response: str = '{"score": 0.8, "reasoning": "Good quality."}'

    def complete_text(self, prompt: str, context: str = "", trace=None) -> str:
        return self._response


class MockReasonerNoLLM:
    is_llm = False


class TestLLMJudge:
    def test_judge_skips_non_llm(self, tmp_path: Path):
        reasoner = MockReasonerNoLLM()
        judge = LLMJudge(reasoner)
        result = judge.evaluate(tmp_path)
        assert result.overall == 0.0

    def test_judge_skips_no_report(self, tmp_path: Path):
        reasoner = MockReasoner()
        judge = LLMJudge(reasoner)
        result = judge.evaluate(tmp_path)
        assert result.overall == 0.0

    def test_judge_evaluates_report(self, tmp_path: Path):
        (tmp_path / "report.md").write_text(
            "# Test\n\nThis is a test report about AI.", encoding="utf-8"
        )
        plan = {
            "topic": "artificial intelligence",
            "research_questions": [
                {"rq_id": "rq_1", "text": "What is AI?", "priority": 1, "needs_verification": False}
            ],
            "outline": [],
            "acceptance_threshold": 0.7,
            "coverage_checklist": [],
        }
        (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

        reasoner = MockReasoner()
        judge = LLMJudge(reasoner)
        result = judge.evaluate(tmp_path)

        assert result.faithfulness == 0.8
        assert result.coverage == 0.8
        assert result.coherence == 0.8
        assert result.relevance == 0.8
        assert result.overall > 0.0

        judge_path = tmp_path / "judge_result.json"
        assert judge_path.exists()

    def test_judge_handles_error(self, tmp_path: Path):
        (tmp_path / "report.md").write_text("# Test\nContent.", encoding="utf-8")
        plan = {
            "topic": "test",
            "research_questions": [],
            "outline": [],
            "acceptance_threshold": 0.7,
            "coverage_checklist": [],
        }
        (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

        reasoner = MockReasoner()
        reasoner._response = "not json at all"
        judge = LLMJudge(reasoner)
        result = judge.evaluate(tmp_path)
        assert result.overall == 0.0

    def test_judge_result_model(self):
        result = JudgeResult(
            faithfulness=0.9, coverage=0.8, coherence=0.7,
            relevance=0.85, overall=0.82, reasoning="Good report.",
        )
        assert result.overall == 0.82
        dumped = result.model_dump()
        assert "faithfulness" in dumped
