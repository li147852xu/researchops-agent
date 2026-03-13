"""Tests for the prompt template system and JSON parser."""

from __future__ import annotations

from researchops.prompts.parser import parse_json_response, parse_json_safe
from researchops.prompts.templates import (
    COLLECTOR_QUERIES,
    JUDGE_FAITHFULNESS,
    PLANNER_BUCKETS,
    PLANNER_RQS,
    REACT_THOUGHT,
    READER_CLAIMS,
    WRITER_SECTION,
    PromptTemplate,
)


class TestPromptTemplate:
    def test_basic_render(self):
        tpl = PromptTemplate(
            name="test",
            system="You are a helper.",
            user="Answer about {topic}.",
        )
        sys_msg, user_msg = tpl.render(topic="AI")
        assert sys_msg == "You are a helper."
        assert "AI" in user_msg

    def test_few_shot_render(self):
        tpl = PromptTemplate(
            name="test_fs",
            system="System.",
            user="Do {task}.",
            few_shot=[{"input": "hello", "output": "world"}],
        )
        sys_msg, user_msg = tpl.render(task="something")
        assert "Example input:" in user_msg
        assert "hello" in user_msg
        assert "world" in user_msg
        assert "something" in user_msg

    def test_planner_rqs_render(self):
        sys_msg, user_msg = PLANNER_RQS.render(num_rqs="3", topic="deep learning")
        assert "deep learning" in user_msg
        assert "3" in user_msg
        assert "research" in sys_msg.lower()

    def test_planner_buckets_render(self):
        sys_msg, user_msg = PLANNER_BUCKETS.render(topic="NLP")
        assert "NLP" in user_msg

    def test_collector_queries_render(self):
        sys_msg, user_msg = COLLECTOR_QUERIES.render(
            rq_text="How does attention work?",
            topic="transformers",
            neg_hint="",
        )
        assert "attention" in user_msg
        assert "transformers" in user_msg

    def test_react_thought_render(self):
        sys_msg, user_msg = REACT_THOUGHT.render(
            topic="RL", gap_rqs="rq_1, rq_2",
            current_sources="3", target_sources="10",
            uncovered_buckets="exploration, model-based",
            history="none",
        )
        assert "RL" in user_msg
        assert "ReAct" in sys_msg

    def test_reader_claims_render(self):
        sys_msg, user_msg = READER_CLAIMS.render(
            title="Test Paper", chunk="Some text about methods.",
            rq_list="- rq_1: What methods exist?",
        )
        assert "Test Paper" in user_msg
        assert "methods" in user_msg

    def test_writer_section_render(self):
        sys_msg, user_msg = WRITER_SECTION.render(
            heading="Methods", lang="English",
            claims_block="[src1] Method A works well.",
            deep_instruction="",
        )
        assert "Methods" in user_msg
        assert "[@source_id]" in sys_msg

    def test_judge_faithfulness_render(self):
        sys_msg, user_msg = JUDGE_FAITHFULNESS.render(
            report="The model is great.", claims="[src1] The model is great.",
        )
        assert "faithfulness" in sys_msg.lower()
        assert "score" in user_msg.lower()


class TestParser:
    def test_parse_direct_json(self):
        raw = '{"key": "value"}'
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_parse_markdown_fenced(self):
        raw = "Here is the result:\n```json\n{\"key\": \"value\"}\n```"
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_parse_embedded_json(self):
        raw = "Some text before {\"key\": \"value\"} and some after."
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_parse_nested_json(self):
        raw = '{"outer": {"inner": [1, 2, 3]}}'
        result = parse_json_response(raw)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_parse_safe_fallback(self):
        raw = "this is not json at all"
        result = parse_json_safe(raw)
        assert result == {}

    def test_parse_with_preamble(self):
        raw = "Sure! Here is the JSON:\n\n{\"questions\": [{\"rq_id\": \"rq_1\", \"text\": \"What?\"}]}"
        result = parse_json_response(raw)
        assert "questions" in result
        assert len(result["questions"]) == 1

    def test_parse_empty_returns_empty(self):
        result = parse_json_safe("")
        assert result == {}
