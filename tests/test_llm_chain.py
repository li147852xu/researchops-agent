"""Tests for the LLM invocation chain — key resolution, base_url normalization, trace events, fail-fast."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from researchops.core.tracing import TraceLogger

# ── D1: Key resolution ──────────────────────────────────────────────


def test_key_from_llm_api_key_env():
    """When only LLM_API_KEY is set, OpenAICompatReasoner should pick it up."""
    env = {"LLM_API_KEY": "test-key-from-env", "OPENAI_API_KEY": "", "DEEPSEEK_API_KEY": ""}
    with patch.dict(os.environ, env, clear=False):
        from researchops.reasoning.openai_compat import OpenAICompatReasoner

        r = OpenAICompatReasoner(api_key="", model="gpt-4o-mini")
        assert r.api_key == "test-key-from-env"


def test_key_from_deepseek_env():
    """When only DEEPSEEK_API_KEY is set, OpenAICompatReasoner should pick it up."""
    env = {"DEEPSEEK_API_KEY": "sk-deep", "OPENAI_API_KEY": "", "LLM_API_KEY": ""}
    with patch.dict(os.environ, env, clear=False):
        from researchops.reasoning.openai_compat import OpenAICompatReasoner

        r = OpenAICompatReasoner(api_key="", model="deepseek-chat", base_url="https://api.deepseek.com")
        assert r.api_key == "sk-deep"


def test_cli_key_takes_precedence():
    """Explicit api_key arg should override env vars."""
    env = {"OPENAI_API_KEY": "env-key", "DEEPSEEK_API_KEY": "env-deep"}
    with patch.dict(os.environ, env, clear=False):
        from researchops.reasoning.openai_compat import OpenAICompatReasoner

        r = OpenAICompatReasoner(api_key="cli-key", model="gpt-4o-mini")
        assert r.api_key == "cli-key"


# ── D2: base_url normalization ───────────────────────────────────────


def test_base_url_with_v1():
    from researchops.reasoning.openai_compat import _normalize_base_url

    assert _normalize_base_url("https://api.deepseek.com/v1") == "https://api.deepseek.com/v1"


def test_base_url_without_v1():
    from researchops.reasoning.openai_compat import _normalize_base_url

    assert _normalize_base_url("https://api.deepseek.com") == "https://api.deepseek.com/v1"


def test_base_url_trailing_slash():
    from researchops.reasoning.openai_compat import _normalize_base_url

    assert _normalize_base_url("https://api.deepseek.com/v1/") == "https://api.deepseek.com/v1"


def test_base_url_openai_default():
    from researchops.reasoning.openai_compat import _normalize_base_url

    assert _normalize_base_url("https://api.openai.com/v1") == "https://api.openai.com/v1"


# ── D3: llm.call trace events ───────────────────────────────────────


def test_llm_trace_events_on_complete_text(tmp_run_dir: Path):
    """Mock httpx and verify that llm.call + llm.result appear in trace."""
    trace = TraceLogger(tmp_run_dir / "trace.jsonl")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    with patch("httpx.post", return_value=mock_response):
        from researchops.reasoning.openai_compat import OpenAICompatReasoner

        r = OpenAICompatReasoner(
            api_key="test-key",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            provider_label="deepseek",
        )
        result = r.complete_text("What is quantum computing?", trace=trace)

    assert result == "Test response"

    events = trace.read_all()
    actions = [e.action for e in events]
    assert "llm.call" in actions, f"Expected llm.call in trace, got: {actions}"
    assert "llm.result" in actions, f"Expected llm.result in trace, got: {actions}"

    call_event = next(e for e in events if e.action == "llm.call")
    assert call_event.meta["provider_label"] == "deepseek"
    assert call_event.meta["model"] == "deepseek-chat"
    assert call_event.meta["input_chars"] > 0

    result_event = next(e for e in events if e.action == "llm.result")
    assert result_event.meta["tokens"] == 15
    assert result_event.meta["output_chars"] > 0


def test_llm_trace_events_on_complete_json(tmp_run_dir: Path):
    """Mock httpx and verify that llm.call + llm.result appear for complete_json."""
    from pydantic import BaseModel

    class TestSchema(BaseModel):
        answer: str = ""

    trace = TraceLogger(tmp_run_dir / "trace.jsonl")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"answer": "42"}'}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
    }

    with patch("httpx.post", return_value=mock_response):
        from researchops.reasoning.openai_compat import OpenAICompatReasoner

        r = OpenAICompatReasoner(api_key="test-key", model="gpt-4o-mini")
        result = r.complete_json(TestSchema, "Answer the question", trace=trace)

    assert result.answer == "42"

    events = trace.read_all()
    actions = [e.action for e in events]
    assert "llm.call" in actions
    assert "llm.result" in actions


# ── D4: No silent fallback ───────────────────────────────────────────


def test_missing_key_raises_error():
    """When no key is set, OpenAICompatReasoner must raise ValueError, not silently fallback."""
    env = {"OPENAI_API_KEY": "", "LLM_API_KEY": "", "DEEPSEEK_API_KEY": ""}
    with patch.dict(os.environ, env, clear=False):
        from researchops.reasoning.openai_compat import OpenAICompatReasoner

        with pytest.raises(ValueError, match="API key required"):
            OpenAICompatReasoner(api_key="", model="gpt-4o-mini")


def test_is_llm_property():
    """OpenAICompatReasoner.is_llm should return True, NoneReasoner.is_llm should return False."""
    from researchops.reasoning.none import NoneReasoner
    from researchops.reasoning.openai_compat import OpenAICompatReasoner

    assert NoneReasoner().is_llm is False

    env = {"OPENAI_API_KEY": "test-key"}
    with patch.dict(os.environ, env, clear=False):
        r = OpenAICompatReasoner(api_key="test-key", model="gpt-4o-mini")
        assert r.is_llm is True


# ── Endpoint construction ────────────────────────────────────────────


def test_endpoint_construction():
    """Verify the endpoint is correctly built."""
    env = {"OPENAI_API_KEY": "test-key"}
    with patch.dict(os.environ, env, clear=False):
        from researchops.reasoning.openai_compat import OpenAICompatReasoner

        r = OpenAICompatReasoner(
            api_key="test-key",
            base_url="https://api.deepseek.com",
        )
        assert r._endpoint() == "https://api.deepseek.com/v1/chat/completions"

        r2 = OpenAICompatReasoner(
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
        )
        assert r2._endpoint() == "https://api.deepseek.com/v1/chat/completions"
