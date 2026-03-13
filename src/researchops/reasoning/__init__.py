from __future__ import annotations

import json
from typing import TYPE_CHECKING

from researchops.reasoning.base import ReasonerBase
from researchops.reasoning.none import NoneReasoner

if TYPE_CHECKING:
    from researchops.config import RunConfig

__all__ = ["ReasonerBase", "NoneReasoner", "create_reasoner"]


def create_reasoner(config: RunConfig) -> ReasonerBase:
    llm = config.llm
    if llm == "none":
        return NoneReasoner()
    elif llm in ("openai", "openai_compat"):
        import contextlib

        from researchops.reasoning.openai_compat import OpenAICompatReasoner

        extra_headers: dict[str, str] = {}
        if config.llm_headers:
            with contextlib.suppress(json.JSONDecodeError):
                extra_headers = json.loads(config.llm_headers)

        return OpenAICompatReasoner(
            api_key=config.llm_api_key,
            model=config.llm_model or "gpt-4o-mini",
            base_url=config.llm_base_url or "https://api.openai.com/v1",
            provider_label=config.llm_provider_label,
            extra_headers=extra_headers,
        )
    elif llm == "anthropic":
        from researchops.reasoning.anthropic_r import AnthropicReasoner

        return AnthropicReasoner(
            api_key=config.llm_api_key,
            model=config.llm_model or "claude-sonnet-4-20250514",
            base_url=config.llm_base_url or "https://api.anthropic.com",
        )
    else:
        raise ValueError(f"Unknown LLM backend: {llm!r}. Use none/openai/openai_compat/anthropic.")
