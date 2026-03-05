from __future__ import annotations

import json
import os
import time

from pydantic import BaseModel

from researchops.reasoning.base import ReasonerBase


class AnthropicReasoner(ReasonerBase):
    """Anthropic Messages API reasoner via httpx."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-20250514",
        base_url: str = "https://api.anthropic.com",
    ) -> None:
        super().__init__()
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set --llm-api-key or ANTHROPIC_API_KEY env var, "
                "or use --llm none for rule-based mode."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete_json(
        self,
        schema: type[BaseModel],
        prompt: str,
        *,
        context: str = "",
    ) -> BaseModel:
        import httpx

        schema_hint = json.dumps(schema.model_json_schema(), indent=2)
        user_msg = prompt
        if context:
            user_msg = f"{prompt}\n\nContext:\n{context}"
        user_msg += f"\n\nReturn ONLY valid JSON matching this schema:\n{schema_hint}"

        for attempt in range(3):
            t0 = time.monotonic()
            resp = httpx.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": user_msg}],
                },
                timeout=60,
            )
            elapsed = (time.monotonic() - t0) * 1000
            resp.raise_for_status()
            data = resp.json()
            in_tok = data.get("usage", {}).get("input_tokens", 0)
            out_tok = data.get("usage", {}).get("output_tokens", 0)
            self._record_call(in_tok + out_tok, elapsed)

            content = data["content"][0]["text"]
            try:
                return schema.model_validate(json.loads(content))
            except Exception:
                if attempt == 2:
                    raise
        raise RuntimeError("Failed to parse JSON after 3 attempts")

    def complete_text(
        self,
        prompt: str,
        *,
        context: str = "",
    ) -> str:
        import httpx

        user_msg = prompt
        if context:
            user_msg = f"{prompt}\n\nContext:\n{context}"

        t0 = time.monotonic()
        resp = httpx.post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": user_msg}],
            },
            timeout=60,
        )
        elapsed = (time.monotonic() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()
        in_tok = data.get("usage", {}).get("input_tokens", 0)
        out_tok = data.get("usage", {}).get("output_tokens", 0)
        self._record_call(in_tok + out_tok, elapsed)
        return data["content"][0]["text"]
