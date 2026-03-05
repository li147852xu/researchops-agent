from __future__ import annotations

import json
import os
import time

from pydantic import BaseModel

from researchops.reasoning.base import ReasonerBase


class OpenAIReasoner(ReasonerBase):
    """OpenAI-compatible API reasoner via httpx."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        super().__init__()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set --llm-api-key or OPENAI_API_KEY env var, "
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

        system = "You are a research assistant. Return valid JSON matching the requested schema."
        user_msg = prompt
        if context:
            user_msg = f"{prompt}\n\nContext:\n{context}"

        schema_hint = json.dumps(schema.model_json_schema(), indent=2)
        user_msg += f"\n\nJSON Schema:\n{schema_hint}"

        for attempt in range(3):
            t0 = time.monotonic()
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.3,
                },
                timeout=60,
            )
            elapsed = (time.monotonic() - t0) * 1000
            resp.raise_for_status()
            data = resp.json()
            tokens = data.get("usage", {}).get("total_tokens", 0)
            self._record_call(tokens, elapsed)

            content = data["choices"][0]["message"]["content"]
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
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": user_msg}],
                "temperature": 0.5,
            },
            timeout=60,
        )
        elapsed = (time.monotonic() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()
        tokens = data.get("usage", {}).get("total_tokens", 0)
        self._record_call(tokens, elapsed)
        return data["choices"][0]["message"]["content"]
