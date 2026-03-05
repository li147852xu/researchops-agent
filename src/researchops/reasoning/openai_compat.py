from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from researchops.reasoning.base import ReasonerBase

if TYPE_CHECKING:
    from researchops.trace import TraceLogger

_KEY_ENV_VARS = ("OPENAI_API_KEY", "LLM_API_KEY", "DEEPSEEK_API_KEY")

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _normalize_base_url(raw: str) -> str:
    url = raw.rstrip("/")
    if url.endswith("/v1"):
        return url
    if "/v1/" in url:
        return url.rsplit("/v1/", 1)[0] + "/v1"
    return url + "/v1"


class OpenAICompatReasoner(ReasonerBase):
    """OpenAI-compatible API reasoner (covers OpenAI, DeepSeek, OpenRouter, vLLM, Ollama, Azure)."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        provider_label: str = "",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        resolved_key = api_key
        if not resolved_key:
            for env_var in _KEY_ENV_VARS:
                resolved_key = os.environ.get(env_var, "")
                if resolved_key:
                    break
        if not resolved_key:
            raise ValueError(
                "API key required for OpenAI-compatible endpoint. "
                "Set --llm-api-key, or one of OPENAI_API_KEY / LLM_API_KEY / DEEPSEEK_API_KEY env vars, "
                "or use --llm none for rule-based mode."
            )
        self.api_key = resolved_key
        self.model = model
        self.base_url = _normalize_base_url(base_url)
        self.provider_label = provider_label or self._infer_provider(base_url)
        self.extra_headers = extra_headers or {}

    @property
    def is_llm(self) -> bool:
        return True

    def _infer_provider(self, base_url: str) -> str:
        url = base_url.lower()
        if "deepseek" in url:
            return "deepseek"
        if "openrouter" in url:
            return "openrouter"
        if "azure" in url:
            return "azure"
        if "localhost" in url or "127.0.0.1" in url:
            return "local"
        return "openai"

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)
        return headers

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _log_call(self, trace: TraceLogger | None, prompt_len: int) -> None:
        if trace:
            trace.log(
                action="llm.call",
                meta={
                    "provider_label": self.provider_label,
                    "base_url": self.base_url,
                    "model": self.model,
                    "input_chars": prompt_len,
                    "estimated_prompt_tokens": prompt_len // 4,
                    "estimate_method": "chars_div_4",
                },
            )

    def _log_result(self, trace: TraceLogger | None, data: dict, elapsed_ms: float) -> None:
        if trace:
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            trace.log(
                action="llm.result",
                duration_ms=elapsed_ms,
                meta={
                    "provider_label": self.provider_label,
                    "model": self.model,
                    "output_chars": len(content),
                    "estimated_prompt_tokens": prompt_tokens,
                    "estimated_completion_tokens": completion_tokens,
                    "tokens": total_tokens,
                    "latency_ms": round(elapsed_ms, 1),
                    "estimate_method": "token_count_from_api" if total_tokens else "none",
                },
            )

    def _log_error(self, trace: TraceLogger | None, error: str, status_code: int = 0) -> None:
        if trace:
            trace.log(
                action="llm.error",
                error=error[:500],
                meta={
                    "provider_label": self.provider_label,
                    "model": self.model,
                    "status_code": status_code,
                    "exception_type": type(error).__name__ if not isinstance(error, str) else "str",
                },
            )

    def _post_with_retry(self, payload: dict, trace: TraceLogger | None) -> dict:
        import httpx

        endpoint = self._endpoint()
        headers = self._build_headers()
        max_retries = 3
        last_exc: Exception | None = None

        for attempt in range(max_retries):
            try:
                t0 = time.monotonic()
                resp = httpx.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=90,
                )
                elapsed = (time.monotonic() - t0) * 1000

                if resp.status_code in _RETRYABLE_STATUS and attempt < max_retries - 1:
                    wait = (2 ** attempt) * 1.0
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                tokens = data.get("usage", {}).get("total_tokens", 0)
                self._record_call(tokens, elapsed)
                self._log_result(trace, data, elapsed)
                return data

            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    status = getattr(getattr(exc, "response", None), "status_code", 0)
                    if status in _RETRYABLE_STATUS or status == 0:
                        wait = (2 ** attempt) * 1.0
                        time.sleep(wait)
                        continue
                self._log_error(trace, str(exc), getattr(getattr(exc, "response", None), "status_code", 0))
                raise

        raise last_exc or RuntimeError("Unexpected retry exhaustion")

    def complete_json(
        self,
        schema: type[BaseModel],
        prompt: str,
        *,
        context: str = "",
        trace: TraceLogger | None = None,
    ) -> BaseModel:
        system = "You are a research assistant. Return valid JSON matching the requested schema."
        user_msg = prompt
        if context:
            user_msg = f"{prompt}\n\nContext:\n{context}"
        schema_hint = json.dumps(schema.model_json_schema(), indent=2)
        user_msg += f"\n\nJSON Schema:\n{schema_hint}"

        self._log_call(trace, len(user_msg))

        for attempt in range(3):
            sys_msg = system
            if attempt > 0:
                sys_msg = "You MUST output ONLY valid JSON. No prose, no markdown fences."
            try:
                data = self._post_with_retry(
                    {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.3,
                    },
                    trace if attempt == 0 else None,
                )
                content = data["choices"][0]["message"]["content"]
                return schema.model_validate(json.loads(content))
            except (json.JSONDecodeError, KeyError, Exception) as exc:
                if attempt == 2:
                    self._log_error(trace, f"JSON parse failed after 3 attempts: {exc}")
                    raise
        raise RuntimeError("Failed to parse JSON after 3 attempts")

    def complete_text(
        self,
        prompt: str,
        *,
        context: str = "",
        trace: TraceLogger | None = None,
    ) -> str:
        user_msg = prompt
        if context:
            user_msg = f"{prompt}\n\nContext:\n{context}"

        self._log_call(trace, len(user_msg))

        data = self._post_with_retry(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": user_msg}],
                "temperature": 0.5,
            },
            trace=trace,
        )
        return data["choices"][0]["message"]["content"]

    def get_stats(self) -> dict[str, Any]:
        stats = super().get_stats()
        stats["provider_label"] = self.provider_label
        return stats
