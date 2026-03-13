from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel

from researchops.reasoning.base import ReasonerBase

if TYPE_CHECKING:
    from researchops.trace import TraceLogger


class NoneReasoner(ReasonerBase):
    """Rule/template-based reasoner that requires no LLM API key."""

    def complete_json(
        self,
        schema: type[BaseModel],
        prompt: str,
        *,
        context: str = "",
        trace: TraceLogger | None = None,
    ) -> BaseModel:
        t0 = time.monotonic()
        fields = schema.model_fields
        data: dict = {}
        for name, field_info in fields.items():
            ann = field_info.annotation
            if ann is str or (hasattr(ann, "__origin__") and ann is str):
                data[name] = self._extract_snippet(prompt, name)
            elif ann is int:
                data[name] = 0
            elif ann is float:
                data[name] = 0.0
            elif ann is bool:
                data[name] = True
            elif ann is list or (hasattr(ann, "__origin__") and getattr(ann, "__origin__", None) is list):
                data[name] = []
            else:
                data[name] = field_info.default if field_info.default is not None else None
        result = schema.model_validate(data)
        self._record_call(0, (time.monotonic() - t0) * 1000)
        return result

    def complete_text(
        self,
        prompt: str,
        *,
        context: str = "",
        trace: TraceLogger | None = None,
    ) -> str:
        t0 = time.monotonic()
        result = self._template_response(prompt, context)
        self._record_call(0, (time.monotonic() - t0) * 1000)
        return result

    def _extract_snippet(self, prompt: str, field_name: str) -> str:
        lines = prompt.strip().splitlines()
        for line in lines:
            if field_name.lower() in line.lower():
                return line.strip()[:200]
        return lines[0][:200] if lines else ""

    def _template_response(self, prompt: str, context: str) -> str:
        lower = prompt.lower()
        if "fix" in lower or "error" in lower or "repair" in lower:
            return self._suggest_fix(prompt, context)
        if "summarize" in lower or "write" in lower or "paragraph" in lower:
            return context[:500] if context else prompt[:500]
        return prompt[:300]

    def _suggest_fix(self, prompt: str, context: str) -> str:
        fixes: list[str] = []
        combined = prompt + "\n" + context
        if "ImportError" in combined or "ModuleNotFoundError" in combined:
            mod = re.search(r"No module named '(\w+)'", combined)
            if mod:
                fixes.append(f"Remove or replace import of '{mod.group(1)}'")
            fixes.append("Wrap import in try/except with fallback")
        if "FileNotFoundError" in combined:
            fixes.append("Check path exists before opening; use Path.exists() guard")
        if "JSONDecodeError" in combined:
            fixes.append("Wrap json.loads in try/except; skip malformed entries")
        if "ZeroDivisionError" in combined:
            fixes.append("Add guard: if denominator == 0, use default value")
        if "UnicodeDecodeError" in combined:
            fixes.append("Add errors='replace' to read_text/open calls")
        if not fixes:
            fixes.append("Add broad try/except around failing section")
        return "; ".join(fixes)
