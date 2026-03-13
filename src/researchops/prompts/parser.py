"""Robust JSON response parser with multiple fallback strategies."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def parse_json_response(raw: str, schema: type[T] | None = None) -> T | dict[str, Any]:
    """Parse an LLM response into a Pydantic model or dict.

    Strategies tried in order:
    1. Direct JSON parse
    2. Strip markdown fences then parse
    3. Extract first ``{...}`` block (greedy, handling nested braces)
    4. Regex extraction for common patterns
    """
    text = raw.strip()

    for _attempt, cleaned in enumerate(_cleaning_passes(text)):
        try:
            data = json.loads(cleaned)
            if schema is not None:
                return schema.model_validate(data)
            return data
        except (json.JSONDecodeError, Exception):
            continue

    logger.warning("All JSON parse strategies failed for input (first 200 chars): %s", text[:200])
    if schema is not None:
        raise ValueError(f"Could not parse LLM response as {schema.__name__}: {text[:300]}")
    return {}


def parse_json_safe(raw: str) -> dict[str, Any]:
    """Best-effort parse returning empty dict on failure."""
    try:
        return parse_json_response(raw, schema=None)
    except Exception:
        return {}


def _cleaning_passes(text: str):
    """Yield progressively cleaned versions of the input."""
    yield text

    stripped = _strip_markdown_fences(text)
    if stripped != text:
        yield stripped

    block = _extract_json_block(text)
    if block:
        yield block

    block_from_stripped = _extract_json_block(stripped)
    if block_from_stripped and block_from_stripped != block:
        yield block_from_stripped


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrapping."""
    pattern = r"```(?:json)?\s*\n?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _extract_json_block(text: str) -> str | None:
    """Extract the first balanced ``{...}`` block."""
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape_next = False
    end = start

    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if depth == 0:
        return text[start : end + 1]
    return text[start:]
