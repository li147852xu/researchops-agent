"""Report formatting helpers for the Research app."""

from __future__ import annotations

import re


def format_section_heading(heading: str, level: int = 2) -> str:
    prefix = "#" * level
    return f"{prefix} {heading}"


def format_citation(source_id: str) -> str:
    return f"[@{source_id}]"


def strip_citation_markers(text: str) -> str:
    return re.sub(r"\[@\w+\]", "", text).strip()
