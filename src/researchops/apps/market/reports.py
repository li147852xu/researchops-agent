"""Market Intelligence report formatting helpers."""

from __future__ import annotations

import re


def format_section_heading(heading: str, level: int = 2) -> str:
    prefix = "#" * level
    return f"{prefix} {heading}"


def format_citation(source_id: str) -> str:
    return f"[@{source_id}]"


def strip_citation_markers(text: str) -> str:
    return re.sub(r"\[@\w+\]", "", text).strip()


def format_metric_line(label: str, value: str, source_id: str = "") -> str:
    cite = f" [@{source_id}]" if source_id else ""
    return f"- **{label}**: {value}{cite}"
