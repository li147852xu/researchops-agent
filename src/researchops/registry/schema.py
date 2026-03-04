from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    name: str
    version: str = "0.1.0"
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    risk_level: Literal["low", "medium", "high"] = "low"
    permissions: list[str] = Field(default_factory=list)
    timeout_default: int = 30
    cache_policy: Literal["none", "session", "persistent"] = "none"
