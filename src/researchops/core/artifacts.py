"""Evidence protocol — first-class types for source/claim/support tracking.

These core types ensure that important output claims link back to evidence,
support relationships are explicit, unsupported assertions are identifiable,
and evaluators can inspect support quality.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AssertionStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    INFERRED = "inferred"
    UNSUPPORTED = "unsupported"


class SourceRecord(BaseModel):
    """Core representation of any ingested source."""

    source_id: str
    origin: str = ""
    url: str = ""
    title: str = ""
    content_hash: str = ""
    retrieved_at: str = ""


class EvidenceChunk(BaseModel):
    """A passage extracted from a source that may support claims."""

    chunk_id: str
    source_id: str
    text: str
    location: str = ""


class ExtractedClaim(BaseModel):
    """A factual assertion extracted from evidence."""

    claim_id: str
    text: str
    evidence_chunks: list[str] = Field(default_factory=list)
    source_id: str = ""
    claim_type: str = ""
    polarity: str = "neutral"


class ClaimSupportLink(BaseModel):
    """Explicit link between a claim and a target (RQ, section, etc.)."""

    claim_id: str
    target_id: str
    strength: AssertionStatus = AssertionStatus.INFERRED


class ReportSectionRef(BaseModel):
    """Tracks evidence backing for a report section."""

    section_id: str
    heading: str
    claim_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    status: AssertionStatus = AssertionStatus.UNSUPPORTED
