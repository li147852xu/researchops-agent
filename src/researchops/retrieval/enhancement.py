"""Retrieval enhancement — query reformulation, source quality scoring, relevance calibration.

These algorithms improve recall and precision of the hybrid retrieval layer
without changing the underlying BM25/embedding indexes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ── Query Reformulation ────────────────────────────────────────────────

_SYNONYM_MAP: dict[str, list[str]] = {
    "machine learning": ["ML", "statistical learning", "predictive modeling"],
    "deep learning": ["neural networks", "DL", "deep neural networks"],
    "natural language processing": ["NLP", "text mining", "computational linguistics"],
    "reinforcement learning": ["RL", "reward-based learning", "policy optimization"],
    "computer vision": ["image recognition", "visual computing", "CV"],
    "transformer": ["attention mechanism", "self-attention", "encoder-decoder"],
    "large language model": ["LLM", "foundation model", "generative AI"],
    "retrieval augmented generation": ["RAG", "retrieval-grounded generation"],
    "artificial intelligence": ["AI", "intelligent systems"],
    "financial analysis": ["equity research", "fundamental analysis", "market analysis"],
    "risk assessment": ["risk evaluation", "risk profiling", "risk management"],
    "competitive analysis": ["market positioning", "competitive landscape"],
}


def expand_query(query: str, max_variants: int = 3) -> list[str]:
    """Generate query variants via synonym expansion.

    Returns the original query plus up to *max_variants* reformulations
    using domain-specific synonym mappings.
    """
    variants: list[str] = [query]
    lower = query.lower()

    for term, synonyms in _SYNONYM_MAP.items():
        if term in lower:
            for syn in synonyms[:max_variants]:
                variant = re.sub(re.escape(term), syn, lower, flags=re.IGNORECASE)
                if variant not in variants:
                    variants.append(variant)
                if len(variants) > max_variants:
                    break
        if len(variants) > max_variants:
            break

    return variants[: max_variants + 1]


def decompose_query(query: str) -> list[str]:
    """Split a compound query into atomic sub-queries.

    Handles common conjunctions and comma-separated clauses so each
    retrieval pass targets a focused aspect.
    """
    separators = [" and ", " vs ", " versus ", " compared to ", "; "]
    parts: list[str] = [query]
    for sep in separators:
        new_parts: list[str] = []
        for p in parts:
            new_parts.extend(p.split(sep))
        parts = new_parts

    return [p.strip() for p in parts if len(p.strip()) > 10]


# ── Source Quality Scoring ─────────────────────────────────────────────

_HIGH_AUTHORITY_DOMAINS = frozenset({
    "arxiv.org", "nature.com", "science.org", "ieee.org", "acm.org",
    "springer.com", "wiley.com", "ncbi.nlm.nih.gov", "scholar.google.com",
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "sec.gov",
    "federalreserve.gov", "imf.org", "worldbank.org",
})

_LOW_QUALITY_SIGNALS = frozenset({
    "reddit.com", "quora.com", "yahoo.com/answers", "medium.com",
    "blogspot.com", "wordpress.com",
})


@dataclass
class SourceQualityScore:
    """Composite quality score for a collected source."""

    domain_authority: float = 0.5
    freshness: float = 0.5
    content_richness: float = 0.5
    composite: float = 0.5


def score_source_quality(
    domain: str,
    published_date: str = "",
    content_length: int = 0,
    has_numerical_data: bool = False,
) -> SourceQualityScore:
    """Compute a multi-factor quality score for a source document."""
    authority = _domain_authority(domain)
    freshness = _freshness_score(published_date)
    richness = _content_richness(content_length, has_numerical_data)

    composite = authority * 0.4 + freshness * 0.3 + richness * 0.3
    return SourceQualityScore(
        domain_authority=round(authority, 3),
        freshness=round(freshness, 3),
        content_richness=round(richness, 3),
        composite=round(composite, 3),
    )


def _domain_authority(domain: str) -> float:
    domain_lower = domain.lower().strip()
    for hq in _HIGH_AUTHORITY_DOMAINS:
        if hq in domain_lower:
            return 0.9
    for lq in _LOW_QUALITY_SIGNALS:
        if lq in domain_lower:
            return 0.2
    if domain_lower.endswith(".edu") or domain_lower.endswith(".gov"):
        return 0.85
    return 0.5


def _freshness_score(published_date: str) -> float:
    if not published_date:
        return 0.3
    try:
        pub = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=UTC)
        age_days = (datetime.now(UTC) - pub).days
        return max(0.1, min(1.0, 1.0 - age_days / 1825))  # 5-year decay
    except (ValueError, TypeError):
        return 0.3


def _content_richness(content_length: int, has_numerical: bool) -> float:
    length_score = min(1.0, content_length / 5000) if content_length > 0 else 0.2
    numerical_bonus = 0.15 if has_numerical else 0.0
    return min(1.0, length_score + numerical_bonus)


# ── Relevance Calibration ──────────────────────────────────────────────

@dataclass
class CalibratedResult:
    """A retrieval result with calibrated relevance score."""

    claim_id: str = ""
    text: str = ""
    raw_score: float = 0.0
    calibrated_score: float = 0.0
    source_quality: float = 0.5
    original: dict[str, Any] = field(default_factory=dict)


def calibrate_relevance(
    results: list[dict[str, Any]],
    source_qualities: dict[str, float] | None = None,
    boost_numerical: bool = False,
) -> list[CalibratedResult]:
    """Re-score retrieval results by combining retrieval rank with source quality.

    Applies min-max normalization on raw retrieval scores, then blends with
    source-level quality signals for a composite ranking.
    """
    if not results:
        return []

    source_qualities = source_qualities or {}
    calibrated: list[CalibratedResult] = []

    max_rank = len(results)
    for rank, item in enumerate(results):
        raw = 1.0 - (rank / max(1, max_rank))
        cid = item.get("claim_id", "")
        sid = item.get("source_id", "")
        text = item.get("text", "")

        sq = source_qualities.get(sid, 0.5)

        bonus = 0.0
        if boost_numerical and re.search(r"\d+\.?\d*\s*[%$€£¥BMK]|\$\d", text):
            bonus = 0.05

        calibrated_score = raw * 0.6 + sq * 0.3 + bonus + 0.1
        calibrated_score = max(0.0, min(1.0, calibrated_score))

        calibrated.append(CalibratedResult(
            claim_id=cid,
            text=text,
            raw_score=round(raw, 3),
            calibrated_score=round(calibrated_score, 3),
            source_quality=round(sq, 3),
            original=item,
        ))

    calibrated.sort(key=lambda c: c.calibrated_score, reverse=True)
    return calibrated
