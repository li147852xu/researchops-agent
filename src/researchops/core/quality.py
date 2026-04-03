"""Quality assessment — evidence density, citation coverage, claim confidence, conflict detection.

Provides fine-grained quality signals that the supervisor and evaluation
harness use to decide rollback, and that the final eval report exposes
as improvement metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ── Evidence Density ───────────────────────────────────────────────────

@dataclass
class SectionDensity:
    """Evidence density metrics for one report section."""

    heading: str = ""
    paragraph_count: int = 0
    cited_paragraphs: int = 0
    total_citations: int = 0
    density_score: float = 0.0


def compute_evidence_density(report_text: str) -> list[SectionDensity]:
    """Score each section by how densely it cites sources.

    Returns a list of per-section density objects.  Density is defined as
    ``cited_paragraphs / paragraph_count`` weighted by citation count.
    """
    sections = re.split(r"^(##\s+.+)$", report_text, flags=re.MULTILINE)
    results: list[SectionDensity] = []

    i = 1
    while i < len(sections):
        heading = sections[i].strip().lstrip("#").strip()
        body = sections[i + 1] if i + 1 < len(sections) else ""
        i += 2

        paragraphs = [p.strip() for p in body.split("\n\n") if len(p.strip()) > 30]
        if not paragraphs:
            results.append(SectionDensity(heading=heading))
            continue

        cited = 0
        total_cites = 0
        for para in paragraphs:
            cites = len(re.findall(r"\[@\w+\]", para))
            total_cites += cites
            if cites > 0:
                cited += 1

        density = cited / len(paragraphs) if paragraphs else 0.0
        results.append(SectionDensity(
            heading=heading,
            paragraph_count=len(paragraphs),
            cited_paragraphs=cited,
            total_citations=total_cites,
            density_score=round(density, 3),
        ))

    return results


def overall_evidence_density(report_text: str) -> float:
    """Single aggregate density score across all sections."""
    sections = compute_evidence_density(report_text)
    if not sections:
        return 0.0
    total_paras = sum(s.paragraph_count for s in sections)
    total_cited = sum(s.cited_paragraphs for s in sections)
    return round(total_cited / max(1, total_paras), 3)


# ── Citation Coverage Check ────────────────────────────────────────────

@dataclass
class CoverageGap:
    """A paragraph or section that lacks adequate citation."""

    section: str = ""
    paragraph_index: int = 0
    text_preview: str = ""
    citation_count: int = 0


def find_citation_gaps(
    report_text: str,
    min_cites_per_paragraph: int = 1,
) -> list[CoverageGap]:
    """Identify paragraphs that fall below the citation threshold.

    Useful for the QA agent to flag under-cited sections before finalizing
    the report.
    """
    sections = re.split(r"^(##\s+.+)$", report_text, flags=re.MULTILINE)
    gaps: list[CoverageGap] = []

    current_heading = "Introduction"
    i = 0
    while i < len(sections):
        part = sections[i]
        if part.startswith("##"):
            current_heading = part.strip().lstrip("#").strip()
            i += 1
            continue

        paragraphs = [p.strip() for p in part.split("\n\n") if len(p.strip()) > 50]
        for pidx, para in enumerate(paragraphs):
            if para.startswith("-") or para.startswith("*"):
                continue
            cite_count = len(re.findall(r"\[@\w+\]", para))
            if cite_count < min_cites_per_paragraph:
                gaps.append(CoverageGap(
                    section=current_heading,
                    paragraph_index=pidx,
                    text_preview=para[:120],
                    citation_count=cite_count,
                ))
        i += 1

    return gaps


# ── Claim Confidence Scoring ──────────────────────────────────────────

@dataclass
class ClaimConfidence:
    """Confidence assessment for a single extracted claim."""

    claim_id: str = ""
    text: str = ""
    confidence: float = 0.5
    factors: dict[str, float] = field(default_factory=dict)


def score_claim_confidence(claims: list[dict[str, Any]]) -> list[ClaimConfidence]:
    """Assign a confidence score to each claim based on structural signals.

    Higher confidence for:
    - Numerical claims (concrete, verifiable)
    - Claims with evidence spans
    - Claims supported by multiple RQs (cross-corroborated)
    - Claims of type "result" or "metric" (empirical)
    """
    results: list[ClaimConfidence] = []
    for claim in claims:
        text = claim.get("text", "")
        factors: dict[str, float] = {}

        has_number = bool(re.search(r"\d+\.?\d*\s*[%$€£¥BMKbmk]|\$\d|\d+\.\d+", text))
        factors["numerical"] = 0.2 if has_number else 0.0

        has_span = bool(claim.get("evidence_spans") or claim.get("evidence_span"))
        factors["evidence_span"] = 0.2 if has_span else 0.0

        rq_count = len(claim.get("supports_rq", []))
        factors["rq_coverage"] = min(0.2, rq_count * 0.1)

        ctype = claim.get("claim_type", "").lower()
        empirical_types = {"result", "metric", "comparison", "method"}
        factors["claim_type"] = 0.15 if ctype in empirical_types else 0.05

        factors["base"] = 0.3

        confidence = sum(factors.values())
        confidence = max(0.1, min(1.0, confidence))

        results.append(ClaimConfidence(
            claim_id=claim.get("claim_id", ""),
            text=text[:200],
            confidence=round(confidence, 3),
            factors=factors,
        ))

    return results


# ── Conflict Detection ─────────────────────────────────────────────────

@dataclass
class ClaimConflict:
    """A detected contradiction between two claims."""

    claim_a_id: str = ""
    claim_b_id: str = ""
    claim_a_text: str = ""
    claim_b_text: str = ""
    conflict_type: str = ""
    severity: float = 0.5


def detect_conflicts(claims: list[dict[str, Any]]) -> list[ClaimConflict]:
    """Detect potential contradictions between extracted claims.

    Uses heuristic signals:
    - Opposing polarity on the same RQ
    - Conflicting numerical values for the same metric
    - Direct negation patterns
    """
    conflicts: list[ClaimConflict] = []

    rq_groups: dict[str, list[dict[str, Any]]] = {}
    for c in claims:
        for rq in c.get("supports_rq", []):
            rq_groups.setdefault(rq, []).append(c)

    for _rq, group in rq_groups.items():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                conflict = _check_pair(a, b)
                if conflict:
                    conflicts.append(conflict)

    return conflicts


_NEGATION_PAIRS = [
    ("increase", "decrease"),
    ("growth", "decline"),
    ("bullish", "bearish"),
    ("support", "oppose"),
    ("positive", "negative"),
    ("improve", "worsen"),
    ("rise", "fall"),
    ("gain", "loss"),
]


def _check_pair(a: dict[str, Any], b: dict[str, Any]) -> ClaimConflict | None:
    pol_a = a.get("polarity", "neutral").lower()
    pol_b = b.get("polarity", "neutral").lower()
    text_a = a.get("text", "").lower()
    text_b = b.get("text", "").lower()

    if pol_a != "neutral" and pol_b != "neutral" and pol_a != pol_b:
        return ClaimConflict(
            claim_a_id=a.get("claim_id", ""),
            claim_b_id=b.get("claim_id", ""),
            claim_a_text=a.get("text", "")[:150],
            claim_b_text=b.get("text", "")[:150],
            conflict_type="polarity_opposition",
            severity=0.7,
        )

    for pos, neg in _NEGATION_PAIRS:
        if (pos in text_a and neg in text_b) or (neg in text_a and pos in text_b):
            return ClaimConflict(
                claim_a_id=a.get("claim_id", ""),
                claim_b_id=b.get("claim_id", ""),
                claim_a_text=a.get("text", "")[:150],
                claim_b_text=b.get("text", "")[:150],
                conflict_type="semantic_negation",
                severity=0.5,
            )

    numbers_a = re.findall(r"(\d+\.?\d*)\s*%", text_a)
    numbers_b = re.findall(r"(\d+\.?\d*)\s*%", text_b)
    if numbers_a and numbers_b:
        try:
            val_a = float(numbers_a[0])
            val_b = float(numbers_b[0])
            if abs(val_a - val_b) > max(val_a, val_b) * 0.5 and min(val_a, val_b) > 0:
                return ClaimConflict(
                    claim_a_id=a.get("claim_id", ""),
                    claim_b_id=b.get("claim_id", ""),
                    claim_a_text=a.get("text", "")[:150],
                    claim_b_text=b.get("text", "")[:150],
                    conflict_type="numerical_discrepancy",
                    severity=0.6,
                )
        except ValueError:
            pass

    return None
