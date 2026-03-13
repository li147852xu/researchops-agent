"""Shared utility functions — deduplicated from reader, writer, qa, evaluator."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from researchops.models import PlanOutput, Source, SourceNotes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loaders (previously duplicated in 4+ files)
# ---------------------------------------------------------------------------

def load_sources(run_dir: Path) -> list[Source]:
    path = run_dir / "sources.jsonl"
    if not path.exists():
        return []
    sources: list[Source] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            try:
                sources.append(Source.model_validate(json.loads(line)))
            except Exception as exc:
                logger.warning("Skipping invalid source line: %s", exc)
    return sources


def load_all_notes(run_dir: Path) -> dict[str, SourceNotes]:
    notes_dir = run_dir / "notes"
    result: dict[str, SourceNotes] = {}
    if not notes_dir.exists():
        return result
    for f in sorted(notes_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            notes = SourceNotes.model_validate(data)
            result[notes.source_id] = notes
        except Exception as exc:
            logger.warning("Skipping invalid notes file %s: %s", f.name, exc)
    return result


def load_plan(run_dir: Path) -> PlanOutput | None:
    plan_path = run_dir / "plan.json"
    if not plan_path.exists():
        return None
    try:
        return PlanOutput.model_validate(
            json.loads(plan_path.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        logger.warning("Failed to load plan: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Negative terms (previously duplicated in supervisor.py + collector.py)
# ---------------------------------------------------------------------------

_NEGATIVE_TERMS: dict[str, list[str]] = {
    "deep learning": ["collider", "particle physics", "astronomy", "geology", "marine biology"],
    "machine learning": ["collider", "particle physics", "astronomy", "deep sea"],
    "quantum computing": ["quantum healing", "quantum mysticism", "spirituality"],
    "natural language processing": ["chemical processing", "food processing", "manufacturing"],
    "computer vision": ["ophthalmology", "optometry", "eye surgery"],
    "reinforcement learning": ["positive reinforcement parenting", "dog training"],
}


def get_negative_terms(topic: str) -> list[str]:
    topic_lower = topic.lower()
    for key, terms in _NEGATIVE_TERMS.items():
        if key in topic_lower:
            return terms
    return []


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """Split *text* into overlapping windows of approximately *chunk_size* words."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def load_claim_dicts(run_dir: Path) -> list[dict]:
    """Load all claims from notes/ as flat dicts for retrieval indexing."""
    notes = load_all_notes(run_dir)
    claims: list[dict] = []
    for _sid, n in notes.items():
        for c in n.claims:
            claims.append({
                "claim_id": c.claim_id, "text": c.text,
                "source_id": n.source_id,
                "supports_rq": c.supports_rq,
                "claim_type": c.claim_type, "polarity": c.polarity,
            })
    return claims


def compute_coverage(run_dir: Path) -> tuple[dict[str, int], float]:
    """Return (rq_counts_dict, coverage_fraction)."""
    plan = load_plan(run_dir)
    if not plan:
        return {}, 1.0
    notes = load_all_notes(run_dir)
    rq_counts: dict[str, int] = {rq.rq_id: 0 for rq in plan.research_questions}
    for _sid, n in notes.items():
        for c in n.claims:
            for rq_id in c.supports_rq:
                if rq_id in rq_counts:
                    rq_counts[rq_id] += 1
    total = len(plan.research_questions)
    covered = sum(1 for v in rq_counts.values() if v > 0)
    return rq_counts, covered / total if total > 0 else 1.0


def truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."
