from __future__ import annotations

import json
import re
from pathlib import Path

from researchops.agents.base import AgentBase, RunContext
from researchops.models import (
    AgentResult,
    Claim,
    PlanOutput,
    Source,
    SourceNotes,
)

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "contribution": ["propose", "introduce", "present", "novel", "contribution", "develop", "design"],
    "method": ["method", "approach", "technique", "algorithm", "framework", "architecture", "implement"],
    "limitation": ["challenge", "limit", "problem", "issue", "drawback", "barrier", "difficult", "gap"],
    "finding": ["result", "show", "demonstrate", "find", "observe", "evidence", "significant", "suggest"],
}


class ReaderAgent(AgentBase):
    name = "reader"

    def execute(self, ctx: RunContext) -> AgentResult:
        ctx.trace.log(stage="READ", agent=self.name, action="start")

        sources = self._load_sources(ctx.run_dir)
        plan = PlanOutput.model_validate(
            json.loads((ctx.run_dir / "plan.json").read_text(encoding="utf-8"))
        )
        rq_ids = [rq.rq_id for rq in plan.research_questions]

        notes_dir = ctx.run_dir / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        total_claims = 0

        for src in sources:
            text = self._extract_text(src, ctx)
            paragraphs = self._split_paragraphs(text)
            claims = self._extract_claims(src.source_id, paragraphs, rq_ids)
            total_claims += len(claims)

            contribution = self._extract_section(text, "contribution")
            method = self._extract_section(text, "method")
            limitations = self._extract_section(text, "limitation")

            notes = SourceNotes(
                source_id=src.source_id,
                claims=claims,
                contribution=contribution,
                method=method,
                limitations=limitations,
            )
            out_path = notes_dir / f"{src.source_id}.json"
            out_path.write_text(notes.model_dump_json(indent=2), encoding="utf-8")

        ctx.trace.log(
            stage="READ",
            agent=self.name,
            action="complete",
            output_summary=f"Extracted {total_claims} claims from {len(sources)} sources",
        )
        return AgentResult(
            success=True,
            message=f"Extracted {total_claims} claims from {len(sources)} sources",
        )

    def _load_sources(self, run_dir: Path) -> list[Source]:
        sources_path = run_dir / "sources.jsonl"
        if not sources_path.exists():
            return []
        sources = []
        for line in sources_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                sources.append(Source.model_validate(json.loads(line)))
        return sources

    def _extract_text(self, src: Source, ctx: RunContext) -> str:
        if not src.local_path or not Path(src.local_path).exists():
            return ""
        try:
            result = ctx.registry.invoke(
                "parse",
                {"file_path": src.local_path, "format": src.type.value},
                trace=ctx.trace,
            )
            return result.get("text", "")
        except Exception:
            return Path(src.local_path).read_text(encoding="utf-8", errors="replace")

    def _split_paragraphs(self, text: str) -> list[tuple[int, str]]:
        paragraphs = []
        for i, para in enumerate(re.split(r"\n\s*\n", text)):
            stripped = para.strip()
            if len(stripped) > 30:
                paragraphs.append((i, stripped))
        return paragraphs

    def _extract_claims(
        self, source_id: str, paragraphs: list[tuple[int, str]], rq_ids: list[str]
    ) -> list[Claim]:
        claims: list[Claim] = []
        for para_idx, para in paragraphs[:15]:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for _si, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) < 30:
                    continue

                category = self._classify_sentence(sentence)
                supports = self._match_rqs(sentence, rq_ids)
                evidence_loc = f"paragraph_{para_idx}"

                claims.append(
                    Claim(
                        claim_id=f"{source_id}_c{len(claims)}",
                        text=sentence[:300],
                        evidence_spans=[sentence[:150]],
                        supports_rq=supports,
                        category=category,
                        evidence_location=evidence_loc,
                    )
                )
                if len(claims) >= 15:
                    return claims
        return claims

    def _classify_sentence(self, sentence: str) -> str:
        lower = sentence.lower()
        scores: dict[str, int] = {}
        for cat, keywords in _CATEGORY_KEYWORDS.items():
            scores[cat] = sum(1 for kw in keywords if kw in lower)
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "other"

    def _match_rqs(self, sentence: str, rq_ids: list[str]) -> list[str]:
        lower = sentence.lower()
        matched = []
        for rq_id in rq_ids:
            tag = rq_id.replace("rq_", "")
            if tag in lower:
                matched.append(rq_id)
        if not matched:
            if any(kw in lower for kw in ("current", "state", "research", "study")):
                matched = [r for r in rq_ids if "state" in r or "overview" in r]
            elif any(kw in lower for kw in ("challenge", "limit", "problem")):
                matched = [r for r in rq_ids if "challenge" in r]
            elif any(kw in lower for kw in ("future", "direction", "trend", "emerging")):
                matched = [r for r in rq_ids if "future" in r]
        return matched or rq_ids[:1]

    def _extract_section(self, text: str, section_type: str) -> str:
        keywords = _CATEGORY_KEYWORDS.get(section_type, [])
        sentences = re.split(r"(?<=[.!?])\s+", text)
        relevant = []
        for s in sentences:
            if any(kw in s.lower() for kw in keywords) and len(s.strip()) > 30:
                relevant.append(s.strip())
        return " ".join(relevant[:3])
