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
            claims = self._extract_claims(src.source_id, text, rq_ids)
            total_claims += len(claims)

            notes = SourceNotes(source_id=src.source_id, claims=claims)
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

    def _extract_claims(
        self, source_id: str, text: str, rq_ids: list[str]
    ) -> list[Claim]:
        if not text.strip():
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text)
        meaningful = [s.strip() for s in sentences if len(s.strip()) > 30]

        claims: list[Claim] = []
        for i, sentence in enumerate(meaningful[:10]):
            supports = []
            lower = sentence.lower()
            if any(kw in lower for kw in ("current", "state", "research", "study")):
                supports = [r for r in rq_ids if "state" in r or "overview" in r]
            elif any(kw in lower for kw in ("challenge", "limit", "problem", "issue")):
                supports = [r for r in rq_ids if "challenge" in r]
            elif any(kw in lower for kw in ("future", "direction", "trend", "emerging")):
                supports = [r for r in rq_ids if "future" in r]
            else:
                supports = rq_ids[:1]

            claims.append(
                Claim(
                    claim_id=f"{source_id}_c{i}",
                    text=sentence[:300],
                    evidence_spans=[sentence[:150]],
                    supports_rq=supports,
                )
            )

        return claims
