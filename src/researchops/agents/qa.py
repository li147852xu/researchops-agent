from __future__ import annotations

import json
import re
from pathlib import Path

from researchops.agents.base import AgentBase, RunContext
from researchops.models import AgentResult, Source, SourceNotes, Stage


class QAAgent(AgentBase):
    name = "qa"

    def execute(self, ctx: RunContext) -> AgentResult:
        ctx.trace.log(stage="QA", agent=self.name, action="start")

        report_path = ctx.run_dir / "report.md"
        if not report_path.exists():
            return AgentResult(
                success=False,
                message="No report found",
                rollback_to=Stage.WRITE,
            )

        report = report_path.read_text(encoding="utf-8")
        sources = self._load_sources(ctx.run_dir)
        all_notes = self._load_all_notes(ctx.run_dir)

        issues: list[str] = []

        cov = self._check_citation_coverage(report)
        if cov < 0.3:
            issues.append(f"Low citation coverage: {cov:.0%}")

        diversity = self._check_source_diversity(sources)
        if diversity["unique_domains"] < 1:
            issues.append("No source diversity")

        traceability = self._check_traceability(ctx.run_dir, all_notes)
        unsupported_rate = traceability.get("unsupported_rate", 0.0)
        if unsupported_rate > 0.5:
            issues.append(f"High unsupported claim rate: {unsupported_rate:.0%}")

        unsupported_assertions = self._scan_unsupported_assertions(report, sources)
        if unsupported_assertions:
            issues.append(f"{len(unsupported_assertions)} assertions without citation markers")

        rollback_target = self._determine_rollback(issues, traceability)

        ctx.trace.log(
            stage="QA",
            agent=self.name,
            action="complete",
            output_summary=f"{len(issues)} issues found" if issues else "QA passed",
            meta={
                "citation_coverage": cov,
                "unsupported_rate": unsupported_rate,
                "issues": issues,
                "diversity": diversity,
            },
        )

        if issues and ctx.state.retry_counts.get("qa", 0) < 1:
            ctx.state.retry_counts["qa"] = ctx.state.retry_counts.get("qa", 0) + 1
            return AgentResult(
                success=False,
                message=f"QA issues: {'; '.join(issues)}",
                rollback_to=rollback_target,
            )

        return AgentResult(
            success=True,
            message="QA passed" if not issues else f"QA accepted with warnings: {'; '.join(issues)}",
            data={
                "issues": issues,
                "citation_coverage": cov,
                "unsupported_rate": unsupported_rate,
                "diversity": diversity,
            },
        )

    def _check_citation_coverage(self, report: str) -> float:
        paragraphs = [p.strip() for p in report.split("\n\n") if len(p.strip()) > 50]
        if not paragraphs:
            return 0.0
        cited = sum(1 for p in paragraphs if re.search(r"\[@\w+\]", p))
        return cited / len(paragraphs) if paragraphs else 0.0

    def _check_source_diversity(self, sources: list[Source]) -> dict:
        domains = {s.domain for s in sources if s.domain}
        type_dist: dict[str, int] = {}
        for s in sources:
            type_dist[s.type.value] = type_dist.get(s.type.value, 0) + 1
        return {
            "unique_domains": len(domains),
            "domains": sorted(domains),
            "type_distribution": type_dist,
        }

    def _check_traceability(
        self, run_dir: Path, all_notes: dict[str, SourceNotes]
    ) -> dict:
        index_path = run_dir / "report_index.json"
        if not index_path.exists():
            return {"unsupported_rate": 0.0, "total": 0, "unsupported": 0}
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return {"unsupported_rate": 0.0, "total": 0, "unsupported": 0}

        entries = index.get("entries", [])
        if not entries:
            return {"unsupported_rate": 0.0, "total": 0, "unsupported": 0}

        known_claim_ids: set[str] = set()
        known_source_ids: set[str] = set()
        for sid, notes in all_notes.items():
            known_source_ids.add(sid)
            for c in notes.claims:
                known_claim_ids.add(c.claim_id)

        unsupported = 0
        for entry in entries:
            has_valid_ref = False
            for cid in entry.get("claim_ids", []):
                if cid in known_claim_ids:
                    has_valid_ref = True
                    break
            if not has_valid_ref:
                for sid in entry.get("source_ids", []):
                    if sid in known_source_ids:
                        has_valid_ref = True
                        break
            if not has_valid_ref:
                unsupported += 1

        rate = unsupported / len(entries) if entries else 0.0
        return {"unsupported_rate": round(rate, 3), "total": len(entries), "unsupported": unsupported}

    def _scan_unsupported_assertions(self, report: str, sources: list[Source]) -> list[str]:
        unsupported = []
        for line in report.split("\n"):
            stripped = line.strip()
            if (
                len(stripped) > 60
                and not stripped.startswith("#")
                and not stripped.startswith("-")
                and not stripped.startswith("*")
                and not re.search(r"\[@\w+\]", stripped)
            ):
                unsupported.append(stripped[:80])
        return unsupported

    def _determine_rollback(self, issues: list[str], traceability: dict) -> Stage:
        issue_text = " ".join(issues).lower()
        if "unsupported claim rate" in issue_text:
            return Stage.READ
        if "no source diversity" in issue_text:
            return Stage.COLLECT
        return Stage.WRITE

    def _load_sources(self, run_dir: Path) -> list[Source]:
        path = run_dir / "sources.jsonl"
        if not path.exists():
            return []
        sources = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                sources.append(Source.model_validate(json.loads(line)))
        return sources

    def _load_all_notes(self, run_dir: Path) -> dict[str, SourceNotes]:
        notes_dir = run_dir / "notes"
        result: dict[str, SourceNotes] = {}
        if not notes_dir.exists():
            return result
        for f in sorted(notes_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                notes = SourceNotes.model_validate(data)
                result[notes.source_id] = notes
            except Exception:
                continue
        return result
