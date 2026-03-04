from __future__ import annotations

import json
import re
from pathlib import Path

from researchops.agents.base import AgentBase, RunContext
from researchops.models import AgentResult, Source, Stage


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

        issues: list[str] = []

        cov = self._check_citation_coverage(report)
        if cov < 0.3:
            issues.append(f"Low citation coverage: {cov:.0%}")

        unsupported = self._check_unsupported_conclusions(report)
        if unsupported:
            issues.append(f"{len(unsupported)} conclusion paragraphs without citations")

        diversity = self._check_source_diversity(sources)
        if diversity["unique_domains"] < 1:
            issues.append("No source diversity")

        ctx.trace.log(
            stage="QA",
            agent=self.name,
            action="complete",
            output_summary=f"{len(issues)} issues found" if issues else "QA passed",
            meta={
                "citation_coverage": cov,
                "issues": issues,
                "diversity": diversity,
            },
        )

        if issues and ctx.state.retry_counts.get("qa", 0) < 1:
            ctx.state.retry_counts["qa"] = ctx.state.retry_counts.get("qa", 0) + 1
            return AgentResult(
                success=False,
                message=f"QA issues: {'; '.join(issues)}",
                rollback_to=Stage.WRITE,
            )

        return AgentResult(
            success=True,
            message="QA passed" if not issues else f"QA accepted with warnings: {'; '.join(issues)}",
            data={"issues": issues, "citation_coverage": cov, "diversity": diversity},
        )

    def _check_citation_coverage(self, report: str) -> float:
        paragraphs = [p.strip() for p in report.split("\n\n") if len(p.strip()) > 50]
        if not paragraphs:
            return 0.0
        cited = sum(1 for p in paragraphs if re.search(r"\[\d+\]", p))
        return cited / len(paragraphs) if paragraphs else 0.0

    def _check_unsupported_conclusions(self, report: str) -> list[str]:
        unsupported = []
        in_conclusion = False
        for line in report.split("\n"):
            if line.strip().lower().startswith("## conclusion"):
                in_conclusion = True
                continue
            if in_conclusion and line.startswith("## "):
                break
            if in_conclusion and len(line.strip()) > 50 and not re.search(r"\[\d+\]", line):
                unsupported.append(line.strip()[:80])
        return unsupported

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

    def _load_sources(self, run_dir: Path) -> list[Source]:
        path = run_dir / "sources.jsonl"
        if not path.exists():
            return []
        sources = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                sources.append(Source.model_validate(json.loads(line)))
        return sources
