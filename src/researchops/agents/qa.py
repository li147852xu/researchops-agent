"""QA agent — checks report quality and triggers rollbacks for gaps."""

from __future__ import annotations

import json
import logging
import re

from researchops.agents.base import AgentBase, RunContext
from researchops.models import AgentResult, SourceNotes, Stage
from researchops.utils import load_all_notes, load_plan, load_sources

logger = logging.getLogger(__name__)

_MIN_SENTENCES_PER_SECTION = 2
_MIN_CITATION_COVERAGE = 0.40
_MAX_SINGLE_SOURCE_RATIO = 0.60


class QAAgent(AgentBase):
    name = "qa"

    def execute(self, ctx: RunContext) -> AgentResult:
        ctx.trace.log(stage="QA", agent=self.name, action="start")

        report_path = ctx.run_dir / "report.md"
        if not report_path.exists():
            return AgentResult(
                success=False, message="No report found", rollback_to=Stage.WRITE,
            )
        report = report_path.read_text(encoding="utf-8")

        plan = load_plan(ctx.run_dir)
        sources = load_sources(ctx.run_dir)
        all_notes = load_all_notes(ctx.run_dir)
        source_map = {s.source_id: s for s in sources}

        issues: list[dict] = []
        citation_ids = set(re.findall(r"\[@(\w+)\]", report))
        valid_citations = {cid for cid in citation_ids if cid in source_map}

        # 1. Citation coverage
        sections = re.split(r"^## ", report, flags=re.MULTILINE)
        total_statements = 0
        cited_statements = 0
        for section in sections[1:]:
            lines = section.strip().splitlines()
            for line in lines[1:]:
                line = line.strip()
                if len(line) < 30 or line.startswith("#") or line.startswith("*"):
                    continue
                total_statements += 1
                if re.search(r"\[@\w+\]", line):
                    cited_statements += 1

        citation_coverage = cited_statements / max(1, total_statements)
        if citation_coverage < _MIN_CITATION_COVERAGE:
            issues.append({
                "type": "low_citation_coverage",
                "detail": f"Citation coverage {citation_coverage:.0%} < {_MIN_CITATION_COVERAGE:.0%}",
                "severity": "high",
            })

        # 2. Invalid citations (referencing non-existent sources)
        invalid = citation_ids - {s.source_id for s in sources}
        if invalid:
            issues.append({
                "type": "invalid_citations",
                "detail": f"{len(invalid)} invalid citation(s): {', '.join(list(invalid)[:3])}",
                "severity": "medium",
            })

        # 3. Source diversity
        if citation_ids:
            from collections import Counter
            cite_counter = Counter(re.findall(r"\[@(\w+)\]", report))
            if cite_counter:
                top_source, top_count = cite_counter.most_common(1)[0]
                ratio = top_count / sum(cite_counter.values())
                if ratio > _MAX_SINGLE_SOURCE_RATIO:
                    issues.append({
                        "type": "source_concentration",
                        "detail": f"Source {top_source} accounts for {ratio:.0%} of citations",
                        "severity": "medium",
                    })

        # 4. Bucket coverage
        bucket_coverage_rate = 1.0
        if plan:
            covered_buckets: set[str] = set()
            for _sid, notes in all_notes.items():
                for bid in notes.bucket_hits:
                    covered_buckets.add(bid)
            total_buckets = len(plan.coverage_checklist)
            if total_buckets > 0:
                bucket_coverage_rate = len(covered_buckets) / total_buckets
                uncovered = [
                    b.get("bucket_name", b.get("bucket_id", ""))
                    for b in plan.coverage_checklist
                    if b.get("bucket_id") not in covered_buckets
                ]
                if uncovered:
                    issues.append({
                        "type": "bucket_gap",
                        "detail": f"Uncovered buckets: {', '.join(uncovered[:3])}",
                        "severity": "high" if bucket_coverage_rate < 0.5 else "medium",
                    })

        # 5. RQ claim coverage
        rq_claim_counts = ctx.shared.get("rq_claim_counts", {})
        low_rqs = [rq_id for rq_id, cnt in rq_claim_counts.items() if cnt < ctx.config.min_claims_per_rq]
        if low_rqs:
            issues.append({
                "type": "rq_undercovered",
                "detail": f"RQs with few claims: {', '.join(low_rqs[:3])}",
                "severity": "medium",
            })

        # 6. Evidence gaps in report
        gap_count = report.count("Evidence Gap") + report.count("证据缺口")
        if gap_count > 0:
            issues.append({
                "type": "evidence_gaps",
                "detail": f"{gap_count} evidence gap section(s) in report",
                "severity": "high",
            })

        # 7. Report length check
        word_count = len(report.split())
        min_words = 300 if ctx.config.mode.value == "fast" else 600
        if word_count < min_words:
            issues.append({
                "type": "report_too_short",
                "detail": f"Report has {word_count} words, minimum is {min_words}",
                "severity": "medium",
            })

        # 8. Conflict detection
        conflicts = self._detect_conflicts(all_notes)
        if conflicts:
            ctx.shared["has_conflicts"] = True
            ctx.shared["conflict_count"] = len(conflicts)
            conflicts_path = ctx.run_dir / "qa_conflicts.json"
            conflicts_path.write_text(
                json.dumps({"conflicts": conflicts}, indent=2), encoding="utf-8",
            )

        # 9. Relevance scoring
        relevance_scores: list[float] = []
        for _sid, notes in all_notes.items():
            if notes.relevance_score > 0:
                relevance_scores.append(notes.relevance_score)
        relevance_avg = sum(relevance_scores) / max(1, len(relevance_scores)) if relevance_scores else 0.0

        # Build QA report
        qa_report = {
            "citation_coverage": round(citation_coverage, 3),
            "valid_citations": len(valid_citations),
            "total_citations": len(citation_ids),
            "bucket_coverage_rate": round(bucket_coverage_rate, 3),
            "relevance_avg": round(relevance_avg, 3),
            "issue_count": len(issues),
            "issues": issues,
        }
        qa_path = ctx.run_dir / "qa_report.json"
        qa_path.write_text(json.dumps(qa_report, indent=2), encoding="utf-8")

        # Decision: pass or rollback
        high_issues = [i for i in issues if i.get("severity") == "high"]
        at_max_rounds = ctx.state.collect_rounds >= ctx.config.max_collect_rounds

        if not high_issues or at_max_rounds:
            ctx.state.coverage_vector = rq_claim_counts
            ctx.trace.log(
                stage="QA", agent=self.name, action="complete",
                output_summary=f"QA passed: {len(issues)} issues (none high-severity or max rounds reached)",
            )
            return AgentResult(
                success=True,
                message=f"QA passed with {len(issues)} issues",
                data={"issues": issues, "qa_report": qa_report},
            )

        # Determine rollback target
        if gap_count > 0 or bucket_coverage_rate < 0.5 or low_rqs:
            rollback_target = Stage.COLLECT
        elif citation_coverage < _MIN_CITATION_COVERAGE:
            rollback_target = Stage.WRITE
        else:
            rollback_target = Stage.COLLECT

        ctx.state.coverage_vector = rq_claim_counts
        ctx.trace.log(
            stage="QA", agent=self.name, action="rollback",
            output_summary=f"QA failed: {len(high_issues)} high-severity issues → {rollback_target.value}",
        )
        return AgentResult(
            success=False,
            message=f"QA found {len(high_issues)} high-severity issues",
            data={"issues": issues, "qa_report": qa_report},
            rollback_to=rollback_target,
        )

    def _detect_conflicts(self, all_notes: dict[str, SourceNotes]) -> list[dict]:
        rq_support: dict[str, int] = {}
        rq_oppose: dict[str, int] = {}
        for _sid, notes in all_notes.items():
            for claim in notes.claims:
                for rq_id in claim.supports_rq:
                    if claim.polarity == "support":
                        rq_support[rq_id] = rq_support.get(rq_id, 0) + 1
                    elif claim.polarity == "oppose":
                        rq_oppose[rq_id] = rq_oppose.get(rq_id, 0) + 1

        conflicts: list[dict] = []
        for rq_id in rq_support:
            sup = rq_support.get(rq_id, 0)
            opp = rq_oppose.get(rq_id, 0)
            if sup > 0 and opp > 0:
                conflicts.append({
                    "rq_id": rq_id,
                    "support_count": sup,
                    "oppose_count": opp,
                })
        return conflicts
