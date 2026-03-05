from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from researchops.agents.base import AgentBase, RunContext
from researchops.models import AgentResult, PlanOutput, Source, SourceNotes, Stage
from researchops.tools.parse_doc import NOISE_PATTERNS


def _normalize_sentence(text: str) -> str:
    return re.sub(r"[\s\d\W]+", "", text.lower())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


_SOURCE_CAP = 0.35


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

        checks: list[dict] = []
        hard_issues: list[str] = []

        code_garbage_issues = self._code_garbage_detector(report)
        hard_issues.extend(code_garbage_issues)
        checks.append({"check": "code_garbage", "passed": len(code_garbage_issues) == 0, "issues": code_garbage_issues})

        boilerplate_issues = self._boilerplate_detector(report)
        hard_issues.extend(boilerplate_issues)
        checks.append({"check": "boilerplate", "passed": len(boilerplate_issues) == 0, "issues": boilerplate_issues})

        paste_issues = self._paste_detector(report)
        hard_issues.extend(paste_issues)
        checks.append({"check": "paste", "passed": len(paste_issues) == 0, "issues": paste_issues})

        redundancy_issues = self._redundancy_detector(report)
        hard_issues.extend(redundancy_issues)
        checks.append({"check": "redundancy", "passed": len(redundancy_issues) == 0, "issues": redundancy_issues})

        source_issues = self._source_availability_check(
            sources, ctx.config.mode.value if hasattr(ctx.config.mode, "value") else str(ctx.config.mode),
        )
        hard_issues.extend(source_issues)
        checks.append({"check": "source_availability", "passed": len(source_issues) == 0, "issues": source_issues})

        sq_issues = self._source_quality_check(ctx)
        hard_issues.extend(sq_issues)
        checks.append({"check": "source_quality", "passed": len(sq_issues) == 0, "issues": sq_issues})

        at_max_rounds = ctx.state.collect_rounds >= ctx.config.max_collect_rounds

        if hard_issues and ctx.state.retry_counts.get("qa", 0) < 1 and not at_max_rounds:
            ctx.state.retry_counts["qa"] = ctx.state.retry_counts.get("qa", 0) + 1
            rollback_target = self._route_quality_rollback(hard_issues, ctx.run_dir, allow_net=ctx.config.allow_net)

            next_actions = self._compute_next_actions(hard_issues, all_notes, sources, ctx)

            ctx.trace.log(
                stage="QA", agent=self.name, action="qa.rollback_decision",
                output_summary=f"Quality fail -> {rollback_target.value}",
                meta={"hard_issues": hard_issues, "target": rollback_target.value, "next_actions": next_actions},
            )
            self._write_qa_report(ctx.run_dir, checks, next_actions=next_actions)
            return AgentResult(
                success=False,
                message=f"QA quality fail: {'; '.join(hard_issues)}",
                rollback_to=rollback_target,
            )

        issues: list[str] = list(hard_issues)

        cov = self._check_citation_coverage(report)
        checks.append({"check": "citation_coverage", "passed": cov >= 0.3, "value": cov, "threshold": 0.3})
        if cov < 0.3:
            issues.append(f"Low citation coverage: {cov:.0%}")

        diversity = self._check_source_diversity(sources)
        checks.append({"check": "source_diversity", "passed": diversity["unique_domains"] >= 1, "value": diversity})
        if diversity["unique_domains"] < 1:
            issues.append("No source diversity")

        domination_issues = self._check_source_domination(report, sources)
        if domination_issues:
            issues.extend(domination_issues)
            checks.append({"check": "source_domination", "passed": False, "issues": domination_issues})
        else:
            checks.append({"check": "source_domination", "passed": True, "issues": []})

        traceability = self._check_traceability(ctx.run_dir, all_notes)
        unsupported_rate = traceability.get("unsupported_rate", 0.0)
        is_deep = ctx.config.mode.value == "deep" if hasattr(ctx.config.mode, "value") else False
        unsupported_threshold = 0.05 if is_deep else 0.20
        checks.append({"check": "unsupported_claim_rate", "passed": unsupported_rate <= unsupported_threshold, "value": unsupported_rate, "threshold": unsupported_threshold})
        if unsupported_rate > unsupported_threshold:
            issues.append(f"High unsupported claim rate: {unsupported_rate:.0%} (max {unsupported_threshold:.0%})")

        unsupported_assertions = self._scan_unsupported_assertions(report)
        if unsupported_assertions:
            issues.append(f"{len(unsupported_assertions)} assertions without citation markers")

        evidence_gaps = self._detect_evidence_gaps(report)
        if evidence_gaps:
            issues.append(f"{len(evidence_gaps)} sections with evidence gaps")
            checks.append({"check": "evidence_gaps", "passed": False, "value": evidence_gaps})
        else:
            checks.append({"check": "evidence_gaps", "passed": True, "value": []})

        bucket_cov_rate = self._check_bucket_coverage(ctx.run_dir, all_notes, ctx)
        bucket_threshold = ctx.config.bucket_coverage_threshold
        checks.append({
            "check": "bucket_coverage",
            "passed": bucket_cov_rate >= bucket_threshold,
            "value": bucket_cov_rate,
            "threshold": bucket_threshold,
        })
        if bucket_cov_rate < bucket_threshold:
            issues.append(f"Bucket coverage {bucket_cov_rate:.0%} below threshold {bucket_threshold:.0%}")

        relevance_avg = self._check_relevance_avg(all_notes)
        rel_threshold = ctx.config.relevance_threshold
        checks.append({
            "check": "relevance_avg",
            "passed": relevance_avg >= rel_threshold,
            "value": relevance_avg,
            "threshold": rel_threshold,
        })
        if relevance_avg < rel_threshold:
            issues.append(f"Average relevance {relevance_avg:.2f} below threshold {rel_threshold:.2f}")

        conflicts = self._conflict_scan(all_notes)
        conflicts_path = ctx.run_dir / "qa_conflicts.json"
        conflicts_path.write_text(json.dumps(conflicts, indent=2), encoding="utf-8")
        ctx.shared["has_conflicts"] = len(conflicts.get("conflicts", [])) > 0
        ctx.shared["conflict_count"] = len(conflicts.get("conflicts", []))
        checks.append({"check": "conflict_scan", "passed": len(conflicts.get("conflicts", [])) == 0, "value": len(conflicts.get("conflicts", []))})

        if conflicts.get("conflicts"):
            issues.append(f"{len(conflicts['conflicts'])} claim conflicts detected")

        next_actions = self._compute_next_actions(issues, all_notes, sources, ctx) if issues else {}
        self._write_qa_report(
            ctx.run_dir, checks, next_actions=next_actions,
            bucket_coverage_rate=bucket_cov_rate, relevance_avg=relevance_avg,
        )

        rollback_target = self._determine_rollback(issues, traceability, conflicts, domination_issues, evidence_gaps, ctx)

        ctx.trace.log(
            stage="QA", agent=self.name, action="complete",
            output_summary=f"{len(issues)} issues found" if issues else "QA passed",
            meta={
                "citation_coverage": cov,
                "unsupported_rate": unsupported_rate,
                "conflict_count": len(conflicts.get("conflicts", [])),
                "issues": issues,
                "diversity": diversity,
                "hard_issues": hard_issues,
            },
        )

        soft_issues = [i for i in issues if i not in hard_issues]
        if soft_issues and ctx.state.retry_counts.get("qa", 0) < 1 and not at_max_rounds and self._should_allow_rollback(ctx):
                ctx.state.retry_counts["qa"] = ctx.state.retry_counts.get("qa", 0) + 1
                ctx.trace.log(
                    stage="QA", agent=self.name, action="qa.rollback_decision",
                    output_summary=f"Rolling back to {rollback_target.value}",
                    meta={"issues": soft_issues, "target": rollback_target.value},
                )
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
                "conflict_count": len(conflicts.get("conflicts", [])),
                "diversity": diversity,
            },
        )

    def _should_allow_rollback(self, ctx: RunContext) -> bool:
        history = ctx.state.rollback_history
        if len(history) < 2:
            return True
        last_two = history[-2:]
        return not (
            last_two[0].get("sources_hash") == last_two[1].get("sources_hash")
            and last_two[0].get("claims_hash") == last_two[1].get("claims_hash")
        )

    def _check_bucket_coverage(
        self, run_dir: Path, all_notes: dict[str, SourceNotes], ctx: RunContext,
    ) -> float:
        plan_path = run_dir / "plan.json"
        if not plan_path.exists():
            return 1.0
        plan = PlanOutput.model_validate(json.loads(plan_path.read_text(encoding="utf-8")))
        checklist = plan.coverage_checklist
        if not checklist:
            return 1.0

        bucket_source_counts: dict[str, int] = {}
        for _sid, notes in all_notes.items():
            for bid in notes.bucket_hits:
                bucket_source_counts[bid] = bucket_source_counts.get(bid, 0) + 1

        covered = 0
        for bucket in checklist:
            bid = bucket.get("bucket_id", "")
            min_sources = bucket.get("min_sources", 1)
            if bucket_source_counts.get(bid, 0) >= min_sources:
                covered += 1

        return covered / max(1, len(checklist))

    def _check_relevance_avg(self, all_notes: dict[str, SourceNotes]) -> float:
        if not all_notes:
            return 0.0
        total = sum(n.relevance_score for n in all_notes.values())
        return total / len(all_notes)

    def _compute_next_actions(
        self, issues: list[str], all_notes: dict[str, SourceNotes],
        sources: list[Source], ctx: RunContext,
    ) -> dict:
        queries: list[str] = []
        categories: list[str] = []
        target_count = ctx.config.target_sources_per_rq
        bucket_gaps: list[dict] = []

        issue_text = " ".join(issues).lower()

        if "evidence gap" in issue_text or "insufficient" in issue_text:
            queries.append(f"{ctx.config.topic} survey overview")
            queries.append(f"{ctx.config.topic} recent advances challenges")

        if "diversity" in issue_text or "domination" in issue_text or "dominates" in issue_text:
            queries.append(f"{ctx.config.topic} alternative approaches")
            queries.append(f"{ctx.config.topic} comparison benchmark")

        if "quality" in issue_text:
            categories.append("cs.LG")
            categories.append("cs.AI")

        if "bucket coverage" in issue_text:
            plan_path = ctx.run_dir / "plan.json"
            if plan_path.exists():
                plan = PlanOutput.model_validate(json.loads(plan_path.read_text(encoding="utf-8")))
                bucket_source_counts: dict[str, int] = {}
                for _sid, notes in all_notes.items():
                    for bid in notes.bucket_hits:
                        bucket_source_counts[bid] = bucket_source_counts.get(bid, 0) + 1
                for bucket in plan.coverage_checklist:
                    bid = bucket.get("bucket_id", "")
                    min_src = bucket.get("min_sources", 1)
                    actual = bucket_source_counts.get(bid, 0)
                    if actual < min_src:
                        bname = bucket.get("bucket_name", bid)
                        bucket_gaps.append({"bucket_id": bid, "bucket_name": bname, "need": min_src - actual})
                        queries.append(f"{ctx.config.topic} {bname}")

        if "relevance" in issue_text:
            queries.append(f"{ctx.config.topic} core concepts foundations")

        return {
            "queries": queries,
            "categories": categories,
            "target_sources_count": target_count,
            "bucket_gaps": bucket_gaps,
        }

    def _detect_evidence_gaps(self, report: str) -> list[str]:
        gaps = []
        sections = re.split(r"^##\s+", report, flags=re.MULTILINE)
        for sec in sections[1:]:
            heading = sec.split("\n", 1)[0].strip()
            lower = sec.lower()
            if "evidence gap" in lower or "evidence insufficient" in lower or "证据不足" in lower or "证据缺口" in lower:
                gaps.append(heading)
        return gaps

    def _check_source_domination(self, report: str, sources: list[Source]) -> list[str]:
        issues: list[str] = []
        sections = re.split(r"^##\s+", report, flags=re.MULTILINE)
        for sec in sections[1:]:
            markers = re.findall(r"\[@(\w+)\]", sec)
            if len(markers) < 3:
                continue
            mc = Counter(markers).most_common(1)
            if mc:
                dominant_id, dominant_count = mc[0]
                ratio = dominant_count / len(markers)
                if ratio > _SOURCE_CAP:
                    heading = sec.split("\n", 1)[0].strip()
                    issues.append(f"Source {dominant_id} dominates section '{heading}' ({ratio:.0%})")
        return issues

    def _write_qa_report(
        self, run_dir: Path, checks: list[dict],
        next_actions: dict | None = None,
        bucket_coverage_rate: float = 0.0,
        relevance_avg: float = 0.0,
    ) -> None:
        report: dict = {
            "checks": checks,
            "all_passed": all(c.get("passed", True) for c in checks),
            "bucket_coverage_rate": round(bucket_coverage_rate, 3),
            "relevance_avg": round(relevance_avg, 3),
        }
        if next_actions:
            report["next_actions"] = next_actions
        (run_dir / "qa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def _source_quality_check(self, ctx: RunContext) -> list[str]:
        issues: list[str] = []
        low_quality = ctx.shared.get("low_quality_sources", [])
        sources = self._load_sources(ctx.run_dir)
        if not sources:
            return issues
        ratio = len(low_quality) / max(1, len(sources))
        if ratio > 0.50:
            issues.append(f"source_quality: {ratio:.0%} of sources are low quality")
        return issues

    _CODE_GARBAGE_KEYWORDS = [
        "function(", "padding", "px;", "rgb(", "rgba(",
        "queryselector", "addeventlistener", "stylesheet",
        "meta charset", "doctype", "viewport", "margin:",
        "@media", "@import", "background-color", "font-size",
    ]

    _MIN_SOURCES = {"fast": 2, "deep": 6}

    def _code_garbage_detector(self, report: str) -> list[str]:
        issues: list[str] = []
        lines = [ln for ln in report.split("\n") if ln.strip() and not ln.strip().startswith("#")]
        hit_count = 0
        for ln in lines:
            lower = ln.lower()
            if any(kw in lower for kw in self._CODE_GARBAGE_KEYWORDS):
                hit_count += 1
        if hit_count > 2:
            issues.append(f"code_garbage_detector: {hit_count} lines with CSS/JS/HTML code keywords")
        return issues

    def _source_availability_check(self, sources: list[Source], mode: str) -> list[str]:
        issues: list[str] = []
        valid_sources = [s for s in sources if s.local_path and Path(s.local_path).exists()]
        min_required = self._MIN_SOURCES.get(mode, 2)
        if len(valid_sources) < min_required:
            issues.append(
                f"source_availability: only {len(valid_sources)} valid sources "
                f"(minimum {min_required} for {mode} mode)"
            )
        return issues

    def _boilerplate_detector(self, report: str) -> list[str]:
        issues: list[str] = []
        lines = report.split("\n")
        content_lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        if not content_lines:
            return issues

        noise_count = 0
        for ln in content_lines:
            lower = ln.lower()
            if any(pat in lower for pat in NOISE_PATTERNS):
                noise_count += 1

        ratio = noise_count / max(1, len(content_lines))
        if noise_count > 3 or ratio > 0.01:
            issues.append(f"boilerplate_detector: {noise_count} noise lines ({ratio:.1%})")
        return issues

    def _paste_detector(self, report: str) -> list[str]:
        issues: list[str] = []
        paragraphs = [p.strip() for p in report.split("\n\n") if len(p.strip()) > 20]

        for para in paragraphs:
            if para.startswith("#") or para.startswith("*") or para.startswith("-"):
                continue
            if len(para) > 600:
                issues.append(f"paste_detector: paragraph exceeds 600 chars ({len(para)})")
                break

        for i in range(len(paragraphs) - 1):
            a = paragraphs[i]
            b = paragraphs[i + 1]
            if a.startswith("#") or b.startswith("#"):
                continue
            words_a = set(re.findall(r"\w{3,}", a.lower()))
            words_b = set(re.findall(r"\w{3,}", b.lower()))
            if _jaccard(words_a, words_b) > 0.85:
                issues.append("paste_detector: consecutive paragraphs highly similar (Jaccard>0.85)")
                break

        return issues

    def _redundancy_detector(self, report: str) -> list[str]:
        issues: list[str] = []
        all_sentences = re.split(r"(?<=[.!?。！？])\s*", report)
        normalized: list[str] = []
        for s in all_sentences:
            s = s.strip()
            if len(s) > 30 and not s.startswith("#") and not s.startswith("-") and not s.startswith("*"):
                normalized.append(_normalize_sentence(s))

        counts = Counter(normalized)
        dup_count = sum(c - 1 for c in counts.values() if c > 1)
        if dup_count > 10:
            issues.append(f"redundancy_detector: {dup_count} duplicate sentences")
        return issues

    def _route_quality_rollback(self, hard_issues: list[str], run_dir: Path, *, allow_net: bool = False) -> Stage:
        issues_text = " ".join(hard_issues).lower()
        if "code_garbage" in issues_text or "source_availability" in issues_text or "source_quality" in issues_text:
            return Stage.COLLECT if allow_net else Stage.READ

        index_path = run_dir / "report_index.json"
        if not index_path.exists():
            return Stage.WRITE

        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            entries = index.get("entries", [])
            noisy_claims = 0
            for entry in entries:
                text = entry.get("text", "").lower()
                if any(p in text for p in NOISE_PATTERNS):
                    noisy_claims += 1
            if noisy_claims > len(entries) * 0.3:
                return Stage.READ
        except Exception:
            pass

        return Stage.WRITE

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
            if entry.get("evidence_gap"):
                continue
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

    def _scan_unsupported_assertions(self, report: str) -> list[str]:
        unsupported = []
        for line in report.split("\n"):
            stripped = line.strip()
            if (
                len(stripped) > 60
                and not stripped.startswith("#")
                and not stripped.startswith("-")
                and not stripped.startswith("*")
                and not re.search(r"\[@\w+\]", stripped)
                and "evidence gap" not in stripped.lower()
                and "证据缺口" not in stripped
                and "证据不足" not in stripped
            ):
                unsupported.append(stripped[:80])
        return unsupported

    def _conflict_scan(self, all_notes: dict[str, SourceNotes]) -> dict:
        rq_claims: dict[str, list[dict]] = {}
        for _sid, notes in all_notes.items():
            for claim in notes.claims:
                for rq_id in claim.supports_rq:
                    rq_claims.setdefault(rq_id, []).append({
                        "claim_id": claim.claim_id,
                        "text": claim.text[:200],
                        "polarity": claim.polarity,
                        "source_id": notes.source_id,
                    })

        conflicts: list[dict] = []
        for rq_id, claims in rq_claims.items():
            supporters = [c for c in claims if c["polarity"] == "support"]
            opposers = [c for c in claims if c["polarity"] == "oppose"]
            if supporters and opposers:
                conflicts.append({
                    "rq_id": rq_id,
                    "support_count": len(supporters),
                    "oppose_count": len(opposers),
                    "example_support": supporters[0]["text"][:100],
                    "example_oppose": opposers[0]["text"][:100],
                })

        return {"conflicts": conflicts, "total_rqs_scanned": len(rq_claims)}

    def _determine_rollback(
        self, issues: list[str], traceability: dict, conflicts: dict,
        domination_issues: list[str], evidence_gaps: list[str], ctx: RunContext,
    ) -> Stage:
        if evidence_gaps and ctx.config.allow_net:
            ctx.trace.log(
                stage="QA", agent=self.name, action="qa.rollback_decision",
                output_summary="Evidence gaps detected -> COLLECT",
            )
            return Stage.COLLECT

        if domination_issues and ctx.config.allow_net:
            ctx.trace.log(
                stage="QA", agent=self.name, action="qa.rollback_decision",
                output_summary="Source domination detected -> COLLECT for diversity",
            )
            return Stage.COLLECT

        if conflicts.get("conflicts") and ctx.config.allow_net:
            ctx.trace.log(
                stage="QA", agent=self.name, action="qa.rollback_decision",
                output_summary="Conflicts detected with net enabled -> COLLECT",
            )
            return Stage.COLLECT

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
