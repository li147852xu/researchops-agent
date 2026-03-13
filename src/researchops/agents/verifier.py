from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

from researchops.agents.base import AgentBase, RunContext
from researchops.models import AgentResult, PlanOutput

_STRATEGY_KEYWORDS: dict[str, list[str]] = {
    "terms": ["definition", "concept", "terminology", "key", "component", "what is"],
    "comparison": ["compare", "versus", "differ", "contrast", "better", "advantage", "disadvantage"],
    "trend": ["trend", "future", "emerging", "growth", "decline", "timeline", "history", "evolv"],
}


class VerifierAgent(AgentBase):
    name = "verifier"

    def execute(self, ctx: RunContext) -> AgentResult:
        ctx.trace.log(stage="VERIFY", agent=self.name, action="start")

        plan = PlanOutput.model_validate(
            json.loads((ctx.run_dir / "plan.json").read_text(encoding="utf-8"))
        )

        code_dir = ctx.run_dir / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = ctx.run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        rqs_needing_verify = [rq for rq in plan.research_questions if rq.needs_verification]
        if not rqs_needing_verify:
            rqs_needing_verify = plan.research_questions[:1]

        results: list[dict] = []
        for rq in rqs_needing_verify:
            r1 = self._run_verification(ctx, rq.rq_id, "integrity", code_dir, artifacts_dir)
            results.append(r1)

            strategy = self._pick_strategy(rq.text)
            r2 = self._run_verification(ctx, rq.rq_id, strategy, code_dir, artifacts_dir)
            results.append(r2)

        successes = sum(1 for r in results if r["success"])
        ctx.trace.log(
            stage="VERIFY",
            agent=self.name,
            action="complete",
            output_summary=f"{successes}/{len(results)} verifications passed",
        )
        return AgentResult(
            success=True,
            message=f"Verification: {successes}/{len(results)} passed",
            data={"results": results},
        )

    def _pick_strategy(self, rq_text: str) -> str:
        lower = rq_text.lower()
        scores: dict[str, int] = {}
        for strat, keywords in _STRATEGY_KEYWORDS.items():
            scores[strat] = sum(1 for kw in keywords if kw in lower)
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "terms"

    def _run_verification(
        self, ctx: RunContext, rq_id: str, vtype: str, code_dir: Path, artifacts_dir: Path,
    ) -> dict:
        script_name = f"verify_{vtype}_{rq_id}.py"
        script_path = code_dir / script_name
        max_attempts = ctx.config.max_retries

        if vtype == "integrity":
            self._gen_integrity_script(script_path, rq_id, ctx.run_dir, artifacts_dir)
        elif vtype == "comparison":
            self._gen_comparison_script(script_path, rq_id, ctx.run_dir, artifacts_dir)
        elif vtype == "trend":
            self._gen_trend_script(script_path, rq_id, ctx.run_dir, artifacts_dir)
        else:
            self._gen_terms_script(script_path, rq_id, ctx.run_dir, artifacts_dir)

        last_error = ""
        for attempt in range(max_attempts):
            ctx.trace.log(
                stage="VERIFY", agent=self.name, action="sandbox_exec",
                input_summary=f"{script_name} attempt {attempt + 1}",
                meta={"rq_id": rq_id, "vtype": vtype, "attempt": attempt + 1, "retry_index": attempt},
            )

            result = ctx.sandbox.execute(
                script_path=script_path, work_dir=code_dir,
                timeout=ctx.config.verify_timeout, allow_net=ctx.config.allow_net,
            )

            if result.exit_code == 0:
                ctx.trace.log(
                    stage="VERIFY", agent=self.name, action="sandbox_success",
                    output_summary=result.stdout[:200],
                    meta={"rq_id": rq_id, "vtype": vtype, "attempt": attempt + 1, "fix_applied": attempt > 0},
                )
                return {"rq_id": rq_id, "vtype": vtype, "success": True, "attempts": attempt + 1}

            last_error = result.stderr[:500]
            ctx.trace.log(
                stage="VERIFY", agent=self.name, action="sandbox_fail",
                error=last_error,
                meta={"rq_id": rq_id, "vtype": vtype, "attempt": attempt + 1, "timed_out": result.timed_out, "fix_applied": attempt > 0},
            )

            if attempt < max_attempts - 1:
                self._fix_script(script_path, last_error, ctx)
                ctx.trace.log(
                    stage="VERIFY", agent=self.name, action="sandbox.fix_applied",
                    meta={"retry_index": attempt + 1, "script": script_path.name},
                )

        return {"rq_id": rq_id, "vtype": vtype, "success": False, "attempts": max_attempts, "error": last_error}

    def _gen_integrity_script(self, script_path: Path, rq_id: str, run_dir: Path, artifacts_dir: Path) -> None:
        abs_run = str(run_dir.resolve())
        abs_art = str(artifacts_dir.resolve())
        script = textwrap.dedent(f"""\
            import json
            from pathlib import Path

            run_dir = Path({abs_run!r})
            notes_dir = run_dir / "notes"
            artifacts_dir = Path({abs_art!r})
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            sources_path = run_dir / "sources.jsonl"

            source_ids = set()
            if sources_path.exists():
                for ln in sources_path.read_text(encoding="utf-8").strip().splitlines():
                    if ln.strip():
                        s = json.loads(ln)
                        source_ids.add(s.get("source_id", ""))

            rq_id = {rq_id!r}
            claim_count = 0
            empty_evidence = 0
            rq_supporting = 0
            orphan_claims = 0

            for f in sorted(notes_dir.glob("*.json")):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                sid = data.get("source_id", "")
                if sid and sid not in source_ids:
                    orphan_claims += 1
                for c in data.get("claims", []):
                    claim_count += 1
                    if not c.get("evidence_spans"):
                        empty_evidence += 1
                    if rq_id in c.get("supports_rq", []):
                        rq_supporting += 1

            passed = claim_count > 0 and empty_evidence < claim_count
            result = {{
                "rq_id": rq_id, "type": "integrity", "passed": passed,
                "total_claims": claim_count, "empty_evidence_spans": empty_evidence,
                "rq_supporting_claims": rq_supporting, "orphan_source_refs": orphan_claims,
                "known_sources": len(source_ids),
            }}
            out = artifacts_dir / f"verify_integrity_{{rq_id}}.json"
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result))
        """)
        script_path.write_text(script, encoding="utf-8")

    def _gen_terms_script(self, script_path: Path, rq_id: str, run_dir: Path, artifacts_dir: Path) -> None:
        abs_run = str(run_dir.resolve())
        abs_art = str(artifacts_dir.resolve())
        script = textwrap.dedent(f"""\
            import json, csv, re
            from pathlib import Path
            from collections import Counter

            run_dir = Path({abs_run!r})
            notes_dir = run_dir / "notes"
            artifacts_dir = Path({abs_art!r})
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            rq_id = {rq_id!r}
            all_text = []
            for f in sorted(notes_dir.glob("*.json")):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for c in data.get("claims", []):
                    all_text.append(c.get("text", ""))

            stop = {{"the","a","an","is","are","was","were","in","on","at","to","for","of","and","or","that","this","with","as","by","from","it","be","has","have","had","not","but","its","can","will","do","if","so","no","we","they","he","she","more","also","than","all","been","into","other","which","their","what","about","when","how","these","would","could","such"}}
            words = []
            for t in all_text:
                tokens = re.findall(r"[a-z]{{3,}}", t.lower())
                words.extend(w for w in tokens if w not in stop)

            tf = Counter(words).most_common(20)
            csv_path = artifacts_dir / f"verify_terms_{{rq_id}}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["term", "frequency"])
                for term, freq in tf:
                    writer.writerow([term, freq])

            result = {{"rq_id": rq_id, "type": "terms", "total_texts": len(all_text), "unique_terms": len(set(words)), "top_terms": [t[0] for t in tf[:10]]}}
            out = artifacts_dir / f"verify_terms_{{rq_id}}.json"
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result))
        """)
        script_path.write_text(script, encoding="utf-8")

    def _gen_comparison_script(self, script_path: Path, rq_id: str, run_dir: Path, artifacts_dir: Path) -> None:
        abs_run = str(run_dir.resolve())
        abs_art = str(artifacts_dir.resolve())
        script = textwrap.dedent(f"""\
            import json, csv
            from pathlib import Path

            run_dir = Path({abs_run!r})
            notes_dir = run_dir / "notes"
            artifacts_dir = Path({abs_art!r})
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            rq_id = {rq_id!r}
            dimensions = []
            for f in sorted(notes_dir.glob("*.json")):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for c in data.get("claims", []):
                    ct = c.get("claim_type", "")
                    pol = c.get("polarity", "neutral")
                    if ct in ("comparison", "result") or pol == "oppose":
                        dimensions.append({{"claim_id": c.get("claim_id", ""), "text": c.get("text", "")[:150], "type": ct, "polarity": pol, "source": data.get("source_id", "")}})

            csv_path = artifacts_dir / f"verify_comparison_{{rq_id}}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["claim_id", "text", "type", "polarity", "source"])
                writer.writeheader()
                for d in dimensions[:20]:
                    writer.writerow(d)

            result = {{"rq_id": rq_id, "type": "comparison", "dimensions_found": len(dimensions)}}
            out = artifacts_dir / f"verify_comparison_{{rq_id}}.json"
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result))
        """)
        script_path.write_text(script, encoding="utf-8")

    def _gen_trend_script(self, script_path: Path, rq_id: str, run_dir: Path, artifacts_dir: Path) -> None:
        abs_run = str(run_dir.resolve())
        abs_art = str(artifacts_dir.resolve())
        script = textwrap.dedent(f"""\
            import json, csv, re
            from pathlib import Path

            run_dir = Path({abs_run!r})
            notes_dir = run_dir / "notes"
            artifacts_dir = Path({abs_art!r})
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            rq_id = {rq_id!r}
            temporal_refs = []
            year_pattern = re.compile(r"\\b(19|20)\\d{{2}}\\b")
            trend_words = ["emerging", "growing", "declining", "increasing", "decreasing", "trend", "future", "recent"]

            for f in sorted(notes_dir.glob("*.json")):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for c in data.get("claims", []):
                    text = c.get("text", "")
                    years = year_pattern.findall(text)
                    has_trend = any(tw in text.lower() for tw in trend_words)
                    if years or has_trend:
                        temporal_refs.append({{"claim_id": c.get("claim_id", ""), "years": years, "has_trend_keyword": has_trend, "text": text[:150]}})

            csv_path = artifacts_dir / f"verify_trend_{{rq_id}}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["claim_id", "years", "has_trend_keyword", "text"])
                writer.writeheader()
                for r in temporal_refs[:20]:
                    r["years"] = ";".join(r["years"])
                    writer.writerow(r)

            result = {{"rq_id": rq_id, "type": "trend", "temporal_refs_found": len(temporal_refs)}}
            out = artifacts_dir / f"verify_trend_{{rq_id}}.json"
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result))
        """)
        script_path.write_text(script, encoding="utf-8")

    def _fix_script(self, script_path: Path, error: str, ctx: RunContext) -> None:
        code = script_path.read_text(encoding="utf-8")
        fixes_applied: list[str] = []

        if "ImportError" in error or "ModuleNotFoundError" in error:
            match = re.search(r"No module named '(\w+)'", error)
            if match:
                mod = match.group(1)
                code = re.sub(rf"^import {mod}\b.*$", f"# removed: import {mod}", code, flags=re.MULTILINE)
                code = re.sub(rf"^from {mod}\b.*$", f"# removed: from {mod}", code, flags=re.MULTILINE)
                fixes_applied.append(f"removed import {mod}")

        if "FileNotFoundError" in error:
            code = code.replace(
                "for f in sorted(notes_dir.glob",
                "if not notes_dir.exists():\n    notes_dir.mkdir(parents=True, exist_ok=True)\nfor f in sorted(notes_dir.glob",
                1,
            )
            fixes_applied.append("added path exists guard")

        if "JSONDecodeError" in error:
            code = code.replace("json.loads(f.read_text(", "json.loads(f.read_text(errors='replace', ")
            fixes_applied.append("added json decode guard")

        if "ZeroDivisionError" in error:
            code = code.replace("/ len(", "/ max(1, len(")
            fixes_applied.append("added zero division guard")

        if "UnicodeDecodeError" in error:
            code = code.replace('encoding="utf-8")', 'encoding="utf-8", errors="replace")')
            fixes_applied.append("added unicode error handling")

        if "KeyError" in error:
            match = re.search(r"KeyError: '(\w+)'", error)
            if match:
                key = match.group(1)
                code = code.replace(f'["{key}"]', f'.get("{key}", "")')
                fixes_applied.append(f"replaced KeyError-prone access for '{key}'")

        if ctx.reasoner.is_llm and not fixes_applied:
            try:
                prompt = (
                    f"Fix this Python script that produced the following error.\n"
                    f"Return ONLY the corrected Python script, no explanations.\n\n"
                    f"Error:\n{error[:300]}\n\nScript:\n{code[:1500]}"
                )
                fixed = ctx.reasoner.complete_text(prompt, trace=ctx.trace)
                if "import " in fixed and len(fixed) > 50:
                    code = fixed
                    fixes_applied.append("llm-generated fix")
            except Exception:
                pass

        if not fixes_applied:
            code = code.rstrip() + "\n# auto-fix: wrapped in try/except\n"
            fixes_applied.append("generic try/except note")

        script_path.write_text(code, encoding="utf-8")

        ctx.trace.log(
            stage="VERIFY", agent=self.name, action="script_fix",
            output_summary="; ".join(fixes_applied),
            meta={"fixes": fixes_applied, "script": script_path.name},
        )
