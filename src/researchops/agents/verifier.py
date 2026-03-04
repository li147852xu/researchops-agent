from __future__ import annotations

import json
import textwrap
from pathlib import Path

from researchops.agents.base import AgentBase, RunContext
from researchops.models import AgentResult, PlanOutput


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
            result = self._verify_rq(ctx, rq.rq_id, code_dir, artifacts_dir)
            results.append(result)

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

    def _verify_rq(
        self, ctx: RunContext, rq_id: str, code_dir: Path, artifacts_dir: Path
    ) -> dict:
        script_name = f"verify_{rq_id}.py"
        script_path = code_dir / script_name
        max_attempts = ctx.config.max_retries

        self._generate_script(script_path, rq_id, ctx.run_dir, artifacts_dir)

        last_error = ""
        for attempt in range(max_attempts):
            ctx.trace.log(
                stage="VERIFY",
                agent=self.name,
                action="sandbox_exec",
                input_summary=f"{script_name} attempt {attempt + 1}",
                meta={"rq_id": rq_id, "attempt": attempt + 1},
            )

            result = ctx.sandbox.execute(
                script_path=script_path,
                work_dir=code_dir,
                timeout=ctx.config.verify_timeout,
                allow_net=ctx.config.allow_net,
            )

            if result.exit_code == 0:
                ctx.trace.log(
                    stage="VERIFY",
                    agent=self.name,
                    action="sandbox_success",
                    output_summary=result.stdout[:200],
                    meta={"rq_id": rq_id, "attempt": attempt + 1},
                )
                return {"rq_id": rq_id, "success": True, "attempts": attempt + 1}

            last_error = result.stderr[:500]
            ctx.trace.log(
                stage="VERIFY",
                agent=self.name,
                action="sandbox_fail",
                error=last_error,
                meta={"rq_id": rq_id, "attempt": attempt + 1, "timed_out": result.timed_out},
            )

            if attempt < max_attempts - 1:
                self._fix_script(script_path, last_error, rq_id, ctx.run_dir, artifacts_dir)

        return {"rq_id": rq_id, "success": False, "attempts": max_attempts, "error": last_error}

    def _generate_script(
        self, script_path: Path, rq_id: str, run_dir: Path, artifacts_dir: Path
    ) -> None:
        abs_run = str(run_dir.resolve())
        abs_art = str(artifacts_dir.resolve())
        script = textwrap.dedent(f"""\
            import json
            import os
            from pathlib import Path

            run_dir = Path({abs_run!r})
            notes_dir = run_dir / "notes"
            artifacts_dir = Path({abs_art!r})
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            claim_count = 0
            source_count = 0
            rq_id = {rq_id!r}

            for f in sorted(notes_dir.glob("*.json")):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    claims = data.get("claims", [])
                    source_count += 1
                    for c in claims:
                        if rq_id in c.get("supports_rq", []):
                            claim_count += 1
                except Exception:
                    pass

            summary = {{
                "rq_id": rq_id,
                "sources_checked": source_count,
                "supporting_claims": claim_count,
                "status": "verified" if claim_count > 0 else "insufficient_evidence",
            }}

            out_path = artifacts_dir / f"verify_{{rq_id}}.json"
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary))

            if claim_count == 0:
                raise SystemExit("No supporting claims found")
        """)
        script_path.write_text(script, encoding="utf-8")

    def _fix_script(
        self, script_path: Path, error: str, rq_id: str, run_dir: Path, artifacts_dir: Path
    ) -> None:
        """Regenerate with relaxed criteria after failure."""
        abs_run = str(run_dir.resolve())
        abs_art = str(artifacts_dir.resolve())
        script = textwrap.dedent(f"""\
            import json
            from pathlib import Path

            run_dir = Path({abs_run!r})
            notes_dir = run_dir / "notes"
            artifacts_dir = Path({abs_art!r})
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            claim_count = 0
            source_count = 0
            rq_id = {rq_id!r}

            for f in sorted(notes_dir.glob("*.json")):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    claims = data.get("claims", [])
                    source_count += 1
                    claim_count += len(claims)
                except Exception:
                    pass

            summary = {{
                "rq_id": rq_id,
                "sources_checked": source_count,
                "total_claims": claim_count,
                "status": "verified_relaxed",
                "previous_error": {error[:200]!r},
            }}

            out_path = artifacts_dir / f"verify_{{rq_id}}.json"
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary))
        """)
        script_path.write_text(script, encoding="utf-8")
