from __future__ import annotations

import hashlib
import importlib.resources
import json
import shutil

from researchops.agents.base import AgentBase, RunContext
from researchops.models import AgentResult, PlanOutput, Source, SourceType


class CollectorAgent(AgentBase):
    name = "collector"

    def execute(self, ctx: RunContext) -> AgentResult:
        ctx.trace.log(stage="COLLECT", agent=self.name, action="start")

        plan_path = ctx.run_dir / "plan.json"
        plan = PlanOutput.model_validate(json.loads(plan_path.read_text(encoding="utf-8")))

        sources: list[Source] = []

        if ctx.config.allow_net:
            sources.extend(self._collect_online(ctx, plan))
        else:
            sources.extend(self._collect_offline(ctx))

        sources_path = ctx.run_dir / "sources.jsonl"
        with sources_path.open("w", encoding="utf-8") as f:
            for src in sources:
                f.write(src.model_dump_json() + "\n")

        ctx.trace.log(
            stage="COLLECT",
            agent=self.name,
            action="complete",
            output_summary=f"{len(sources)} sources collected",
        )
        return AgentResult(success=True, message=f"Collected {len(sources)} sources")

    def _collect_online(self, ctx: RunContext, plan: PlanOutput) -> list[Source]:
        sources: list[Source] = []
        for rq in plan.research_questions[: ctx.config.max_collect]:
            try:
                results = ctx.registry.invoke(
                    "web_search",
                    {"query": rq.text, "max_results": 3},
                    trace=ctx.trace,
                )
            except Exception:
                results = []

            for i, r in enumerate(results[:2]):
                url = r.get("url", "")
                try:
                    fetch_result = ctx.registry.invoke(
                        "fetch",
                        {"url": url, "dest_dir": str(ctx.run_dir / "notes")},
                        trace=ctx.trace,
                    )
                    local_path = fetch_result.get("local_path", "")
                    content_hash = fetch_result.get("content_hash", "")
                except Exception:
                    local_path = ""
                    content_hash = ""

                domain = ""
                if "://" in url:
                    domain = url.split("://", 1)[-1].split("/", 1)[0]

                sources.append(
                    Source(
                        source_id=f"src_{rq.rq_id}_{i}",
                        type=SourceType.HTML,
                        url=url,
                        domain=domain,
                        title=r.get("title", ""),
                        local_path=local_path,
                        hash=content_hash,
                    )
                )
        return sources

    def _collect_offline(self, ctx: RunContext) -> list[Source]:
        notes_dir = ctx.run_dir / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)

        samples_pkg = importlib.resources.files("researchops.samples")
        sources: list[Source] = []

        for idx, (sample_name, src_type) in enumerate(
            [("demo.html", SourceType.HTML), ("demo.txt", SourceType.TEXT)]
        ):
            sample_ref = samples_pkg / sample_name
            dest = notes_dir / sample_name
            with importlib.resources.as_file(sample_ref) as src_file:
                shutil.copy2(str(src_file), str(dest))

            content = dest.read_bytes()
            content_hash = hashlib.sha256(content).hexdigest()

            sources.append(
                Source(
                    source_id=f"src_builtin_{idx}",
                    type=src_type,
                    url=f"builtin://{sample_name}",
                    domain="builtin",
                    title=f"Demo {sample_name}",
                    local_path=str(dest),
                    hash=content_hash,
                )
            )

        return sources
