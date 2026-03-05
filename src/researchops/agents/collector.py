from __future__ import annotations

import hashlib
import importlib.resources
import json
import shutil

from researchops.agents.base import AgentBase, RunContext
from researchops.models import AgentResult, PlanOutput, Source, SourceType
from researchops.registry.manager import ToolPermissionError

_TYPE_MAP = {"pdf": SourceType.PDF, "html": SourceType.HTML, "text": SourceType.TEXT}


class CollectorAgent(AgentBase):
    name = "collector"

    def execute(self, ctx: RunContext) -> AgentResult:
        ctx.trace.log(stage="COLLECT", agent=self.name, action="start")

        plan_path = ctx.run_dir / "plan.json"
        plan = PlanOutput.model_validate(json.loads(plan_path.read_text(encoding="utf-8")))

        sources: list[Source] = []

        if ctx.config.allow_net:
            try:
                sources.extend(self._collect_online(ctx, plan))
            except ToolPermissionError:
                ctx.trace.log(
                    stage="COLLECT",
                    agent=self.name,
                    action="tool.denied_fallback",
                    output_summary="Net tools denied, falling back to offline samples",
                )
                sources.extend(self._collect_offline(ctx))
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
        downloads_dir = str(ctx.run_dir / "downloads")

        for rq in plan.research_questions[: ctx.config.max_collect]:
            try:
                results = ctx.registry.invoke(
                    "web_search",
                    {"query": rq.text, "max_results": 3},
                    trace=ctx.trace,
                )
            except ToolPermissionError:
                raise
            except Exception:
                results = []

            for i, r in enumerate(results[:3]):
                url = r.get("url", "")
                if not url:
                    continue

                try:
                    fetch_result = ctx.registry.invoke(
                        "fetch",
                        {"url": url, "dest_dir": downloads_dir},
                        trace=ctx.trace,
                    )
                except ToolPermissionError:
                    raise
                except Exception as exc:
                    ctx.trace.log(
                        stage="COLLECT", agent=self.name,
                        action="tool.fetch.failed",
                        error=str(exc),
                        meta={"url": url, "rq_id": rq.rq_id},
                    )
                    continue

                if fetch_result.get("status") != "success" or not fetch_result.get("local_path"):
                    ctx.trace.log(
                        stage="COLLECT", agent=self.name,
                        action="tool.fetch.failed",
                        error=fetch_result.get("error", "unknown"),
                        meta={
                            "url": url,
                            "rq_id": rq.rq_id,
                            "http_status": fetch_result.get("http_status", 0),
                            "detected_type": fetch_result.get("detected_type", ""),
                            "bytes": fetch_result.get("bytes", 0),
                        },
                    )
                    continue

                detected = fetch_result.get("detected_type", "html")
                src_type = _TYPE_MAP.get(detected, SourceType.HTML)

                domain = ""
                if "://" in url:
                    domain = url.split("://", 1)[-1].split("/", 1)[0]

                sources.append(
                    Source(
                        source_id=f"src_{rq.rq_id}_{i}",
                        type=src_type,
                        url=url,
                        domain=domain,
                        title=r.get("title", ""),
                        local_path=fetch_result["local_path"],
                        hash=fetch_result.get("content_hash", ""),
                    )
                )

                if len([s for s in sources if rq.rq_id in s.source_id]) >= 2:
                    break

        return sources

    def _collect_offline(self, ctx: RunContext) -> list[Source]:
        downloads_dir = ctx.run_dir / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)

        samples_pkg = importlib.resources.files("researchops.samples")
        sources: list[Source] = []

        for idx, (sample_name, src_type) in enumerate(
            [("demo.html", SourceType.HTML), ("demo.txt", SourceType.TEXT)]
        ):
            sample_ref = samples_pkg / sample_name
            dest = downloads_dir / sample_name
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
