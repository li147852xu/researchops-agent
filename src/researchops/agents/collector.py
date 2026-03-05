from __future__ import annotations

import hashlib
import importlib.resources
import json
import shutil
from pathlib import Path

from researchops.agents.base import AgentBase, RunContext
from researchops.models import AgentResult, PlanOutput, Source, SourceType
from researchops.registry.manager import ToolPermissionError

_TYPE_MAP = {"pdf": SourceType.PDF, "html": SourceType.HTML, "text": SourceType.TEXT, "api": SourceType.API}


class CollectorAgent(AgentBase):
    name = "collector"

    def execute(self, ctx: RunContext) -> AgentResult:
        ctx.trace.log(stage="COLLECT", agent=self.name, action="start")

        plan_path = ctx.run_dir / "plan.json"
        plan = PlanOutput.model_validate(json.loads(plan_path.read_text(encoding="utf-8")))

        sources: list[Source] = []
        failures: list[dict] = []
        strategy = self._select_strategy(ctx)

        if strategy == "demo":
            sources.extend(self._collect_offline(ctx))
        elif strategy == "arxiv":
            s, f = self._collect_arxiv(ctx, plan)
            sources.extend(s)
            failures.extend(f)
        elif strategy == "web":
            try:
                s, f = self._collect_online(ctx, plan)
                sources.extend(s)
                failures.extend(f)
            except ToolPermissionError:
                ctx.trace.log(
                    stage="COLLECT", agent=self.name, action="tool.denied_fallback",
                    output_summary="Net tools denied, falling back to offline samples",
                )
                sources.extend(self._collect_offline(ctx))
        elif strategy == "hybrid":
            s, f = self._collect_arxiv(ctx, plan)
            sources.extend(s)
            failures.extend(f)
            if len(sources) < ctx.config.max_collect:
                try:
                    s2, f2 = self._collect_online(ctx, plan)
                    sources.extend(s2)
                    failures.extend(f2)
                except ToolPermissionError:
                    pass

        seen_hashes: set[str] = set()
        deduped: list[Source] = []
        for src in sources:
            if src.hash and src.hash in seen_hashes:
                continue
            if src.hash:
                seen_hashes.add(src.hash)
            deduped.append(src)
        sources = deduped

        sources_path = ctx.run_dir / "sources.jsonl"
        with sources_path.open("w", encoding="utf-8") as f:
            for src in sources:
                f.write(src.model_dump_json() + "\n")

        if failures:
            failures_path = ctx.run_dir / "failures.json"
            failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")

        ctx.trace.log(
            stage="COLLECT", agent=self.name, action="complete",
            output_summary=f"{len(sources)} sources collected, {len(failures)} failed",
            meta={"strategy": strategy, "failures_count": len(failures)},
        )
        return AgentResult(success=True, message=f"Collected {len(sources)} sources ({strategy})")

    def _select_strategy(self, ctx: RunContext) -> str:
        if not ctx.config.allow_net:
            return "demo"
        return ctx.config.sources.value

    def _collect_arxiv(self, ctx: RunContext, plan: PlanOutput) -> tuple[list[Source], list[dict]]:
        sources: list[Source] = []
        failures: list[dict] = []
        downloads_dir = str(ctx.run_dir / "downloads")

        for rq in plan.research_questions[:ctx.config.max_collect]:
            try:
                results = ctx.registry.invoke(
                    "arxiv_search",
                    {"query": rq.text, "max_results": 5},
                    trace=ctx.trace,
                )
            except ToolPermissionError:
                raise
            except Exception as exc:
                failures.append({"rq_id": rq.rq_id, "tool": "arxiv_search", "error": str(exc)})
                continue

            if not results or (len(results) == 1 and "error" in results[0]):
                failures.append({"rq_id": rq.rq_id, "tool": "arxiv_search", "error": str(results)})
                continue

            for i, paper in enumerate(results[:3]):
                if isinstance(paper, dict) and "error" in paper:
                    continue

                arxiv_id = paper.get("arxiv_id", f"unknown_{i}")
                source_id = f"src_arxiv_{rq.rq_id}_{i}"

                abstract_text = paper.get("abstract", "")
                abstract_path = Path(downloads_dir) / f"{source_id}_abstract.txt"
                abstract_path.parent.mkdir(parents=True, exist_ok=True)
                abstract_path.write_text(abstract_text, encoding="utf-8")
                abstract_hash = hashlib.sha256(abstract_text.encode()).hexdigest()

                sources.append(Source(
                    source_id=source_id,
                    type=SourceType.API,
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    domain="arxiv.org",
                    title=paper.get("title", ""),
                    local_path=str(abstract_path),
                    hash=abstract_hash,
                    source_type_detail="arxiv_meta",
                ))

                if ctx.config.mode.value == "deep" and paper.get("pdf_url"):
                    try:
                        dl_result = ctx.registry.invoke(
                            "arxiv_download_pdf",
                            {"pdf_url": paper["pdf_url"], "dest_dir": downloads_dir},
                            trace=ctx.trace,
                        )
                        if dl_result.get("status") == "success":
                            pdf_source_id = f"src_arxiv_pdf_{rq.rq_id}_{i}"
                            sources.append(Source(
                                source_id=pdf_source_id,
                                type=SourceType.PDF,
                                url=paper["pdf_url"],
                                domain="arxiv.org",
                                title=paper.get("title", "") + " [PDF]",
                                local_path=dl_result["local_path"],
                                hash=dl_result.get("content_hash", ""),
                                source_type_detail="arxiv_pdf",
                            ))
                        else:
                            failures.append({
                                "rq_id": rq.rq_id, "tool": "arxiv_download_pdf",
                                "url": paper["pdf_url"], "error": dl_result.get("error", ""),
                            })
                    except Exception as exc:
                        failures.append({
                            "rq_id": rq.rq_id, "tool": "arxiv_download_pdf",
                            "url": paper.get("pdf_url", ""), "error": str(exc),
                        })

                if self._rq_coverage_met(sources, rq.rq_id, ctx):
                    break

        return sources, failures

    def _collect_online(self, ctx: RunContext, plan: PlanOutput) -> tuple[list[Source], list[dict]]:
        sources: list[Source] = []
        failures: list[dict] = []
        downloads_dir = str(ctx.run_dir / "downloads")

        for rq in plan.research_questions[:ctx.config.max_collect]:
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
                    failures.append({
                        "rq_id": rq.rq_id, "tool": "fetch", "url": url, "error": str(exc),
                    })
                    continue

                if fetch_result.get("status") != "success" or not fetch_result.get("local_path"):
                    failures.append({
                        "rq_id": rq.rq_id, "tool": "fetch", "url": url,
                        "error": fetch_result.get("error", "unknown"),
                        "http_status": fetch_result.get("http_status", 0),
                    })
                    continue

                detected = fetch_result.get("detected_type", "html")
                src_type = _TYPE_MAP.get(detected, SourceType.HTML)

                domain = ""
                if "://" in url:
                    domain = url.split("://", 1)[-1].split("/", 1)[0]

                sources.append(Source(
                    source_id=f"src_{rq.rq_id}_{i}",
                    type=src_type,
                    url=url,
                    domain=domain,
                    title=r.get("title", ""),
                    local_path=fetch_result["local_path"],
                    hash=fetch_result.get("content_hash", ""),
                    source_type_detail=f"web_{detected}",
                ))

                if self._rq_coverage_met(sources, rq.rq_id, ctx):
                    break

        return sources, failures

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

            sources.append(Source(
                source_id=f"src_builtin_{idx}",
                type=src_type,
                url=f"builtin://{sample_name}",
                domain="builtin",
                title=f"Demo {sample_name}",
                local_path=str(dest),
                hash=content_hash,
                source_type_detail="demo",
            ))

        return sources

    def _rq_coverage_met(self, sources: list[Source], rq_id: str, ctx: RunContext) -> bool:
        rq_sources = [s for s in sources if rq_id in s.source_id]
        return len(rq_sources) >= ctx.config.min_sources_per_rq
