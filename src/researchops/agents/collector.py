from __future__ import annotations

import hashlib
import importlib.resources
import json
import logging
import re
import shutil
from pathlib import Path

from researchops.agents.base import AgentBase, RunContext
from researchops.models import AgentResult, PlanOutput, Source, SourceType
from researchops.prompts import COLLECTOR_QUERIES
from researchops.registry.manager import ToolPermissionError
from researchops.utils import get_negative_terms, load_sources

logger = logging.getLogger(__name__)

_TYPE_MAP = {"pdf": SourceType.PDF, "html": SourceType.HTML, "text": SourceType.TEXT, "api": SourceType.API}

_STRATEGY_QUERY_SUFFIXES: dict[int, list[str]] = {
    0: [],
    1: ["overview", "techniques", "applications"],
    2: ["survey", "tutorial", "benchmark", "roadmap", "review"],
}

_LOW_QUALITY_DOMAINS = {
    "pinterest.com", "quora.com", "reddit.com", "linkedin.com",
    "facebook.com", "twitter.com", "instagram.com", "tiktok.com",
}

_JS_HEAVY_DOMAINS = {
    "openreview.net", "www.nature.com", "nature.com",
    "link.springer.com", "www.sciencedirect.com", "sciencedirect.com",
    "ieeexplore.ieee.org", "dl.acm.org", "www.semanticscholar.org",
    "semanticscholar.org", "medium.com", "towardsdatascience.com",
}


class CollectorAgent(AgentBase):
    name = "collector"

    def execute(self, ctx: RunContext) -> AgentResult:
        ctx.trace.log(stage="COLLECT", agent=self.name, action="start")

        plan_path = ctx.run_dir / "plan.json"
        plan = PlanOutput.model_validate(json.loads(plan_path.read_text(encoding="utf-8")))

        existing_sources = load_sources(ctx.run_dir)
        existing_hashes: set[str] = {s.hash for s in existing_sources if s.hash}

        sources: list[Source] = list(existing_sources)
        failures: list[dict] = []
        strategy = self._select_strategy(ctx)
        collect_round = ctx.state.collect_rounds
        strategy_level = ctx.state.collect_strategy_level

        negative_terms = self._get_negative_terms(plan.topic, ctx)
        used_queries = self._load_used_queries(ctx)

        gap_rqs = self._identify_gap_rqs(plan, sources, ctx)
        bucket_queries = self._generate_bucket_queries(plan, ctx)

        if strategy == "demo":
            if not existing_sources:
                sources.extend(self._collect_offline(ctx))
        elif strategy == "arxiv":
            s, f = self._collect_arxiv(
                ctx, plan, gap_rqs, collect_round, strategy_level,
                existing_hashes, negative_terms, used_queries, bucket_queries,
            )
            sources.extend(s)
            failures.extend(f)
        elif strategy == "web":
            try:
                s, f = self._collect_online(
                    ctx, plan, gap_rqs, collect_round, strategy_level,
                    existing_hashes, negative_terms, used_queries,
                )
                sources.extend(s)
                failures.extend(f)
            except ToolPermissionError:
                ctx.trace.log(
                    stage="COLLECT", agent=self.name, action="tool.denied_fallback",
                    output_summary="Net tools denied, falling back to offline samples",
                )
                if not existing_sources:
                    sources.extend(self._collect_offline(ctx))
        elif strategy == "hybrid":
            s, f = self._collect_arxiv(
                ctx, plan, gap_rqs, collect_round, strategy_level,
                existing_hashes, negative_terms, used_queries, bucket_queries,
            )
            sources.extend(s)
            failures.extend(f)
            if len(sources) < ctx.config.max_total_sources:
                try:
                    s2, f2 = self._collect_online(
                        ctx, plan, gap_rqs, collect_round, strategy_level,
                        existing_hashes, negative_terms, used_queries,
                    )
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
        sources = deduped[:ctx.config.max_total_sources]

        self._check_diversity_constraints(sources, ctx, plan)

        sources_path = ctx.run_dir / "sources.jsonl"
        with sources_path.open("w", encoding="utf-8") as f:
            for src in sources:
                f.write(src.model_dump_json() + "\n")

        if failures:
            failures_path = ctx.run_dir / "failures.json"
            failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")

        self._save_used_queries(ctx, used_queries)

        new_count = len(sources) - len(existing_sources)
        ctx.trace.log(
            stage="COLLECT", agent=self.name, action="complete",
            output_summary=f"{len(sources)} total sources ({new_count} new), {len(failures)} failed",
            meta={
                "strategy": strategy, "collect_round": collect_round,
                "strategy_level": strategy_level, "gap_rqs": [r.rq_id for r in gap_rqs],
                "total_sources": len(sources), "new_sources": new_count,
                "negative_terms": negative_terms[:5],
                "bucket_queries_used": len(bucket_queries),
            },
        )
        return AgentResult(success=True, message=f"Collected {len(sources)} sources ({strategy}, round {collect_round})")

    def _select_strategy(self, ctx: RunContext) -> str:
        if not ctx.config.allow_net:
            return "demo"
        return ctx.config.sources.value

    def _identify_gap_rqs(
        self, plan: PlanOutput, sources: list[Source], ctx: RunContext,
    ) -> list:
        cov_vec = ctx.state.coverage_vector
        if not cov_vec:
            return list(plan.research_questions[:ctx.config.max_collect])

        gap_rqs = []
        for rq in plan.research_questions:
            claims_count = cov_vec.get(rq.rq_id, 0)
            rq_sources = [s for s in sources if rq.rq_id in s.source_id or rq.rq_id in s.query_id]
            if claims_count < ctx.config.min_claims_per_rq or len(rq_sources) < ctx.config.target_sources_per_rq:
                gap_rqs.append(rq)

        return gap_rqs if gap_rqs else list(plan.research_questions[:ctx.config.max_collect])

    def _get_negative_terms(self, topic: str, ctx: RunContext) -> list[str]:
        return get_negative_terms(topic)

    def _generate_bucket_queries(self, plan: PlanOutput, ctx: RunContext) -> list[str]:
        checklist = plan.coverage_checklist
        if not checklist:
            return []
        queries: list[str] = []
        for bucket in checklist:
            bname = bucket.get("bucket_name", "")
            if bname:
                queries.append(f"{plan.topic} {bname}")
        return queries

    def _load_used_queries(self, ctx: RunContext) -> set[str]:
        cache_path = ctx.run_dir / "cache_used_queries.json"
        if cache_path.exists():
            try:
                return set(json.loads(cache_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        return set()

    def _save_used_queries(self, ctx: RunContext, used: set[str]) -> None:
        cache_path = ctx.run_dir / "cache_used_queries.json"
        cache_path.write_text(json.dumps(sorted(used)), encoding="utf-8")

    def _apply_relevance_filter(
        self, paper: dict, topic: str, checklist: list[dict], threshold: float,
    ) -> float:
        anchor_terms = set(re.findall(r"\w{3,}", topic.lower()))
        for bucket in checklist:
            desc = bucket.get("description", "") + " " + bucket.get("bucket_name", "")
            anchor_terms.update(re.findall(r"\w{3,}", desc.lower()))

        text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
        text_words = set(re.findall(r"\w{3,}", text))
        hits = len(anchor_terms & text_words)
        total = max(1, len(anchor_terms))
        return hits / total

    def _enforce_query_novelty(
        self, queries: list[str], used_queries: set[str], strategy_level: int,
    ) -> list[str]:
        if strategy_level == 0 or not used_queries:
            used_queries.update(queries)
            return queries

        novel = [q for q in queries if q not in used_queries]
        reused = [q for q in queries if q in used_queries]
        min_novel = max(1, len(queries) // 2)
        if len(novel) < min_novel:
            result = novel + reused[: len(queries) - len(novel)]
        else:
            result = novel + reused[: max(0, len(queries) - len(novel))]
        used_queries.update(result)
        return result

    def _generate_queries(
        self, rq_text: str, topic: str, strategy_level: int, ctx: RunContext,
        negative_terms: list[str] | None = None,
        used_queries: set[str] | None = None,
        bucket_queries: list[str] | None = None,
    ) -> list[str]:
        base_query = rq_text
        queries = [base_query]

        if bucket_queries:
            queries.extend(bucket_queries[:3])

        suffixes = _STRATEGY_QUERY_SUFFIXES.get(strategy_level, [])
        for suffix in suffixes:
            queries.append(f"{topic} {suffix}")

        if strategy_level >= 1:
            queries.append(f"{rq_text} recent advances")
            queries.append(f"{topic} state of the art")

        if strategy_level >= 2:
            queries.append(f"{topic} survey paper")
            queries.append(f"{topic} comprehensive review")
            queries.append(f"{rq_text} challenges and future directions")

        if ctx.reasoner.is_llm and strategy_level >= 1:
            try:
                neg_hint = ""
                if negative_terms:
                    neg_hint = f"Avoid topics: {', '.join(negative_terms[:5])}.\n"
                sys_msg, user_msg = COLLECTOR_QUERIES.render(
                    rq_text=rq_text, topic=topic, neg_hint=neg_hint,
                )
                raw = ctx.reasoner.complete_text(user_msg, context=sys_msg, trace=ctx.trace)
                for line in raw.strip().splitlines():
                    line = line.strip().lstrip("0123456789.-) ")
                    if 10 < len(line) < 200:
                        queries.append(line)
            except Exception:
                pass

        if used_queries is not None:
            queries = self._enforce_query_novelty(queries, used_queries, strategy_level)

        return queries

    def _collect_arxiv(
        self, ctx: RunContext, plan: PlanOutput, gap_rqs: list,
        collect_round: int, strategy_level: int, existing_hashes: set[str],
        negative_terms: list[str] | None = None,
        used_queries: set[str] | None = None,
        bucket_queries: list[str] | None = None,
    ) -> tuple[list[Source], list[dict]]:
        sources: list[Source] = []
        failures: list[dict] = []
        downloads_dir = str(ctx.run_dir / "downloads")
        checklist = plan.coverage_checklist
        rel_threshold = ctx.config.relevance_threshold

        for rq in gap_rqs:
            if self._total_reached(sources, ctx):
                break

            queries = self._generate_queries(
                rq.text, plan.topic, strategy_level, ctx,
                negative_terms=negative_terms,
                used_queries=used_queries,
                bucket_queries=bucket_queries,
            )

            for q_idx, query in enumerate(queries):
                if self._rq_target_met(sources, rq.rq_id, ctx):
                    break

                query_id = f"q{collect_round}_{rq.rq_id}_{q_idx}"
                if ctx.config.mode.value == "deep":
                    max_results = 8 if strategy_level == 0 else 12
                else:
                    max_results = 5 if strategy_level == 0 else 8

                try:
                    results = ctx.registry.invoke(
                        "arxiv_search",
                        {"query": query, "max_results": max_results},
                        trace=ctx.trace,
                    )
                except ToolPermissionError:
                    raise
                except Exception as exc:
                    failures.append({"rq_id": rq.rq_id, "tool": "arxiv_search", "error": str(exc), "query": query})
                    continue

                if not results or (len(results) == 1 and "error" in results[0]):
                    failures.append({"rq_id": rq.rq_id, "tool": "arxiv_search", "error": str(results), "query": query})
                    continue

                papers_per_query = 4 if strategy_level >= 1 else 3
                for i, paper in enumerate(results[:papers_per_query]):
                    if isinstance(paper, dict) and "error" in paper:
                        continue

                    if negative_terms and self._has_negative_terms(paper, negative_terms):
                        continue

                    if checklist:
                        score = self._apply_relevance_filter(paper, plan.topic, checklist, rel_threshold)
                        if score < rel_threshold * 0.5:
                            continue

                    arxiv_id = paper.get("arxiv_id", f"unknown_{i}")
                    source_id = f"src_arxiv_{rq.rq_id}_r{collect_round}_{q_idx}_{i}"

                    abstract_text = paper.get("abstract", "")
                    abstract_hash = hashlib.sha256(abstract_text.encode()).hexdigest()

                    if abstract_hash in existing_hashes:
                        continue

                    abstract_path = Path(downloads_dir) / f"{source_id}_abstract.txt"
                    abstract_path.parent.mkdir(parents=True, exist_ok=True)
                    abstract_path.write_text(abstract_text, encoding="utf-8")

                    sources.append(Source(
                        source_id=source_id,
                        type=SourceType.API,
                        url=f"https://arxiv.org/abs/{arxiv_id}",
                        domain="arxiv.org",
                        title=paper.get("title", ""),
                        local_path=str(abstract_path),
                        hash=abstract_hash,
                        source_type_detail="arxiv_meta",
                        collect_round=collect_round,
                        query_id=query_id,
                    ))
                    existing_hashes.add(abstract_hash)

                    if ctx.config.mode.value == "deep" and paper.get("pdf_url"):
                        try:
                            dl_result = ctx.registry.invoke(
                                "arxiv_download_pdf",
                                {"pdf_url": paper["pdf_url"], "dest_dir": downloads_dir},
                                trace=ctx.trace,
                            )
                            if dl_result.get("status") == "success":
                                pdf_hash = dl_result.get("content_hash", "")
                                if pdf_hash and pdf_hash in existing_hashes:
                                    continue
                                pdf_source_id = f"src_arxiv_pdf_{rq.rq_id}_r{collect_round}_{q_idx}_{i}"
                                sources.append(Source(
                                    source_id=pdf_source_id,
                                    type=SourceType.PDF,
                                    url=paper["pdf_url"],
                                    domain="arxiv.org",
                                    title=paper.get("title", "") + " [PDF]",
                                    local_path=dl_result["local_path"],
                                    hash=pdf_hash,
                                    source_type_detail="arxiv_pdf",
                                    collect_round=collect_round,
                                    query_id=query_id,
                                ))
                                if pdf_hash:
                                    existing_hashes.add(pdf_hash)
                        except Exception as exc:
                            failures.append({
                                "rq_id": rq.rq_id, "tool": "arxiv_download_pdf",
                                "url": paper.get("pdf_url", ""), "error": str(exc),
                            })

                    if self._rq_target_met(sources, rq.rq_id, ctx):
                        break

        return sources, failures

    def _has_negative_terms(self, paper: dict, negative_terms: list[str]) -> bool:
        text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
        return any(term.lower() in text for term in negative_terms)

    def _collect_online(
        self, ctx: RunContext, plan: PlanOutput, gap_rqs: list,
        collect_round: int, strategy_level: int, existing_hashes: set[str],
        negative_terms: list[str] | None = None,
        used_queries: set[str] | None = None,
    ) -> tuple[list[Source], list[dict]]:
        sources: list[Source] = []
        failures: list[dict] = []
        downloads_dir = str(ctx.run_dir / "downloads")

        for rq in gap_rqs:
            if self._total_reached(sources, ctx):
                break

            queries = self._generate_queries(
                rq.text, plan.topic, strategy_level, ctx,
                negative_terms=negative_terms,
                used_queries=used_queries,
            )

            max_web_queries = 4 if ctx.config.mode.value == "deep" else 2
            for q_idx, query in enumerate(queries[:max_web_queries]):
                if self._rq_target_met(sources, rq.rq_id, ctx):
                    break

                query_id = f"wq{collect_round}_{rq.rq_id}_{q_idx}"

                try:
                    results = ctx.registry.invoke(
                        "web_search",
                        {"query": query, "max_results": 5},
                        trace=ctx.trace,
                    )
                except ToolPermissionError:
                    raise
                except Exception:
                    results = []

                for i, r in enumerate(results[:4]):
                    url = r.get("url", "")
                    if not url:
                        continue

                    domain = ""
                    if "://" in url:
                        domain = url.split("://", 1)[-1].split("/", 1)[0]
                    if domain in _LOW_QUALITY_DOMAINS or domain in _JS_HEAVY_DOMAINS:
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
                        })
                        continue

                    content_hash = fetch_result.get("content_hash", "")
                    if content_hash and content_hash in existing_hashes:
                        continue

                    detected = fetch_result.get("detected_type", "html")
                    src_type = _TYPE_MAP.get(detected, SourceType.HTML)

                    sources.append(Source(
                        source_id=f"src_web_{rq.rq_id}_r{collect_round}_{q_idx}_{i}",
                        type=src_type,
                        url=url,
                        domain=domain,
                        title=r.get("title", ""),
                        local_path=fetch_result["local_path"],
                        hash=content_hash,
                        source_type_detail=f"web_{detected}",
                        collect_round=collect_round,
                        query_id=query_id,
                    ))
                    if content_hash:
                        existing_hashes.add(content_hash)

                    if self._rq_target_met(sources, rq.rq_id, ctx):
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
                collect_round=1,
                query_id="demo",
            ))

        return sources

    def _rq_target_met(self, new_sources: list[Source], rq_id: str, ctx: RunContext) -> bool:
        rq_sources = [s for s in new_sources if rq_id in s.source_id or rq_id in s.query_id]
        return len(rq_sources) >= ctx.config.target_sources_per_rq

    def _total_reached(self, new_sources: list[Source], ctx: RunContext) -> bool:
        return len(new_sources) >= ctx.config.max_total_sources


    def _check_diversity_constraints(
        self, sources: list[Source], ctx: RunContext, plan: PlanOutput,
    ) -> None:
        arxiv_count = sum(1 for s in sources if "arxiv" in s.source_type_detail)
        total = len(sources)
        if total == 0:
            return

        arxiv_ratio = arxiv_count / total
        is_deep_hybrid = (
            ctx.config.mode.value == "deep"
            and ctx.config.sources.value == "hybrid"
        )

        ctx.trace.log(
            stage="COLLECT", agent=self.name, action="diversity_check",
            meta={
                "arxiv_count": arxiv_count,
                "total": total,
                "arxiv_ratio": round(arxiv_ratio, 2),
                "is_deep_hybrid": is_deep_hybrid,
                "meets_60pct": arxiv_ratio >= 0.6 or not is_deep_hybrid,
            },
        )

