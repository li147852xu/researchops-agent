from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from researchops.models import EvalResult, Source, StateSnapshot, TraceEvent
from researchops.trace import TraceLogger

if TYPE_CHECKING:
    from researchops.config import RunConfig


def compute_eval(run_dir: Path, *, config: RunConfig | None = None, llm_enabled: bool = False) -> EvalResult:
    report_path = run_dir / "report.md"
    sources = _load_sources(run_dir)
    trace = TraceLogger(run_dir / "trace.jsonl")
    events = trace.read_all()
    state = _load_state(run_dir)

    citation_cov = _citation_coverage(report_path)
    diversity = _source_diversity(sources)
    repro_rate = _reproduction_rate(events)
    tool_calls = _count_tool_calls(events)
    latency = _compute_latency(events)
    steps = _count_steps(events)
    unsupported = _unsupported_claim_rate(run_dir)
    cache_hit = _cache_hit_rate(events)
    conflict_count = _count_conflicts(run_dir)
    artifacts_count = _count_artifacts(run_dir)

    is_llm = llm_enabled or (config is not None and config.llm != "none")
    has_llm_calls = any(e.action in ("llm.call", "llm.result") for e in events)
    is_llm = is_llm or has_llm_calls
    provider_label = config.llm_provider_label if config else ""
    if not provider_label:
        for e in events:
            if e.action == "llm.call" and e.meta.get("provider_label"):
                provider_label = e.meta["provider_label"]
                break

    est_tokens = _estimate_tokens(events)
    est_method = "token_count_from_api" if has_llm_calls else "none"

    papers_rq = _papers_per_rq(sources, run_dir)
    lq_rate = _low_quality_source_rate(events, sources)
    sec_rate = _section_nonempty_rate(report_path)

    incomplete = state.incomplete_sections if state else []
    collect_total = state.collect_rounds if state else 1
    max_rollback = collect_total >= (config.max_collect_rounds if config else 6) if state else False
    src_per_rq = _sources_per_rq(sources, run_dir)
    bucket_cov = _bucket_coverage_rate(run_dir)
    rel_avg = _relevance_avg(run_dir)
    dec_count = _decision_count(run_dir)

    result = EvalResult(
        citation_coverage=citation_cov,
        source_diversity=diversity,
        reproduction_rate=repro_rate,
        tool_calls=tool_calls,
        latency_sec=latency,
        steps=steps,
        unsupported_claim_rate=unsupported,
        cache_hit_rate=cache_hit,
        llm_enabled=is_llm,
        estimated_tokens=est_tokens,
        estimated_cost_usd=0.0,
        estimate_method=est_method,
        conflict_count=conflict_count,
        plan_refinement_count=state.refinement_count if state else 0,
        collect_rounds=state.collect_rounds if state else 1,
        artifacts_count=artifacts_count,
        llm_provider_label=provider_label,
        papers_per_rq=papers_rq,
        low_quality_source_rate=lq_rate,
        section_nonempty_rate=sec_rate,
        incomplete_sections=incomplete,
        collect_rounds_total=collect_total,
        sources_per_rq=src_per_rq,
        max_rollback_used=max_rollback,
        bucket_coverage_rate=bucket_cov,
        relevance_avg=rel_avg,
        decision_count=dec_count,
    )

    eval_path = run_dir / "eval.json"
    eval_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def _citation_coverage(report_path: Path) -> float:
    if not report_path.exists():
        return 0.0
    text = report_path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    if not paragraphs:
        return 0.0
    cited = sum(1 for p in paragraphs if re.search(r"\[@\w+\]", p))
    return round(cited / len(paragraphs), 3)


def _source_diversity(sources: list[Source]) -> dict:
    domains = {s.domain for s in sources if s.domain}
    type_dist: dict[str, int] = {}
    for s in sources:
        type_dist[s.type.value] = type_dist.get(s.type.value, 0) + 1
    return {
        "unique_domains": len(domains),
        "domains": sorted(domains),
        "type_distribution": type_dist,
    }


def _reproduction_rate(events: list[TraceEvent]) -> float:
    execs = [e for e in events if e.action == "sandbox_exec"]
    if not execs:
        return 1.0
    successes = sum(1 for e in events if e.action == "sandbox_success")
    return round(successes / len(execs), 3) if execs else 1.0


def _count_tool_calls(events: list[TraceEvent]) -> int:
    return sum(1 for e in events if e.action == "invoke")


def _compute_latency(events: list[TraceEvent]) -> float:
    for e in events:
        if e.action == "run_complete" and e.duration_ms > 0:
            return round(e.duration_ms / 1000, 2)
    total = sum(e.duration_ms for e in events) / 1000
    return round(total, 2)


def _count_steps(events: list[TraceEvent]) -> int:
    stages_seen: set[str] = set()
    for e in events:
        if e.stage and e.action in ("start", "complete"):
            stages_seen.add(e.stage)
    return len(stages_seen)


def _unsupported_claim_rate(run_dir: Path) -> float:
    index_path = run_dir / "report_index.json"
    if not index_path.exists():
        return 0.0
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        entries = index.get("entries", [])
        if not entries:
            return 0.0
        unsupported = sum(1 for e in entries if not e.get("claim_ids"))
        return round(unsupported / len(entries), 3)
    except Exception:
        return 0.0


def _cache_hit_rate(events: list[TraceEvent]) -> float:
    invokes = sum(1 for e in events if e.action in ("invoke", "cache_hit"))
    hits = sum(1 for e in events if e.action == "cache_hit")
    if invokes == 0:
        return 0.0
    return round(hits / invokes, 3)


def _count_conflicts(run_dir: Path) -> int:
    path = run_dir / "qa_conflicts.json"
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return len(data.get("conflicts", []))
    except Exception:
        return 0


def _estimate_tokens(events: list[TraceEvent]) -> int:
    total = 0
    for e in events:
        if e.action == "llm.result":
            total += e.meta.get("tokens", 0)
    return total


def _count_artifacts(run_dir: Path) -> int:
    art_dir = run_dir / "artifacts"
    if not art_dir.exists():
        return 0
    return sum(1 for _ in art_dir.iterdir())


def _papers_per_rq(sources: list[Source], run_dir: Path) -> float:
    arxiv_sources = [s for s in sources if "arxiv" in s.source_type_detail]
    plan_path = run_dir / "plan.json"
    if not plan_path.exists():
        return 0.0
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        rq_count = len(plan.get("research_questions", []))
        if rq_count == 0:
            return 0.0
        return round(len(arxiv_sources) / rq_count, 2)
    except Exception:
        return 0.0


def _low_quality_source_rate(events: list[TraceEvent], sources: list[Source]) -> float:
    if not sources:
        return 0.0
    lq_count = sum(1 for e in events if e.action == "parse.low_quality")
    return round(lq_count / len(sources), 3)


def _section_nonempty_rate(report_path: Path) -> float:
    if not report_path.exists():
        return 0.0
    text = report_path.read_text(encoding="utf-8")
    sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
    if len(sections) <= 1:
        return 0.0
    content_sections = sections[1:]
    nonempty = 0
    for sec in content_sections:
        paragraphs = [p.strip() for p in sec.split("\n\n") if len(p.strip()) > 30 and not p.strip().startswith("#")]
        if paragraphs:
            nonempty += 1
    return round(nonempty / len(content_sections), 3) if content_sections else 0.0


def _load_sources(run_dir: Path) -> list[Source]:
    path = run_dir / "sources.jsonl"
    if not path.exists():
        return []
    sources = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            sources.append(Source.model_validate(json.loads(line)))
    return sources


def _sources_per_rq(sources: list[Source], run_dir: Path) -> dict[str, int]:
    plan_path = run_dir / "plan.json"
    if not plan_path.exists():
        return {}
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        rqs = plan.get("research_questions", [])
        result: dict[str, int] = {}
        for rq in rqs:
            rq_id = rq.get("rq_id", "")
            count = sum(1 for s in sources if rq_id in s.source_id or rq_id in (s.query_id if hasattr(s, "query_id") else ""))
            result[rq_id] = count
        return result
    except Exception:
        return {}


def _bucket_coverage_rate(run_dir: Path) -> float:
    qa_path = run_dir / "qa_report.json"
    if qa_path.exists():
        try:
            data = json.loads(qa_path.read_text(encoding="utf-8"))
            return data.get("bucket_coverage_rate", 0.0)
        except Exception:
            pass

    emap_path = run_dir / "evidence_map.json"
    if not emap_path.exists():
        return 0.0
    try:
        emap = json.loads(emap_path.read_text(encoding="utf-8"))
        if not emap:
            return 0.0
        with_claims = sum(1 for v in emap.values() if v.get("claims"))
        return round(with_claims / len(emap), 3)
    except Exception:
        return 0.0


def _relevance_avg(run_dir: Path) -> float:
    qa_path = run_dir / "qa_report.json"
    if qa_path.exists():
        try:
            data = json.loads(qa_path.read_text(encoding="utf-8"))
            val = data.get("relevance_avg", 0.0)
            if val > 0:
                return val
        except Exception:
            pass

    notes_dir = run_dir / "notes"
    if not notes_dir.exists():
        return 0.0
    scores: list[float] = []
    for f in notes_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            score = data.get("relevance_score", 0.0)
            if score > 0:
                scores.append(score)
        except Exception:
            continue
    return round(sum(scores) / max(1, len(scores)), 3) if scores else 0.0


def _decision_count(run_dir: Path) -> int:
    path = run_dir / "decisions.jsonl"
    if not path.exists():
        return 0
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        return len([line for line in lines if line.strip()])
    except Exception:
        return 0


def _load_state(run_dir: Path) -> StateSnapshot | None:
    path = run_dir / "state.json"
    if not path.exists():
        return None
    try:
        return StateSnapshot.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None
