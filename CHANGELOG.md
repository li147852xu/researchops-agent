# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [1.1.0] — 2026-03-05

Strategy-driven research with Supervisor policy layer, topic-adaptive coverage buckets, relevance gating, and evidence packing.

### Added

- **Supervisor / Policy layer** — `Supervisor` class acts as a policy advisor before rollback decisions, producing `Decision` records with `reason_codes`, `action_plan`, `confidence`; written to `decisions.jsonl` for full auditability
- **Coverage Checklist** — Planner generates topic-adaptive buckets (e.g. for "deep learning": architectures, optimization, generalization, robustness, scaling, applications); LLM mode produces domain-specific buckets, rule-based fallback uses keyword templates
- **Relevance Gate** — Reader computes per-source `relevance_score` (0–1) using anchor term overlap; sources below threshold flagged `low_relevance`; QA enforces `relevance_avg` as a hard gate
- **Bucket Coverage Gate** — QA computes `bucket_coverage_rate`; below-threshold triggers rollback to COLLECT with bucket-specific queries and suggested categories
- **Evidence Packing** — Writer generates `evidence_map.json` before writing, mapping sections to matched buckets, claim lists, source counts, and diversity status
- **Bucket-driven Collection** — Collector generates queries per bucket; applies negative terms to filter off-topic results; relevance pre-filter; query novelty enforcement (>=50% new queries on strategy bumps)
- **Decision Records** — Every rollback logged to `decisions.jsonl` with structured reason codes (`coverage_gap`, `relevance_drift`, `source_quality_low`, `domination`, `unsupported_high`, `conflict_found`, `no_progress_loop`, `bucket_incomplete`)
- New eval metrics: `bucket_coverage_rate`, `relevance_avg`, `decision_count`
- Verification script: `scripts/verify_research_quality.py`
- Tests: `tests/test_v11_features.py` (7 test cases)
- Makefile target: `make verify-quality`

### Changed

- `PlanOutput`: added `coverage_checklist` field
- `SourceNotes`: added `relevance_score` and `bucket_hits` fields
- `StateSnapshot`: added `write_rounds` and `bucket_coverage` fields
- `EvalResult`: added `bucket_coverage_rate`, `relevance_avg`, `decision_count` fields
- `RunConfig`: added `target_claims_per_rq`, `target_sources_per_bucket`, `relevance_threshold`, `bucket_coverage_threshold` properties
- Orchestrator consults Supervisor before executing rollbacks
- Collector uses bucket-driven query generation with negative term filtering
- Reader performs anchor-term relevance scoring and bucket-hit computation
- Writer generates `evidence_map.json`; cleans RQ headings
- QA enforces bucket coverage and relevance gates

---

## [1.0.1] — 2026-03-05

Fixes infinite rollback loops. Introduces adaptive collection, LLM reading cards, explanatory writing, and progress detection.

### Fixed

- **Infinite rollback loop** — added `max_collect_rounds` (deep=6, fast=3) with hard cap; `progress_detector` tracks `sources_hash`/`claims_hash` between rounds; 2 consecutive no-progress rounds trigger strategy upgrade or degrade-completion
- **Identical sources/claims on rollback** — Collector uses gap-driven query expansion with 3 strategy levels; LLM-powered query generation when available; incremental append instead of overwrite; duplicate hash rejection
- **Content weakness** — Writer produces structured sections (overview + bullets + trends) instead of claim concatenation

### Added

- **Adaptive collection** — gap RQ identification via `coverage_vector`; targeted queries per gap; diversity constraints (arXiv >=60%, >=3 sources/RQ, single-source cap 35%); `collect_round`/`query_id` tracking
- **LLM reading cards** — structured claim extraction with `claim_type`, `polarity`, `evidence_span`; quality gate rejects sources <200 chars
- **Explanatory writing** — LLM-synthesized overview paragraphs with mandatory citation markers; evidence gap sections when at max rounds
- **QA rollback policy** — evidence gaps and source domination route to COLLECT; `next_actions` with suggested queries; progress-aware retry
- **Progress detection** — `sources_hash`, `claims_hash`, `coverage_vector`, `rollback_history`, `no_progress_streak`, `collect_strategy_level`
- **Degrade-completion** — pipeline finishes with explicit "Evidence Gap" sections when max rounds reached
- New eval metrics: `incomplete_sections`, `collect_rounds_total`, `sources_per_rq`, `max_rollback_used`
- Verification: `scripts/verify_no_infinite_rollback.py`
- Tests: `tests/test_rollback_loop.py`
- Makefile target: `make verify-loop`

### Changed

- `Source` model: added `collect_round` and `query_id` fields
- `RunConfig`: added `max_collect_rounds`, `target_sources_per_rq`, `max_total_sources`
- Collector: target-driven stop replaces fixed count
- Writer: `_SOURCE_CAP_RATIO` tightened to 0.35; `_MAX_CLAIMS_PER_SECTION` raised to 8
- QA: `_SOURCE_CAP` tightened to 0.35; added domination and evidence gap detection

---

## [1.0.0] — 2026-03-05

Production-grade release. arXiv-first ingestion, evidence-first writing, run-scoped BM25 retrieval, data quality governance, evalset batch runner, and full LLM ecosystem.

### Added

- **arXiv-first ingestion** — `arxiv_search` (Atom API) and `arxiv_download_pdf` tools; `--sources {demo,arxiv,web,hybrid}`
- **Run-scoped retrieval** — BM25 index (rank-bm25) built after READ; Writer, QA, Verifier consume ranked claims
- **Evidence-first writing** — every report sentence requires citation; insufficient evidence triggers rollback
- **Collector strategy engine** — dispatches to demo/arxiv/web/hybrid; adaptive stop per-RQ; `failures.json` for failed sources
- **Enhanced reading cards** — `bibliographic` (paper_id/authors/year), `quality` (readability, noise flags, extraction method), `source_type_detail`
- **QA enhancements** — source quality check, `qa_report.json`, tighter unsupported claim gates (deep<=5%, fast<=20%)
- **Evalset** — `evalset/topics.jsonl` (30 topics EN/CN), `scripts/run_evalset.py`, `aggregate_metrics.json`
- **CLI** — `verify-run`, `verify-repo`; `--retrieval`, `--embedder` options
- **HTML parsing** — trafilatura as optional primary parser (fallback to bs4)
- Eval metrics: `papers_per_rq`, `low_quality_source_rate`, `section_nonempty_rate`

### Changed

- `rank-bm25>=0.2` added as core dependency; `trafilatura>=1.6` as optional
- Collector: strategy-based dispatch replaces simple online/offline branching
- Orchestrator: builds retrieval index after READ
- Writer: uses retrieval for claim lookup; evidence-first hard constraint

---

## [0.3.3] — 2026-03-05

### Fixed

- `fetch_page`: structured result with magic-number detection, min-size threshold, cross-check rejects PDF URLs returning HTML
- Collector: raw downloads go to `downloads/`; failed sources filtered; source `type` from `detected_type`
- `parse_doc`: quality gate rejects text <500 chars or code-heavy content; fake PDFs rejected
- Reader: skips empty parsed text; code/template filter; marks sources <2 valid claims as `low_quality`
- Writer: filters LLM sentences with discard patterns
- QA: `_code_garbage_detector`, `_source_availability_check`, rollback targets

### Added

- `scripts/verify_run_integrity.py` and `make verify-run`
- `downloads/` directory in run output

---

## [0.3.2] — 2026-03-05

### Fixed

- LLM invocation chain: agents now call `ctx.reasoner`
- `ReasonerBase`: `trace` parameter for `llm.call`/`llm.result` events
- `OpenAICompatReasoner`: base_url normalization, DEEPSEEK_API_KEY fallback
- API key resolution: `--llm-api-key` > `OPENAI_API_KEY` > `LLM_API_KEY` > `DEEPSEEK_API_KEY`
- HTML parsing: `<head>/<meta>/<link>` decomposition; 20+ noise patterns
- Missing key raises `ValueError` immediately
- Retry with exponential backoff on 429/5xx

### Added

- `scripts/verify_llm_path.py` and `make verify-llm`
- `estimated_tokens` and `estimate_method` in eval
- LLM status line at run start
- `is_llm` property on `ReasonerBase`

---

## [0.3.0] — 2026-03-05

### Added

- OpenAI-compatible LLM reasoner (DeepSeek, OpenRouter, vLLM, Ollama, Azure)
- Plan refinement loop: checks RQ coverage after READ, rolls back if below threshold
- 3 verification strategies: TERMS, COMPARISON, TREND
- Reader: `claim_type` and `polarity` classification
- QA: conflict scan with `qa_conflicts.json`
- Writer: "Disagreements and Conflicts" section; evidence-limited note for offline runs
- Collector: graceful fallback on `ToolPermissionError`
- Eval: `conflict_count`, `plan_refinement_count`, `collect_rounds`, `artifacts_count`, `llm_provider_label`
- `scripts/verify_repo.py` and `make verify`
- Replay `--json` with structured per-event fields

### Changed

- `--llm openai` maps to `OpenAICompatReasoner` (backward compatible)
- Verifier picks strategy based on RQ keywords
- Orchestrator uses rich-optional printing

---

## [0.2.0] — 2026-03-05

### Added

- Pluggable Reasoner/LLM layer: `--llm none|openai|anthropic`
- Two-type verification: integrity + analysis with self-correction
- Report traceability: `report_index.json`
- Writer: `[@source_id]` citation markers
- QA traceability check and multi-level rollback
- Persistent tool cache, permission denial logging, enhanced sandbox netblock
- Replay `--json` and `--no-tools` flags
- Eval: `unsupported_claim_rate`, `cache_hit_rate`, `llm_enabled`, `estimated_cost_usd`

---

## [0.1.0] — 2026-03-05

Initial release.

### Added

- Multi-agent pipeline: Planner, Collector, Reader, Verifier, Writer, QA
- Orchestrator with explicit state machine (PLAN/COLLECT/READ/VERIFY/WRITE/QA/DONE)
- CLI: `run`, `resume`, `replay`, `eval`
- Subprocess sandbox with timeout, resource limits, network blocking
- Tool registry with schema validation, permission governance, caching
- Built-in tools: `web_search`, `fetch_page`, `parse_doc`, `sandbox_exec`, `cite`
- Checkpoint/resume, trace auditing, evaluation metrics
- Offline demo mode with built-in samples
