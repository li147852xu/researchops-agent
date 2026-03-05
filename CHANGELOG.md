# Changelog

## v0.3.3 — 2026-03-05

### Fixed

- `fetch_page`: structured result with `status/http_status/detected_type/content_hash`; magic-number detection (`%PDF-` vs `<html`); min-size threshold (2KB); cross-check rejects PDF URLs that return HTML (captcha/redirect)
- Collector: raw downloads now go to `downloads/` instead of `notes/`; failed sources filtered out of `sources.jsonl`; source `type` derived from `detected_type` instead of hardcoded HTML
- `parse_doc`: quality gate rejects text < 500 chars or code-heavy content (>30% code lines); fake PDFs (no `%PDF-` header) rejected immediately
- Reader: skips sources with empty parsed text; code/template paragraph filter (`function(`, `padding:`, `querySelector`, etc.); marks sources with < 2 valid claims as `low_quality`
- Writer: filters out LLM sentences containing "not a research claim", code keywords, or discard patterns
- QA: new `_code_garbage_detector` (CSS/JS/HTML keywords > 2 lines = hard fail); `_source_availability_check` (minimum valid sources per mode); rollback targets code garbage to COLLECT (allow_net) or READ

### Added

- `scripts/verify_run_integrity.py`: checks `sources.jsonl` integrity (local_path exists, hash non-empty, type matches magic number), notes/ contains only JSON, report has no boilerplate
- `make verify-run RUN=<dir>` Makefile target
- `downloads/` directory in run output structure for raw HTML/PDF files
- Tests: magic-number detection, PDF-URL-HTML rejection, fetch success validation, small response rejection, fake PDF parse rejection, code-heavy rejection, QA code garbage detector, QA source availability check

## v0.3.2 — 2026-03-05

### Fixed

- LLM invocation chain: agents now actually call `ctx.reasoner` (Planner, Writer, Reader, Verifier)
- `ReasonerBase` abstract interface: `trace` parameter added so LLM calls produce `llm.call`/`llm.result` events
- `OpenAICompatReasoner`: base_url normalization (auto-appends `/v1` if missing), DEEPSEEK_API_KEY env fallback
- API key resolution priority: `--llm-api-key` > `OPENAI_API_KEY` > `LLM_API_KEY` > `DEEPSEEK_API_KEY`
- HTML parsing: `<head>/<meta>/<link>` tags now decomposed; added 20+ noise patterns for HTML metadata garbage
- Missing key raises `ValueError` immediately (no silent fallback to NoneReasoner)
- Retry with exponential backoff on 429/5xx HTTP status codes

### Added

- `scripts/verify_llm_path.py` and `make verify-llm` to audit LLM trace presence
- `estimated_tokens` and `estimate_method` fields in `EvalResult` / `eval.json`
- LLM status line printed at run start (provider, base_url, model, key presence)
- `is_llm` property on `ReasonerBase` for agents to detect LLM availability
- Tests: key resolution, base_url normalization, llm.call trace events, no-silent-fallback

## v0.3.0 — 2026-03-05

### Added

- OpenAI-compatible LLM reasoner covering DeepSeek, OpenRouter, vLLM, Ollama, Azure via `--llm openai_compat`
- CLI options: `--llm-provider-label`, `--llm-headers` (JSON), `LLM_API_KEY` env fallback
- LLM call trace events: `llm.call`, `llm.result`, `llm.error` with token/cost estimation
- Plan refinement loop: after READ, checks RQ coverage and rolls back to COLLECT if below threshold
- 3 verification strategies: TERMS (term extraction), COMPARISON (dimension pairs), TREND (temporal refs)
- Reader: `claim_type` (definition/method/result/limitation/trend/comparison) and `polarity` (support/oppose/neutral)
- QA conflict scan: detects opposing claims per RQ, outputs `qa_conflicts.json`
- Writer: "Disagreements and Conflicts" section when conflicts detected
- Writer: "Note" section for evidence-limited offline runs
- Collector: graceful fallback to offline samples on `ToolPermissionError`
- Eval: `conflict_count`, `plan_refinement_count`, `collect_rounds`, `artifacts_count`, `llm_provider_label`
- `scripts/verify_repo.py` verification script + `make verify`
- Tests: md constraint, plan refinement, conflict scan, replay no-tools monkeypatch, openai_compat headers
- Replay `--json` output includes structured per-event fields (step/stage/agent/outcome/latency/cache_hit)

### Changed

- `--llm openai` now maps to `OpenAICompatReasoner` (backward compatible)
- Verifier picks strategy (terms/comparison/trend) based on RQ text keywords
- Verifier self-fix now handles `KeyError` in addition to existing error patterns
- Orchestrator uses rich-optional printing (falls back to plain print)
- StateSnapshot tracks `refinement_count` and `collect_rounds`

## v0.2.0 — 2026-03-05

### Added

- Pluggable Reasoner/LLM layer: `--llm none|openai|anthropic`
- NoneReasoner for rule/template-based operation
- Two-type verification: integrity + analysis with self-correction
- Report traceability: `report_index.json` maps sentences to sources/claims
- Writer: `[@source_id]` citation markers
- QA traceability check and multi-level rollback
- Persistent tool cache, permission denial logging, enhanced sandbox netblock
- Replay `--json` and `--no-tools` flags
- Eval: unsupported_claim_rate, cache_hit_rate, llm_enabled, estimated_cost_usd

## v0.1.0 — 2026-03-05

Initial release.

### Added

- Multi-agent pipeline: Planner, Collector, Reader, Verifier, Writer, QA
- Orchestrator with explicit state machine
- CLI: `run`, `resume`, `replay`, `eval`
- Subprocess sandbox with timeout, resource limits, network blocking
- Tool registry with schema validation, permission governance, caching
- Built-in tools: web_search, fetch, parse, sandbox_exec, cite
- Checkpoint/resume, trace auditing, evaluation metrics
- Offline demo mode with built-in samples
