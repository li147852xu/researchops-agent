# Changelog

## v0.2.0 — 2026-03-05

### Added

- Pluggable Reasoner/LLM layer: `--llm none|openai|anthropic` (default none, no API key required)
- NoneReasoner for rule/template-based operation without external dependencies
- OpenAIReasoner and AnthropicReasoner for LLM-enhanced plan/read/write/qa/verify
- Two-type verification: integrity checks (source/claim validation) + analysis (term frequency, coverage stats)
- Real self-correction: pattern-matching error fixes (ImportError, FileNotFoundError, JSONDecodeError, etc.)
- Report traceability: `report_index.json` maps each sentence to source_ids and claim_ids
- Writer uses `[@source_id]` citation markers for traceable references
- QA traceability check: validates report sentences against claims/sources
- QA multi-level rollback: can roll back to READ, VERIFY, or WRITE based on issue type
- Persistent tool cache (`cache.json` per run) alongside session cache
- Permission denial logging in trace before raising errors
- Enhanced sandbox netblock: patches http.client, requests, httpx in addition to socket/urllib
- Replay `--json` flag for machine-readable output
- Replay `--no-tools` dry-run: shows what would execute without invoking tools
- New eval metrics: unsupported_claim_rate, cache_hit_rate, llm_enabled, estimated_cost_usd
- Enhanced Reader: claim categorization (contribution/method/limitation/finding) + evidence location
- SourceNotes now includes contribution, method, limitations summaries
- New tests: verifier self-fix, replay no-tools, registry permission denial, persistent cache, http.client blocking

### Changed

- Verifier generates two script types per RQ instead of one
- Citation format changed from `[N]` to `[@source_id]` for traceability
- Registry tracks permission denials and cache hits in trace
- Orchestrator clears downstream completed stages on rollback

## v0.1.0 — 2026-03-05

Initial release.

### Added

- Multi-agent pipeline: Planner, Collector, Reader, Verifier, Writer, QA
- Orchestrator with explicit state machine (PLAN -> COLLECT -> READ -> VERIFY -> WRITE -> QA -> DONE)
- CLI commands: `run`, `resume`, `replay`, `eval`
- Subprocess sandbox with timeout, resource limits, network blocking, log capture
- Docker sandbox placeholder
- Tool registry with schema validation, permission governance, session caching
- Built-in tools: web_search, fetch, parse, sandbox_exec, cite
- Checkpoint/resume support via state.json
- Trace auditing via trace.jsonl
- Evaluation metrics: citation_coverage, source_diversity, reproduction_rate, tool_calls, latency_sec, steps
- Offline demo mode with built-in sample data
- pytest test suite covering checkpoint, trace, registry, sandbox
- ruff configuration for linting and formatting
