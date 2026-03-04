# Changelog

## v0.1.0 — 2026-03-05

Initial release.

### Added

- Multi-agent pipeline: Planner, Collector, Reader, Verifier, Writer, QA
- Orchestrator with explicit state machine (PLAN → COLLECT → READ → VERIFY → WRITE → QA → DONE)
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
