# ResearchOps Agent

A multi-agent research orchestration harness that decomposes complex research topics into structured pipelines: **plan -> collect -> read -> verify -> write -> qa -> eval**. Each stage is handled by a specialized agent with full trace auditing, checkpoint/resume/replay, sandboxed code execution, governed tool registry, and pluggable LLM reasoning.

## Features

- **Multi-agent pipeline** with explicit state machine (PLAN -> COLLECT -> READ -> VERIFY -> WRITE -> QA -> DONE)
- **Pluggable LLM reasoning** — runs fully offline with `--llm none` (default), or enhanced with `--llm openai` / `--llm anthropic`
- **Checkpoint / Resume / Replay** — interrupt and continue from any stage; replay traces with `--no-tools` dry-run or `--json` machine-readable output
- **Sandboxed execution** — subprocess sandbox with timeout, resource limits, network blocking (socket + http.client + urllib + requests + httpx), and log capture
- **Self-correcting verification** — two verification types (integrity + analysis) with automatic error detection and script repair (2-3 retries)
- **Report traceability** — every sentence traced to source/claim via `report_index.json`; QA validates coverage
- **Tool & Skill Registry** — schema-validated tool definitions with permission governance, risk levels, session + persistent caching
- **Evaluation scoring** — citation coverage, source diversity, reproduction rate, unsupported claim rate, cache hit rate, tool calls, latency

## Installation

```bash
git clone <repo-url> && cd researchops-agent
pip install -e .

# With dev dependencies
pip install -e ".[dev]"

# With LLM support (optional)
pip install -e ".[llm-openai]"
```

Requires **Python 3.11+**.

## Quick Start

```bash
# Offline demo (no network, no API key needed)
researchops run "demo topic" --mode fast --allow-net false

# With network search
researchops run "large language model safety" --mode deep --allow-net true

# With LLM-enhanced reasoning
researchops run "quantum computing" --mode deep --llm openai --llm-api-key $OPENAI_API_KEY
```

## CLI Reference

```bash
# Full run options
researchops run "<topic>" \
  --mode {fast,deep} \
  --checkpoint <path> \
  --budget <float> \
  --max-steps <int> \
  --allow-net {true,false} \
  --net-allowlist "arxiv.org,scholar.google.com" \
  --sandbox {subprocess,docker} \
  --llm {none,openai,anthropic} \
  --llm-model "<model>" \
  --llm-base-url "<url>" \
  --llm-api-key "<key>" \
  --seed <int>

# Resume interrupted run
researchops resume runs/<run_id>

# Replay trace
researchops replay runs/<run_id> --from-step 5
researchops replay runs/<run_id> --no-tools          # dry-run
researchops replay runs/<run_id> --json               # machine-readable

# Recompute evaluation
researchops eval runs/<run_id>
```

## Run Artifact Structure

```
runs/<run_id>/
  plan.json             Research questions, outline, thresholds
  sources.jsonl         One source per line (id, type, url, domain, hash)
  report.md             Synthesized report with [@source_id] citations
  report_index.json     Traceability: sentence -> source/claim IDs
  trace.jsonl           Full audit trail of all agent/tool actions
  eval.json             Evaluation metrics
  state.json            Checkpoint for resume
  cache.json            Persistent tool cache
  notes/                Per-source claim extractions (.json)
  code/                 Verification scripts
    logs/               stdout/stderr per sandbox execution
  artifacts/            Verification results (json/csv)
```

## Architecture

| Agent | Stage | Responsibility |
|-------|-------|---------------|
| **Planner** | PLAN | Decompose topic into research questions and report outline |
| **Collector** | COLLECT | Search and fetch sources via tool registry |
| **Reader** | READ | Extract structured claims with categories and evidence locations |
| **Verifier** | VERIFY | Run integrity + analysis verification scripts in sandbox; self-correct on failure |
| **Writer** | WRITE | Synthesize report with traceable citation markers; generate report_index.json |
| **QA** | QA | Check traceability, coverage, diversity; trigger multi-level rollbacks |

### State Machine

```
PLAN -> COLLECT -> READ -> VERIFY -> WRITE -> QA -> DONE
                    ^       ^         ^        |
                    +-------+---------+--------+
                           (rollback)
```

QA can roll back to READ (missing claims), VERIFY (failed verification), or WRITE (low coverage).

### Tool Registry

All external capabilities go through the governed registry:

- **Schema** — typed input/output definitions
- **Permissions** — `net`, `sandbox`, `fs`; denied tools logged in trace
- **Cache** — session (in-memory) or persistent (cache.json per run)
- **Risk levels** — low / medium / high

Built-in tools: `web_search`, `fetch`, `parse`, `sandbox_exec`, `cite`.

## Evaluation Metrics

`eval.json` contains:

- `citation_coverage` — ratio of paragraphs with citation markers
- `source_diversity` — unique domains and type distribution
- `reproduction_rate` — verification script success rate
- `unsupported_claim_rate` — report sentences not traceable to claims/sources
- `cache_hit_rate` — tool cache utilization
- `tool_calls` / `latency_sec` / `steps`
- `llm_enabled` / `estimated_cost_usd`

## Security

### Sandbox Constraints

1. **Working directory isolation** — scripts run in `runs/<id>/code/` only
2. **Timeout enforcement** — configurable, kills hung processes
3. **Resource limits** — Linux: `resource.setrlimit` for memory/CPU; macOS/Windows: graceful degradation recorded in trace
4. **Network blocking** (`--allow-net false`):
   - Monkeypatches: `socket.connect`, `urllib.request.urlopen`, `http.client.HTTP(S)Connection.request`, `requests.Session.send`, `httpx.Client.send`
   - Environment: `RESEARCHOPS_NO_NET=1`
   - **Limitation**: Best-effort Python-level blocking. Does not provide OS-level network namespace isolation. Use `--sandbox docker` (when implemented) for full isolation.
5. **Log capture** — stdout/stderr to `code/logs/<step>.out|.err`

### Tool Permissions

- `net` permission only granted with `--allow-net true`
- Denied tool calls logged in trace with `action=permission_denied` before raising error
- Collector gracefully falls back to built-in samples when `net` tools are denied

## Development

```bash
pip install -e ".[dev]"
pytest -v
ruff check src/ tests/
ruff format src/ tests/
make demo
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Ensure `pytest -v` passes and `ruff check .` is clean
4. Submit a pull request

All external tool integrations must go through the tool registry.

## License

MIT
