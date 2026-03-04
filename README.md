# ResearchOps Agent

A multi-agent research orchestration harness that decomposes complex research topics into structured pipelines: **plan → collect → read → verify → write → qa → eval**. Each stage is handled by a specialized agent with full trace auditing, checkpoint/resume capability, sandboxed code execution, and a governed tool registry.

## Features

- **Multi-agent pipeline** with explicit state machine orchestration (PLAN → COLLECT → READ → VERIFY → WRITE → QA → DONE)
- **Checkpoint / Resume / Replay** — interrupt and continue from any stage; replay execution traces for auditing
- **Sandboxed execution** — subprocess sandbox with timeout, resource limits (Linux cgroups), network blocking via monkeypatch, and log capture
- **Tool & Skill Registry** — schema-validated tool definitions with permission governance, risk levels, caching policies
- **Structured run artifacts** — every run produces a fixed directory of plan, sources, notes, code, artifacts, report, trace, eval
- **Evaluation scoring** — citation coverage, source diversity, reproduction rate, tool call counts, latency
- **Offline demo mode** — runs end-to-end without network access using built-in sample data

## Installation

```bash
# Clone and install
git clone <repo-url> && cd researchops-agent
pip install -e .

# With dev dependencies (pytest, ruff)
pip install -e ".[dev]"
```

Requires **Python 3.11+**.

## CLI Usage

### Run a research pipeline

```bash
researchops run "large language model safety" --mode deep --allow-net true

# Offline demo (no network, uses built-in sample data)
researchops run "demo topic" --mode fast --allow-net false

# Full options
researchops run "<topic>" \
  --mode {fast,deep} \
  --checkpoint <path> \
  --budget <float> \
  --max-steps <int> \
  --allow-net {true,false} \
  --net-allowlist "arxiv.org,scholar.google.com" \
  --sandbox {subprocess,docker}
```

### Resume an interrupted run

```bash
researchops resume runs/<run_id>
```

### Replay a trace

```bash
researchops replay runs/<run_id> --from-step 5
researchops replay runs/<run_id> --no-tools
```

### Recompute evaluation

```bash
researchops eval runs/<run_id>
```

## Run Artifact Structure

Each run produces a fixed directory layout:

```
runs/<run_id>/
  plan.json           Research questions, outline, acceptance thresholds
  sources.jsonl       One source per line (id, type, url, domain, hash)
  report.md           Final synthesized report with citation markers
  trace.jsonl         Full audit trail of all agent/tool actions
  eval.json           Evaluation metrics
  state.json          Checkpoint state for resume
  notes/              Per-source claim extractions (.json)
  code/               Verification scripts
    logs/             stdout/stderr per sandbox execution
  artifacts/          Generated charts, tables, CSVs
```

## Architecture

### Agent Roles

| Agent | Stage | Responsibility |
|-------|-------|---------------|
| **Planner** | PLAN | Decompose topic into research questions and report outline |
| **Collector** | COLLECT | Search and fetch sources via tool registry |
| **Reader** | READ | Extract structured claims from each source |
| **Verifier** | VERIFY | Generate and run verification scripts in sandbox |
| **Writer** | WRITE | Synthesize report with citation markers |
| **QA** | QA | Check citation coverage, source diversity; trigger rollbacks |

### Tool Registry

All external capabilities are accessed through a governed registry. Each tool has:

- **Schema** — typed input/output definitions
- **Risk level** — low / medium / high
- **Permissions** — required permission set (e.g. `net`, `sandbox`)
- **Timeout** — per-tool default
- **Cache policy** — none / session / persistent

Built-in tools: `web_search`, `fetch`, `parse`, `sandbox_exec`, `cite`.

### State Machine

```
PLAN → COLLECT → READ → VERIFY → WRITE → QA → DONE
                                          ↑     |
                                          +-----+
                                        (rollback)
```

QA can trigger partial rollback to WRITE (or earlier stages). Retry budgets prevent infinite loops.

## Evaluation Metrics

`eval.json` contains:

- `citation_coverage` — ratio of report paragraphs containing citation markers
- `source_diversity` — unique domain count and source type distribution
- `reproduction_rate` — sandbox verification script success rate
- `tool_calls` — total tool invocations during the run
- `latency_sec` — total pipeline execution time
- `steps` — number of pipeline stages executed

## Security

### Sandbox Constraints

The subprocess sandbox enforces:

1. **Working directory isolation** — scripts execute in `runs/<id>/code/` only
2. **Timeout enforcement** — configurable per-run, kills hung processes
3. **Resource limits** — on Linux, `resource.setrlimit` caps memory and CPU; on macOS/Windows, limits degrade gracefully (recorded in trace)
4. **Network blocking** (`--allow-net false`):
   - Sets `RESEARCHOPS_NO_NET=1` environment variable
   - Injects a preamble that monkeypatches `socket.connect`, `urllib.request.urlopen` to raise `OSError`
   - **Limitation**: This is a best-effort block at the Python level. It does not provide OS-level network namespace isolation. A determined script could bypass it via ctypes or subprocess calls. For full network isolation, use `--sandbox docker` (when implemented) or run within an external network-restricted environment.
5. **Log capture** — all stdout/stderr written to `code/logs/<step>.out|.err`

### Tool Permissions

Agents cannot invoke tools without the required permissions being granted. The `net` permission is only granted when `--allow-net true`. The `sandbox` permission is always granted for verification.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Run offline demo
make demo
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Ensure tests pass (`pytest -v`) and code is formatted (`ruff format .`)
4. Submit a pull request with a clear description

All external tool integrations must go through the tool registry. Direct use of `requests`, `subprocess`, or network calls from agent code is prohibited by design.

## License

MIT
