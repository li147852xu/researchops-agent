# ResearchOps Agent

**An open-source research orchestration harness** that plans, collects, reads, verifies, writes, and evaluates — producing traceable, evidence-backed research reports with full audit trails.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.1.0-orange.svg)](CHANGELOG.md)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Why ResearchOps Agent](#why-researchops-agent)
- [Core Features](#core-features)
  - [Multi-Agent Pipeline](#multi-agent-pipeline)
  - [Supervisor & Policy Layer](#supervisor--policy-layer)
  - [Sandbox & Verification](#sandbox--verification)
  - [Tool Registry & Governance](#tool-registry--governance)
  - [Evidence-First Writing](#evidence-first-writing)
  - [Checkpoint / Resume / Replay](#checkpoint--resume--replay)
- [Architecture](#architecture)
- [Run Workspace](#run-workspace)
- [Configuration](#configuration)
  - [LLM Providers](#llm-providers)
  - [Source Strategies](#source-strategies)
  - [Run Modes](#run-modes)
- [CLI Reference](#cli-reference)
- [Evaluation](#evaluation)
- [Security](#security)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Quick Start

### 1. Install

```bash
git clone https://github.com/li147852xu/researchops-agent.git
cd researchops-agent
pip install -e .
```

### 2. Run your first research (offline, no API keys needed)

```bash
researchops run "demo topic" --mode fast --allow-net false --llm none --sources demo
```

This runs the full pipeline — plan, collect, read, verify, write, QA, eval — using built-in sample data. No network, no LLM, zero configuration.

### 3. Check the output

```bash
ls runs/run_*/
# plan.json  sources.jsonl  notes/  code/  artifacts/
# report.md  trace.jsonl  eval.json  state.json  ...
```

### 4. (Optional) Install extras

```bash
pip install -e ".[dev]"        # pytest + ruff
pip install -e ".[quality]"    # trafilatura for better HTML extraction
```

## Why ResearchOps Agent

Most research agents stop at "search and summarize." They produce shallow outputs with no traceability, no verification, and no way to audit what went wrong.

ResearchOps Agent takes a different approach. It runs a **full research lifecycle** — from planning research questions to verifying claims in a sandboxed environment — and produces structured, citable artifacts at every stage. Every sentence in the final report traces back to a source. Every decision the system makes is logged. Every run is reproducible.

The result: research you can actually trust, inspect, and build on.

## Core Features

### Multi-Agent Pipeline

Research is not a single prompt. It's a process.

ResearchOps decomposes research into six specialized agents, each with a clear responsibility and structured output. The Orchestrator drives them through an explicit state machine, handling rollbacks, retries, and adaptive strategy changes automatically.

```
PLAN → COLLECT → READ → VERIFY → WRITE → QA → EVAL
                         ↑                  │
                         └── rollback ───────┘
```

| Agent | Responsibility | Output |
|-------|---------------|--------|
| **Planner** | Generates research questions, report outline, coverage checklist | `plan.json` |
| **Collector** | Searches arXiv / web, downloads sources, deduplicates | `sources.jsonl` |
| **Reader** | Extracts structured claims with evidence spans | `notes/<source_id>.json` |
| **Verifier** | Generates and runs verification scripts in sandbox | `code/`, `artifacts/` |
| **Writer** | Produces evidence-backed report with mandatory citations | `report.md` |
| **QA** | Checks coverage, diversity, conflicts; triggers rollbacks | `qa_report.json` |

### Supervisor & Policy Layer

Agents don't just blindly retry. A **Supervisor** layer advises on every rollback decision — diagnosing the root cause, generating an action plan, and logging a structured **Decision Record**.

When QA detects insufficient bucket coverage, the Supervisor doesn't just say "collect more." It identifies *which* topic buckets are underserved, suggests *specific queries* with negative filter terms to prevent drift, and records its reasoning for full auditability.

```
runs/<run_id>/decisions.jsonl
```

Each decision includes `reason_codes`, `action_plan`, `confidence`, and `policy_version` — making the system's behavior fully explainable and debuggable.

### Sandbox & Verification

ResearchOps doesn't just *write about* findings. It verifies them.

The Verifier agent generates Python scripts to validate claims — checking statistical consistency, reproducing calculations, extracting structured data. Scripts run in an isolated subprocess sandbox with timeout enforcement, resource limits, and network blocking.

When a script fails, the Verifier reads the error, patches the script, and retries — at least twice — before reporting failure. All stdout/stderr is captured to `code/logs/`.

```
runs/<run_id>/code/
├── verify_rq1.py          # Generated verification script
├── verify_rq2.py
└── logs/
    ├── 4.out              # stdout
    └── 4.err              # stderr
```

### Tool Registry & Governance

All external capabilities — web search, page fetch, PDF parsing, sandbox execution — go through a centralized **Tool Registry**. No agent can make raw HTTP requests or spawn subprocesses directly.

Each tool is defined with a schema: `name`, `version`, `input_schema`, `output_schema`, `risk_level`, `permissions`, `timeout_default`, and `cache_policy`. Permission checks happen on every call. Results are cached per-session. Everything is traced.

Built-in tools:

| Tool | Description |
|------|-------------|
| `web_search` | DuckDuckGo / SerpAPI with graceful degradation |
| `arxiv_search` | arXiv Atom API for paper discovery |
| `arxiv_download_pdf` | PDF download with hash verification |
| `fetch_page` | HTML/PDF download with magic-number detection |
| `parse_doc` | HTML (bs4/trafilatura) and PDF (pypdf) extraction |
| `sandbox_exec` | Subprocess sandbox with timeout and net-block |
| `cite` | Source/claim to citation marker mapping |

### Evidence-First Writing

Every sentence in the report must cite its source. This is not a guideline — it's a hard constraint enforced by QA.

The Writer builds an **evidence map** before writing, grouping claims by section and checking source diversity. Sections that lack sufficient evidence get explicit gap messaging rather than hallucinated content. QA validates citation coverage and triggers rollbacks to Collector when evidence is insufficient.

```
runs/<run_id>/
├── report.md              # Sentences with [@source_id] markers
├── report_index.json      # Sentence → claim/source mapping
└── evidence_map.json      # Section → bucket/claims/diversity
```

### Checkpoint / Resume / Replay

Long research runs can be interrupted and resumed. The Orchestrator saves full pipeline state to `state.json` after every stage transition.

```bash
# Resume from where you left off
researchops resume runs/<run_id>

# Replay the trace log (read-only, no tool execution)
researchops replay runs/<run_id> --no-tools

# Replay as structured JSON events
researchops replay runs/<run_id> --json
```

## Architecture

ResearchOps Agent is built on a three-layer design:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Harness                                       │
│  Orchestrator + State Machine + Checkpoint/Resume       │
│                                                         │
│   PLAN → COLLECT → READ → VERIFY → WRITE → QA → DONE   │
│                     ↑                        │          │
│                     └──── rollback ──────────┘          │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Policy                                        │
│  Supervisor + Coverage Checklist + Relevance Gate       │
│                                                         │
│   Topic-adaptive buckets → Decision Records             │
│   Negative terms → Anti-drift filtering                 │
│   Reason codes → Explainable rollbacks                  │
├─────────────────────────────────────────────────────────┤
│  Layer 3: Knowledge                                     │
│  Tool Registry + BM25 Retrieval + LLM                   │
│                                                         │
│   Schema-validated tools with permission governance     │
│   Run-scoped BM25 index over extracted claims           │
│   Pluggable LLM: none / openai_compat / anthropic      │
└─────────────────────────────────────────────────────────┘
```

**Layer 1 (Harness)** handles orchestration, state management, and fault tolerance. It knows nothing about research — it runs a state machine.

**Layer 2 (Policy)** makes the system intelligent. The Supervisor generates coverage checklists tailored to the research topic, scores source relevance, and produces structured decisions with explainable reason codes.

**Layer 3 (Knowledge)** provides the capabilities. Tools for searching, fetching, parsing. BM25 retrieval for finding relevant claims. LLM integration for smarter planning, reading, and writing.

## Run Workspace

Every run produces the same structured workspace:

```
runs/<run_id>/
├── plan.json              # Research questions + outline + coverage checklist
├── sources.jsonl          # Collected sources (one per line)
├── failures.json          # Failed source attempts
├── downloads/             # Raw HTML/PDF files
├── notes/                 # Structured reading cards (JSON)
│   └── <source_id>.json   #   claims, relevance_score, bucket_hits
├── code/                  # Verification scripts
│   └── logs/              #   stdout/stderr per step
├── artifacts/             # Verification outputs (JSON/CSV)
├── report.md              # Generated report with citations
├── report_index.json      # Sentence → claim/source traceability
├── evidence_map.json      # Section → bucket/claims/diversity
├── decisions.jsonl        # Supervisor decision records
├── retrieval_index.json   # BM25 claim index
├── qa_report.json         # QA results + bucket coverage
├── qa_conflicts.json      # Detected claim conflicts
├── trace.jsonl            # Full event trace
├── eval.json              # Evaluation metrics (18+)
├── state.json             # Pipeline state (for resume)
└── cache.json             # Tool result cache
```

## Configuration

### LLM Providers

ResearchOps works without any LLM (`--llm none`) using rule-based templates. For higher quality, connect any OpenAI-compatible or Anthropic API:

```bash
# No LLM (default) — fully functional, rule-based
researchops run "topic" --llm none

# DeepSeek
researchops run "topic" --llm openai_compat \
  --llm-base-url https://api.deepseek.com/v1 \
  --llm-model deepseek-chat \
  --llm-api-key "$DEEPSEEK_API_KEY"

# OpenAI
researchops run "topic" --llm openai_compat \
  --llm-base-url https://api.openai.com/v1 \
  --llm-model gpt-4o \
  --llm-api-key "$OPENAI_API_KEY"

# Anthropic
researchops run "topic" --llm anthropic \
  --llm-model claude-sonnet-4-20250514 \
  --llm-api-key "$ANTHROPIC_API_KEY"

# Local (Ollama, vLLM, etc.)
researchops run "topic" --llm openai_compat \
  --llm-base-url http://localhost:11434/v1 \
  --llm-model llama3
```

API key resolution priority: `--llm-api-key` > `OPENAI_API_KEY` > `LLM_API_KEY` > `DEEPSEEK_API_KEY`

### Source Strategies

| Strategy | Description | Network |
|----------|-------------|---------|
| `demo` | Built-in sample HTML/TXT/PDF | Offline |
| `arxiv` | arXiv Atom API search + PDF download | Required |
| `web` | Web search (DuckDuckGo) + page fetch | Required |
| `hybrid` | arXiv + web combined | Required |

```bash
researchops run "quantum computing" --sources arxiv --allow-net true
researchops run "market analysis" --sources web --allow-net true
researchops run "deep learning" --sources hybrid --mode deep
```

### Run Modes

| Mode | Sources | Collect Rounds | Bucket Coverage | Description |
|------|---------|---------------|-----------------|-------------|
| `fast` | 12 max | 3 | 60% threshold | Quick exploration |
| `deep` | 40 max | 6 | 80% threshold | Thorough research |

## CLI Reference

| Command | Description |
|---------|-------------|
| `researchops run "<topic>"` | Full pipeline execution |
| `researchops resume <run_dir>` | Continue from checkpoint |
| `researchops replay <run_dir>` | Replay trace events |
| `researchops eval <run_dir>` | Recompute eval.json |
| `researchops verify-run <run_dir>` | Check run artifact integrity |
| `researchops verify-repo` | Check repository constraints |

### `run` options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `fast` | `fast` or `deep` |
| `--sources` | `hybrid` | `demo` / `arxiv` / `web` / `hybrid` |
| `--retrieval` | `bm25` | `none` / `bm25` |
| `--llm` | `none` | `none` / `openai_compat` / `anthropic` |
| `--allow-net` | `true` | Network access control |
| `--sandbox` | `subprocess` | `subprocess` / `docker` |
| `--budget` | `10.0` | Budget cap (USD) |
| `--max-steps` | `50` | Maximum pipeline steps |

## Evaluation

Each run outputs `eval.json` with 18+ metrics:

| Metric | Description |
|--------|-------------|
| `citation_coverage` | Ratio of paragraphs with citation markers |
| `unsupported_claim_rate` | Sentences without evidence backing |
| `bucket_coverage_rate` | Fraction of topic buckets with sufficient evidence |
| `relevance_avg` | Average relevance score across sources |
| `decision_count` | Number of Supervisor decision records |
| `papers_per_rq` | arXiv papers per research question |
| `source_diversity` | Unique domains and type distribution |
| `reproduction_rate` | Verification script success rate |
| `section_nonempty_rate` | Report sections with content |
| `low_quality_source_rate` | Sources flagged as low quality |
| `conflict_count` | Detected claim conflicts across sources |
| `latency_sec` | Total pipeline execution time |

**Batch evaluation** across multiple topics:

```bash
# Edit evalset/topics.jsonl to define topics
python scripts/run_evalset.py
# Results in runs_batch/aggregate_metrics.json
```

## Security

- **Network isolation**: `--allow-net false` blocks all network access via socket/http/urllib/requests/httpx monkeypatching. Strategy is best-effort at the subprocess level; see `trace.jsonl` for policy details.
- **Sandbox isolation**: Verification scripts run in subprocess sandbox with working directory restricted to `runs/<id>/code`, configurable timeout, and resource limits (Linux `resource` module; degraded on other platforms with trace logging).
- **Tool governance**: All external capabilities route through the Tool Registry with schema validation, permission checks, and audit logging. No agent can bypass the registry.
- **API key handling**: Keys are never logged in trace. Resolution via CLI flag > environment variables.

## Development

```bash
make dev              # Install with dev dependencies
make test             # Run full test suite (97+ tests)
make lint             # Run ruff linter
make fmt              # Auto-format with ruff
make demo             # Quick offline demo run
make verify           # Repository constraint checks
make verify-run RUN=<dir>     # Run artifact integrity
make verify-loop RUN=<dir>    # Infinite rollback detection
make verify-quality RUN=<dir> # Research quality verification
make evalset          # Batch evaluation
make clean            # Remove runs, caches, build artifacts
```

### Project Structure

```
src/researchops/
├── cli.py                 # Typer CLI entry point
├── orchestrator.py        # State machine + pipeline driver
├── supervisor.py          # Policy advisor + decision records
├── config.py              # RunConfig + RunMode + enums
├── models.py              # Pydantic data models
├── evaluator.py           # Metric computation
├── checkpoint.py          # State save/restore
├── trace.py               # Event tracing
├── agents/                # Six specialized agents
│   ├── planner.py
│   ├── collector.py
│   ├── reader.py
│   ├── verifier.py
│   ├── writer.py
│   └── qa.py
├── registry/              # Tool registry + governance
│   ├── manager.py
│   ├── schema.py
│   └── builtin.py
├── tools/                 # Tool implementations
│   ├── web_search.py
│   ├── fetch_page.py
│   ├── parse_doc.py
│   ├── arxiv_search.py
│   ├── arxiv_download.py
│   ├── sandbox_exec.py
│   └── cite.py
├── sandbox/               # Execution isolation
│   ├── proc.py
│   └── container.py
├── reasoning/             # LLM abstraction
│   ├── none.py
│   ├── openai_compat.py
│   └── anthropic_r.py
└── retrieval/             # Run-scoped search
    └── bm25.py
```

## Contributing

Contributions are welcome. Please ensure:

1. All tests pass: `make test`
2. Linter is clean: `make lint`
3. New features include tests
4. Repository constraints hold: `make verify`

## License

This project is open source and available under the [MIT License](./LICENSE).

## Acknowledgments

ResearchOps Agent is built with the open-source ecosystem:

- **[Pydantic](https://github.com/pydantic/pydantic)** — data validation and settings management
- **[Typer](https://github.com/tiangolo/typer)** — CLI framework
- **[Rich](https://github.com/Textualize/rich)** — terminal formatting
- **[rank-bm25](https://github.com/dorianbrown/rank_bm25)** — BM25 retrieval
- **[Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/)** — HTML parsing
- **[pypdf](https://github.com/py-pdf/pypdf)** — PDF extraction
- **[httpx](https://github.com/encode/httpx)** — HTTP client
- **[Trafilatura](https://github.com/adbar/trafilatura)** — web content extraction
