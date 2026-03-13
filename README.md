# ResearchOps Agent

**An open-source multi-agent research orchestration system** that plans, collects, reads, verifies, writes, and evaluates — producing traceable, evidence-backed research reports with full audit trails.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-orange.svg)](CHANGELOG.md)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Why ResearchOps Agent](#why-researchops-agent)
- [Architecture](#architecture)
- [Core Features](#core-features)
  - [Multi-Agent Pipeline](#multi-agent-pipeline)
  - [LangGraph Orchestration](#langgraph-orchestration)
  - [Supervisor & Policy Layer](#supervisor--policy-layer)
  - [Hybrid RAG](#hybrid-rag)
  - [LLM-as-Judge Evaluation](#llm-as-judge-evaluation)
  - [Sandbox & Verification](#sandbox--verification)
  - [Tool Registry & Governance](#tool-registry--governance)
  - [Evidence-First Writing](#evidence-first-writing)
  - [Prompt Engineering](#prompt-engineering)
  - [Checkpoint / Resume / Replay](#checkpoint--resume--replay)
- [Run Workspace](#run-workspace)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [Evaluation](#evaluation)
- [Security](#security)
- [Development](#development)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Quick Start

### One-Click Setup

```bash
git clone https://github.com/li147852xu/researchops-agent.git
cd researchops-agent
make quickstart
```

This will check your Python version, create a virtual environment, install all dependencies (including dev and embeddings extras), run the test suite, and print next-step instructions.

### Manual Setup

```bash
pip install -e .                   # Core only
pip install -e ".[dev]"            # + pytest, ruff
pip install -e ".[embeddings]"     # + sentence-transformers for hybrid RAG
```

### Run Your First Research (offline, no API key)

```bash
researchops run "demo topic" --mode fast --allow-net false --llm none --sources demo
```

### Run with LLM (requires API key)

```bash
researchops run "deep learning optimization" \
  --llm openai_compat \
  --llm-base-url https://api.deepseek.com/v1 \
  --llm-model deepseek-chat \
  --llm-api-key "$DEEPSEEK_API_KEY" \
  --sources hybrid --retrieval hybrid --graph --judge
```

### Check Output

```bash
ls runs/run_*/
# plan.json  sources.jsonl  notes/  code/  artifacts/
# report.md  trace.jsonl  eval.json  state.json  ...
```

## Why ResearchOps Agent

Most research agents stop at "search and summarize." They produce shallow outputs with no traceability, no verification, and no way to audit what went wrong.

ResearchOps Agent runs a **full research lifecycle** — from planning research questions to verifying claims in a sandboxed environment — and produces structured, citable artifacts at every stage. Every sentence in the final report traces back to a source. Every decision the system makes is logged. Every run is reproducible.

## Architecture

ResearchOps Agent is built on a four-layer design:

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 1: Orchestration (LangGraph)                              │
│  StateGraph + Conditional Edges + Typed State (ResearchState)    │
│                                                                  │
│   PLAN → COLLECT → READ → VERIFY → WRITE → QA → EVAL            │
│                     ↑               ↑       │                    │
│                     └── supervisor ←─┘──────┘                    │
├──────────────────────────────────────────────────────────────────┤
│  Layer 2: Policy                                                 │
│  Supervisor + Coverage Checklist + Relevance Gate                │
│                                                                  │
│   Topic-adaptive buckets → Decision Records                      │
│   Negative terms → Anti-drift filtering                          │
│   Reason codes → Explainable rollbacks                           │
├──────────────────────────────────────────────────────────────────┤
│  Layer 3: Knowledge & Retrieval                                  │
│  Tool Registry + Hybrid RAG (BM25 + Embedding + RRF)             │
│                                                                  │
│   Schema-validated tools with permission governance              │
│   Run-scoped hybrid retrieval over extracted claims              │
│   Pluggable LLM: none / openai_compat / anthropic               │
├──────────────────────────────────────────────────────────────────┤
│  Layer 4: Evaluation                                             │
│  18+ Automated Metrics + LLM-as-Judge (RAGAS-style)              │
│                                                                  │
│   Faithfulness, Coverage, Coherence, Relevance scoring           │
│   Batch evaluation across topic sets                             │
└──────────────────────────────────────────────────────────────────┘
```

**Layer 1 (Orchestration)** uses LangGraph to define a compiled state graph with conditional edges. Nodes wrap agent logic; edges encode rollback policies. Typed state (`ResearchState` TypedDict) flows through the graph. A legacy sequential orchestrator is also available via `--no-graph`.

**Layer 2 (Policy)** makes the system intelligent. The Supervisor diagnoses rollback root causes, generates action plans with specific query suggestions, and logs structured Decision Records.

**Layer 3 (Knowledge)** provides capabilities. Tools for searching, fetching, parsing. Hybrid RAG (BM25 + sentence-transformer embeddings fused with Reciprocal Rank Fusion) for finding relevant claims. Pluggable LLM integration.

**Layer 4 (Evaluation)** quantifies output quality. 18+ automated metrics plus optional LLM-as-Judge scoring across faithfulness, coverage, coherence, and relevance dimensions.

## Core Features

### Multi-Agent Pipeline

Research is decomposed into six specialized agents, each with a clear responsibility and structured output:

```
PLAN → COLLECT → READ → VERIFY → WRITE → QA → EVAL
                         ↑                  │
                         └── rollback ───────┘
```

| Agent | Responsibility | Output |
|-------|---------------|--------|
| **Planner** | Generates research questions, report outline, coverage checklist | `plan.json` |
| **Collector** | Searches arXiv / web, downloads sources, deduplicates | `sources.jsonl` |
| **Reader** | Extracts structured claims with chunked text processing | `notes/<source_id>.json` |
| **Verifier** | Generates and runs verification scripts in sandbox | `code/`, `artifacts/` |
| **Writer** | Produces evidence-backed report with mandatory citations | `report.md` |
| **QA** | Checks coverage, diversity, conflicts; triggers rollbacks | `qa_report.json` |

### LangGraph Orchestration

The default orchestrator compiles a LangGraph `StateGraph` with:

- **Typed state** (`ResearchState` TypedDict) flowing between nodes
- **Conditional edges** for dynamic routing (e.g., `after_qa` routes to `evaluate` on pass, `write` or `supervisor` on failure)
- **Supervisor node** for intelligent rollback decisions
- **Evaluation node** that runs after QA passes

The graph supports multiple rollback cycles. A `--no-graph` flag falls back to the legacy sequential orchestrator with a while-loop state machine.

### Supervisor & Policy Layer

A **Supervisor** layer advises on every rollback decision — diagnosing the root cause, generating an action plan, and logging a structured **Decision Record** to `decisions.jsonl`.

When QA detects insufficient bucket coverage, the Supervisor identifies *which* topic buckets are underserved, suggests *specific queries* with negative filter terms, and records its reasoning with `reason_codes`, `action_plan`, `confidence`, and `policy_version`.

### Hybrid RAG

Three retrieval modes for finding relevant claims during writing and QA:

| Mode | Method | Use Case |
|------|--------|----------|
| `none` | No retrieval | Minimal runs |
| `bm25` | BM25 keyword matching (rank-bm25) | Default, fast |
| `hybrid` | BM25 + sentence-transformer embeddings fused with Reciprocal Rank Fusion | Best quality |

The hybrid mode uses a local `sentence-transformers` model (default: `all-MiniLM-L6-v2`) for semantic search, combined with BM25 for keyword matching. Results are fused using RRF (k=60). Install the embeddings extra: `pip install -e ".[embeddings]"`.

### LLM-as-Judge Evaluation

Beyond automated metrics, the `--judge` flag enables RAGAS-style LLM evaluation:

| Dimension | What It Measures |
|-----------|-----------------|
| **Faithfulness** | Are claims supported by cited sources? |
| **Coverage** | Does the report address all research questions? |
| **Coherence** | Is the report well-structured and logical? |
| **Relevance** | Is the content relevant to the topic? |

Each dimension is scored 0-1 by the LLM, producing an overall quality score.

### Sandbox & Verification

The Verifier agent generates Python scripts to validate claims — checking statistical consistency, reproducing calculations, extracting structured data. Scripts run in an isolated subprocess sandbox with timeout enforcement, resource limits, and network blocking.

### Tool Registry & Governance

All external capabilities route through a centralized **Tool Registry** with schema validation, permission checks, and audit logging. Built-in tools:

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

Every sentence in the report must cite its source — a hard constraint enforced by QA. The Writer builds an **evidence map** before writing, grouping claims by section and checking source diversity.

### Prompt Engineering

All LLM interactions use centralized `PromptTemplate` dataclasses with:
- Dedicated system prompts per agent role
- Structured user prompt templates with variable substitution
- Few-shot examples for consistent output formatting
- Robust JSON parsing with multiple fallback strategies

### Checkpoint / Resume / Replay

Long research runs can be interrupted and resumed:

```bash
researchops resume runs/<run_id>            # Resume from checkpoint
researchops replay runs/<run_id> --no-tools  # Dry-run replay
researchops replay runs/<run_id> --json      # Structured JSON replay
```

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
| `--retrieval` | `bm25` | `none` / `bm25` / `hybrid` |
| `--graph/--no-graph` | `--graph` | LangGraph orchestrator (default) or legacy sequential |
| `--judge` | off | Enable LLM-as-Judge evaluation after run |
| `--embedder-model` | `all-MiniLM-L6-v2` | SentenceTransformer model for hybrid retrieval |
| `--llm` | `none` | `none` / `openai_compat` / `anthropic` |
| `--llm-model` | | Model name for the LLM provider |
| `--llm-base-url` | | API base URL for openai_compat |
| `--llm-api-key` | | API key (or use env var) |
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

With `--judge`, additional LLM-as-Judge scores are included: `faithfulness`, `coverage`, `coherence`, `relevance`, and `overall`.

**Batch evaluation** across multiple topics:

```bash
python scripts/run_evalset.py
# Results in runs_batch/aggregate_metrics.json
```

## Security

- **Network isolation**: `--allow-net false` blocks all network access
- **Sandbox isolation**: Verification scripts run in subprocess sandbox with timeout and resource limits
- **Tool governance**: All external capabilities route through the Tool Registry with schema validation and permission checks
- **API key handling**: Keys are never logged in trace

## Development

```bash
make quickstart        # Full one-click setup (venv + deps + test)
make dev               # Install with dev dependencies
make test              # Run full test suite (130+ tests)
make lint              # Run ruff linter
make fmt               # Auto-format with ruff
make demo              # Quick offline demo run
make run-llm TOPIC="deep learning" LLM_ARGS="--llm openai_compat ..."
make verify            # Repository constraint checks
make verify-run RUN=<dir>     # Run artifact integrity
make verify-loop RUN=<dir>    # Infinite rollback detection
make verify-quality RUN=<dir> # Research quality verification
make evalset           # Batch evaluation
make clean             # Remove runs, caches, build artifacts
```

## Project Structure

```
src/researchops/
├── cli.py                 # Typer CLI entry point
├── orchestrator.py        # LangGraph + legacy sequential orchestrator
├── supervisor.py          # Policy advisor + decision records
├── config.py              # RunConfig + RunMode + enums
├── models.py              # Pydantic data models
├── utils.py               # Shared helpers (loaders, chunking, coverage)
├── evaluator.py           # Automated metric computation (18+)
├── evaluator_llm.py       # LLM-as-Judge evaluation (RAGAS-style)
├── checkpoint.py          # State save/restore
├── trace.py               # Event tracing
├── graph/                 # LangGraph orchestration
│   ├── builder.py         #   Graph compilation (StateGraph)
│   ├── state.py           #   ResearchState TypedDict + WorkingMemory
│   ├── nodes.py           #   Node functions wrapping agents
│   └── edges.py           #   Conditional edge functions
├── prompts/               # Prompt engineering
│   ├── templates.py       #   PromptTemplate dataclass + all templates
│   └── parser.py          #   Robust JSON response parser
├── agents/                # Six specialized agents
│   ├── base.py            #   AgentBase + RunContext
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
│   ├── base.py
│   ├── proc.py
│   └── container.py
├── reasoning/             # LLM abstraction
│   ├── base.py
│   ├── none.py
│   ├── openai_compat.py
│   └── anthropic_r.py
└── retrieval/             # Run-scoped search
    ├── __init__.py
    ├── base.py
    ├── bm25.py
    ├── embedding.py       # SentenceTransformer embeddings
    └── hybrid.py          # BM25 + Embedding with RRF
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

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — agent graph orchestration
- **[langchain-core](https://github.com/langchain-ai/langchain)** — foundation for LangGraph
- **[sentence-transformers](https://github.com/UKPLab/sentence-transformers)** — local embedding models for hybrid RAG
- **[Pydantic](https://github.com/pydantic/pydantic)** — data validation and settings management
- **[Typer](https://github.com/tiangolo/typer)** — CLI framework
- **[Rich](https://github.com/Textualize/rich)** — terminal formatting
- **[rank-bm25](https://github.com/dorianbrown/rank_bm25)** — BM25 retrieval
- **[Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/)** — HTML parsing
- **[pypdf](https://github.com/py-pdf/pypdf)** — PDF extraction
- **[httpx](https://github.com/encode/httpx)** — HTTP client
- **[Trafilatura](https://github.com/adbar/trafilatura)** — web content extraction
