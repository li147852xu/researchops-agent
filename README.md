# ResearchOps

**A local-first AI workflow platform for research, market intelligence, and structured analytics, combining multi-agent orchestration, evidence-grounded outputs, and executable data analysis.**

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-green.svg)
![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-orange.svg)

---

## Why This Project?

Most "AI research assistants" are single-prompt wrappers tightly coupled to one domain. They cannot adapt to new verticals, lack source traceability, and offer no quality gating on their outputs.

ResearchOps takes a fundamentally different approach — a **reusable multi-agent orchestration core** that can be assembled into different vertical applications through lightweight configuration:

```
Configuration → Multi-Agent Pipeline → Evidence-Grounded Report → Quality Evaluation
 (thin app)     (7 agents + supervisor)   (source-traceable)        (automated metrics)
```

It is designed to demonstrate **production-grade AI engineering** — not just calling LLM APIs, but building a platform where agents collaborate, evidence is tracked, quality is measured, and new domains plug in without touching pipeline code.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          ResearchOps                                   │
├──────────────┬────────────────┬──────────────────┬────────────────────┤
│ Interface    │ App Layer      │ Agent Pipeline   │ Platform Core      │
├──────────────┼────────────────┼──────────────────┼────────────────────┤
│ Gradio UI    │ Gen. Research  │ Planner          │ Tool Registry      │
│ FastAPI      │ Market Intel   │ Collector        │ Hybrid Retrieval   │
│ Typer CLI    │ (your app)     │ Reader           │ LLM Abstraction    │
│ Demo Presets │ Config-driven  │ Verifier         │ Sandbox Execution  │
│              │                │ Writer           │ Checkpoint/Resume  │
│              │                │ QA + Supervisor   │ Evaluation Harness │
└──────────────┴────────────────┴──────────────────┴────────────────────┘
```

### Multi-Agent Pipeline (shared by all apps)

```
PLAN --> COLLECT --> READ --> VERIFY --> WRITE --> QA --> EVAL
  ^                    |                    |           |
  |                    +-(coverage low)-----+           |
  +---- Supervisor (rollback / retry) <----------------+
```

The Supervisor monitors quality signals (coverage gaps, low relevance, bucket incompleteness) and triggers targeted rollbacks with LLM-assisted remediation planning when evidence is insufficient.

### Agent Responsibilities

| Agent | Role |
|-------|------|
| **PlannerAgent** | Decomposes topic into research questions, builds outline and coverage checklist |
| **CollectorAgent** | Searches sources (arXiv, web, Semantic Scholar, Wikipedia), fetches and stores documents |
| **ReaderAgent** | Extracts structured claims with evidence spans from source text |
| **VerifierAgent** | Sandbox-executes verification scripts for numerical/code claims |
| **WriterAgent** | Generates report sections with mandatory citation markers |
| **QAAgent** | Checks coverage, detects gaps, triggers rollback if quality is insufficient |
| **Supervisor** | Analyses diagnostics and decides corrective rollback strategy with LLM-assisted remediation |

## Demo Apps

### General Research (`--app research`)

| **Input** | Any research topic or question (technology, policy, science, industry) |
|-----------|----------------------------------------------------------------------|
| **Output** | Structured Markdown report with numbered citations, evidence map, evaluation metrics |
| **Sources** | arXiv, web search, Semantic Scholar, Wikipedia (hybrid retrieval) |
| **Metrics** | Citation coverage, bucket coverage rate, relevance average, reproduction rate |

### Market Intelligence (`--app market`)

| **Input** | Company/sector analysis query + ticker symbol |
|-----------|----------------------------------------------|
| **Output** | Financial research memo with source-grounded claims, numerical extraction, freshness scoring |
| **Sources** | Same hybrid retrieval with finance-focused prompts and domain weighting |
| **Metrics** | Numerical claim rate, financial freshness score, type diversity, section coverage |

### Adding a New App

Each app is a **thin configuration layer** — no pipeline code, no duplicated agents:

```
apps/myapp/
├── __init__.py   (AppSpec registration)
├── config.py     (domain-specific fields)
├── prompts.py    (task-specific templates)
├── adapters.py   (tool registration)
└── evaluators.py (quality metrics)
```

Everything else — orchestration, state management, agent execution, retrieval, sandbox, checkpoint, tracing — comes from the shared `core/`.

## Quick Start

### Prerequisites

- Python 3.11+
- An LLM API key (DeepSeek, OpenAI, or Anthropic)

### Installation

```bash
pip install -e ".[web]"
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your provider settings (see Configuration section below)
```

### Launch

```bash
# Web UI (recommended) — includes demo presets for both apps
researchops web

# Or run via CLI
researchops run "transformer architectures" --app research
researchops run "NVDA competitive position" --app market --ticker NVDA
```

The Web UI includes **demo examples** for both apps — select from the dropdown or type your own query.

## Configuration

ResearchOps loads LLM settings from a `.env` file in the project root. Both the CLI and Web UI read this file automatically via `python-dotenv`.

**1. Copy the template:**

```bash
cp .env.example .env
```

**2. Edit `.env` with your provider settings:**

```bash
# LLM provider: openai_compat | openai | anthropic | none
LLM_BACKEND=openai_compat

# Model name
LLM_MODEL=deepseek-chat          # or gpt-4o-mini, claude-sonnet-4-20250514, etc.

# API base URL (required for openai_compat with non-OpenAI providers)
LLM_BASE_URL=https://api.deepseek.com/v1

# API key — set the one matching your provider
DEEPSEEK_API_KEY=sk-your-key-here
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

**Supported providers:**

| Provider | `LLM_BACKEND` | Key env var | Notes |
|----------|---------------|-------------|-------|
| DeepSeek | `openai_compat` | `DEEPSEEK_API_KEY` | Set `LLM_BASE_URL=https://api.deepseek.com/v1` |
| OpenAI | `openai_compat` or `openai` | `OPENAI_API_KEY` | Default base URL works for OpenAI |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | Claude models |
| OpenRouter / vLLM / Ollama | `openai_compat` | `OPENAI_API_KEY` or `LLM_API_KEY` | Set `LLM_BASE_URL` accordingly |
| None (rule-based) | `none` | — | For development without API access |

## Run Outputs

Each run produces a workspace under `runs/<run_id>/`:

| File | Description |
|------|-------------|
| `plan.json` | Research plan with questions, outline, coverage checklist |
| `sources.jsonl` | Collected sources with provenance metadata |
| `notes/*.json` | Per-source extracted claims with evidence spans |
| `report.md` | Final report with citation markers |
| `evidence_map.json` | Section-to-evidence mapping |
| `qa_report.json` | Quality assessment with issue classification |
| `eval.json` | Evaluation metrics (coverage, relevance, bucket completeness) |
| `trace.jsonl` | Full pipeline trace (every agent action, tool call, LLM invocation) |
| `state.json` | Checkpoint state for resume/replay |

## Key Design Decisions

**Supervisor-driven rollback with LLM remediation:** When QA detects coverage gaps, the Supervisor reads the diagnostics (coverage vector, bucket rates, relevance scores), uses the LLM to plan targeted remediation queries, and routes the pipeline back to collection. On re-collect, the Collector reads the Supervisor's suggested queries and negative terms from `last_decision.json`, clears all tool caches, escalates search strategy level, and enforces query novelty — ensuring each round fetches genuinely different sources.

**Evidence protocol:** Every claim in the report links to a specific source with explicit support status. The Writer agent is constrained to cite every statement; the QA agent validates citation coverage. In the Web UI, raw `[@source_id]` markers are transformed into numbered references with clickable links to source URLs.

**Hybrid retrieval with RRF:** BM25 keyword search and SentenceTransformer embedding search are combined via Reciprocal Rank Fusion, balancing lexical precision with semantic similarity for more comprehensive source discovery.

**Configuration-driven app architecture:** New domains require only a config class, prompt templates, tool registration, and an evaluator — no pipeline code, no agent code, no orchestration code. The core pipeline, all seven agents, and the full infrastructure are reused unchanged.

## CLI Reference

| Command | Description |
|---------|-------------|
| `researchops run TOPIC --app NAME` | Run a multi-agent pipeline |
| `researchops list-apps` | List registered apps |
| `researchops inspect-app NAME` | Show app specification |
| `researchops inspect-config NAME` | Show config JSON schema |
| `researchops web` | Launch Gradio Web UI |
| `researchops api` | Launch FastAPI server |
| `researchops eval RUN_DIR --app NAME` | Recompute evaluation |
| `researchops resume RUN_DIR` | Resume interrupted run |
| `researchops replay RUN_DIR` | Replay trace |

## Project Structure

```
src/researchops/
├── core/                       # Platform core (shared by all apps)
│   ├── config.py               # BaseAppConfig, shared enums
│   ├── pipeline.py             # Generic LangGraph pipeline builder
│   ├── state.py                # PipelineState, StateSnapshot, Decision
│   ├── context.py              # RunContext
│   ├── orchestration/          # GraphOrchestrator, Supervisor
│   ├── tools/                  # ToolRegistry, builtins, schema validation
│   ├── sandbox/                # Isolated execution runtime
│   ├── evaluation/             # Core metrics + evaluation harness
│   ├── quality.py              # Quality scoring algorithms
│   ├── checkpoint.py           # State persistence
│   ├── tracing.py              # Structured trace logging
│   ├── artifacts.py            # Evidence protocol
│   ├── persistence.py          # SQLite run index
│   └── replay.py               # Run replay
├── agents/                     # Multi-agent layer (7 agents)
├── reasoning/                  # LLM abstraction (OpenAI, Anthropic, rule-based)
├── retrieval/                  # BM25 + embedding + RRF + enhancement
├── apps/
│   ├── registry.py             # AppSpec, register/get/list apps
│   ├── research/               # General Research app (thin config layer)
│   └── market/                 # Market Intelligence app (thin config layer)
├── web/                        # Gradio Web UI (unified, demo presets)
├── api/                        # FastAPI REST API (unified)
├── cli.py                      # Typer CLI (unified)
└── utils.py                    # Shared helpers
```

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Orchestration** | LangGraph (StateGraph with conditional edges + supervisor rollback) |
| **LLM** | OpenAI-compatible (DeepSeek, OpenRouter, vLLM, Ollama), Anthropic, rule-based fallback |
| **Retrieval** | BM25 + SentenceTransformers + Reciprocal Rank Fusion |
| **API / Service** | FastAPI + Uvicorn |
| **Web UI** | Gradio (demo presets, live pipeline progress, citation rendering) |
| **CLI** | Typer + Rich |
| **Data Models** | Pydantic v2 |
| **Persistence** | SQLite + SQLAlchemy |
| **Sandbox** | Subprocess isolation (Docker-ready) |
| **Dev** | pytest, Ruff, Makefile |

## Development

```bash
pip install -e ".[dev]"
make check    # lint + test
make fmt      # auto-format
```

## Resume Talking Points

This project demonstrates:

- **Reusable multi-agent workflow core** — seven agents collaborating through LangGraph with supervisor-driven rollback, LLM-assisted remediation planning, and quality gating
- **AI application platform** — configuration-driven app assembly where new domains require only prompts, schemas, and tool policy; no pipeline code
- **Production-grade AI engineering** — LangGraph orchestration, FastAPI, Pydantic v2, tool governance with permissions/caching/audit, sandbox execution, hybrid retrieval (BM25 + embeddings + RRF), checkpoint/resume, evaluation pipelines
- **Evidence-grounded outputs** — every claim links to a source; citation coverage validated by QA agent; numbered references rendered in the Web UI
- **Intelligent re-collection** — supervisor suggestions wired into collector; query novelty enforcement; strategy escalation; cache invalidation ensures each rollback round fetches genuinely different sources
- **Configurable domain apps** — General Research and Market Intelligence as proof of core reusability; adding a new domain requires ~5 files

## License

MIT
