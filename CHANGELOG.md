# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2026-03-13

Initial release. Full-featured multi-agent research orchestration system.

### Core Architecture

- **Multi-agent pipeline** — six specialized agents (Planner, Collector, Reader, Verifier, Writer, QA) with structured inputs/outputs
- **LangGraph orchestration** — compiled `StateGraph` with typed state (`ResearchState` TypedDict), conditional edges for dynamic routing, supervisor node for intelligent rollback decisions; legacy sequential orchestrator available via `--no-graph`
- **Supervisor policy layer** — diagnoses rollback root causes with structured Decision Records (`reason_codes`, `action_plan`, `confidence`, `policy_version`); topic-adaptive coverage buckets; negative-term anti-drift filtering

### Retrieval & Knowledge

- **Hybrid RAG** — BM25 keyword matching + sentence-transformer embeddings (default: `all-MiniLM-L6-v2`) fused with Reciprocal Rank Fusion (RRF, k=60); three modes: `none`, `bm25`, `hybrid`
- **arXiv-first ingestion** — `arxiv_search` (Atom API) and `arxiv_download_pdf` tools; source strategies: `demo`, `arxiv`, `web`, `hybrid`
- **Tool registry** — centralized governance with schema validation, permission checks, persistent caching, and audit logging; 7 built-in tools

### Writing & Evaluation

- **Evidence-first writing** — mandatory citation markers enforced by QA; evidence map generation; source diversity checks
- **LLM-as-Judge** — RAGAS-style evaluation across faithfulness, coverage, coherence, and relevance (enabled via `--judge`)
- **18+ automated metrics** — citation coverage, unsupported claim rate, bucket coverage, relevance avg, reproduction rate, source diversity, and more

### Engineering

- **Prompt template system** — centralized `PromptTemplate` dataclasses with system prompts, user templates, and few-shot examples; robust JSON response parser with multiple fallback strategies
- **Typed state management** — `ResearchState` TypedDict for graph state; `WorkingMemory` for inter-node context
- **Shared utilities** — deduplicated data loaders (`load_sources`, `load_all_notes`, `load_plan`, `load_claim_dicts`, `compute_coverage`), text chunking, negative-term lookup
- **Sandbox verification** — subprocess sandbox with timeout, resource limits, and network blocking; self-correcting script retry
- **Checkpoint/resume/replay** — full pipeline state persistence; trace-based replay with `--json` and `--no-tools` modes
- **Pluggable LLM** — supports `none` (rule-based), `openai_compat` (DeepSeek, OpenAI, Ollama, vLLM), and `anthropic` backends
- **130+ tests** with pytest; ruff linting; batch evaluation across topic sets
- **One-click setup** — `make quickstart` for full environment bootstrap
