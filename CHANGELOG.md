# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [1.1.0] — 2026-03-13

### Fixed

- **Rollback triggering**: RQs with zero claims now escalate to `high` severity, correctly triggering collect-stage rollbacks instead of silently passing QA
- **Bucket gap threshold**: `bucket_gap` severity now uses `config.bucket_coverage_threshold` (0.6 for deep mode) instead of a hardcoded 0.5
- **Conclusion quality**: Conclusion section now synthesises findings across all body sections via a dedicated `WRITER_CONCLUSION` prompt, instead of reusing the generic section-writing logic
- **Section headings**: Planner generates concise 4-8 word headings via LLM (with improved rule-based fallback) instead of using full research question text

### Improved

- **Web source parsing success rate**: Lowered quality thresholds across the parsing pipeline (`_MIN_QUALITY_CHARS` 500->200, `_MIN_TEXT_LEN_FOR_QUALITY` 200->100, entropy 6.5->7.0, code density 0.30->0.45, min claims 2->1, min content bytes 2048->1024)
- **HTML extraction chain**: Added `readability-lxml` as a second-tier extraction strategy between trafilatura and BeautifulSoup; trafilatura now falls through to alternatives when extraction is too short
- **Fetch reliability**: User-Agent string updated from `ResearchOps/0.3` to a realistic browser UA to avoid 403 rejections
- **Quickstart compatibility**: `quickstart.sh` now scans multiple Python executables (`python3.13`, `python3.12`, `python3.11`, `python3`, `python`) and provides conda hints when no suitable interpreter is found

### Security

- **API key redaction**: `state.json` no longer stores `llm_api_key` or `llm_headers` in cleartext; `RunConfig.safe_dump()` replaces sensitive values with `***REDACTED***`

### Added

- `PLANNER_HEADINGS` prompt template for LLM-based section heading generation
- `WRITER_CONCLUSION` prompt template for cross-section synthesis
- `RunConfig.safe_dump()` method for secure serialization

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
