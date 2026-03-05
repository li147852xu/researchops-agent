# ResearchOps Agent

A multi-agent research orchestration harness that decomposes complex research topics into structured pipelines: **plan -> collect -> read -> verify -> write -> qa -> eval**. Each stage is handled by a specialized agent with full trace auditing, checkpoint/resume/replay, sandboxed code execution, governed tool registry, and pluggable LLM reasoning.

> Inspired by the quality bar of DeerFlow-style super agent harnesses — but built from scratch with an independent architecture.

## Features

- **Multi-agent pipeline** with explicit state machine (PLAN -> COLLECT -> READ -> VERIFY -> WRITE -> QA -> DONE)
- **Pluggable LLM reasoning** — runs fully offline with `--llm none` (default); supports OpenAI-compatible endpoints (DeepSeek, OpenRouter, vLLM, Ollama, Azure) via `--llm openai_compat`
- **Data pipeline integrity** — magic-number content detection, min-size thresholds, fake-PDF rejection, raw downloads separated from structured notes
- **Plan refinement** — automatically detects low RQ coverage after READ and triggers additional collection rounds
- **Conflict detection** — identifies opposing claims across sources, outputs `qa_conflicts.json`, and adds a Disagreements section to the report
- **Checkpoint / Resume / Replay** — interrupt and continue from any stage; replay traces with `--no-tools` dry-run or `--json` machine-readable output
- **Sandboxed execution** — subprocess sandbox with timeout, resource limits, network blocking (socket + http.client + urllib + requests + httpx), and log capture
- **Self-correcting verification** — 3+ strategy types (TERMS, COMPARISON, TREND, integrity) with automatic error detection and script repair
- **Quality gates** — parse rejects low-quality/code-heavy content; reader filters CSS/JS/template fragments; QA detects report garbage and triggers rollback
- **Report traceability** — every sentence traced to source/claim via `report_index.json`; QA validates coverage
- **Tool & Skill Registry** — schema-validated tool definitions with permission governance, risk levels, session + persistent caching
- **Evaluation scoring** — citation coverage, source diversity, reproduction rate, unsupported claim rate, conflict count, cache hit rate, artifacts count

## Quick Start

```bash
# Install
pip install -e .

# Offline demo (no network, no API key)
researchops run "demo topic" --mode fast --allow-net false --llm none

# Deep research with DeepSeek
researchops run "quantum computing safety" --mode deep --allow-net true \
  --llm openai_compat --llm-base-url https://api.deepseek.com/v1 \
  --llm-provider-label deepseek --llm-model deepseek-chat

# Resume interrupted run
researchops resume runs/<run_id>

# Replay trace (dry-run, machine-readable)
researchops replay runs/<run_id> --no-tools --json
```

Requires **Python 3.11+**.

## CLI Reference

```bash
researchops run "<topic>" \
  --mode {fast,deep} \
  --allow-net {true,false} \
  --llm {none,openai,openai_compat,anthropic} \
  --llm-model "<model>" \
  --llm-base-url "<url>" \
  --llm-api-key "<key>" \
  --llm-provider-label "<label>" \
  --llm-headers '{"X-Custom": "value"}' \
  --sandbox {subprocess,docker} \
  --budget <float> --max-steps <int> --seed <int>

researchops resume <run_dir>
researchops replay <run_dir> [--from-step N] [--no-tools] [--json]
researchops eval <run_dir>
```

API keys: `--llm-api-key`, or env vars `OPENAI_API_KEY` / `LLM_API_KEY` / `DEEPSEEK_API_KEY` / `ANTHROPIC_API_KEY`.

## Run Artifact Structure

```
runs/<run_id>/
  plan.json              Research questions, outline, thresholds
  sources.jsonl          One source per line (only successfully fetched sources)
  report.md              Synthesized report with [@source_id] citations
  report_index.json      Traceability: sentence -> source/claim IDs
  qa_conflicts.json      Detected claim conflicts per RQ
  trace.jsonl            Full audit trail
  eval.json              Evaluation metrics
  state.json             Checkpoint for resume
  cache.json             Persistent tool cache
  downloads/             Raw downloaded HTML/PDF files
  notes/                 Per-source structured reading cards (.json only)
  code/                  Verification scripts + logs/
  artifacts/             Verification results (json/csv)
```

## Architecture

| Agent | Stage | Responsibility |
|-------|-------|---------------|
| **Planner** | PLAN | Decompose topic into research questions and report outline |
| **Collector** | COLLECT | Search and fetch sources via tool registry; fallback to offline samples |
| **Reader** | READ | Extract claims with type, polarity, category, and evidence location |
| **Verifier** | VERIFY | Run integrity + strategy-specific verification in sandbox; self-correct |
| **Writer** | WRITE | Synthesize report with traceable citations and conflict notes |
| **QA** | QA | Check traceability, coverage, diversity, conflicts; trigger rollbacks |

## Security

- **Sandbox**: working directory isolation, timeout, resource limits (Linux), network blocking (best-effort Python-level monkeypatching of socket/http.client/urllib/requests/httpx)
- **Tool permissions**: `net` only granted with `--allow-net true`; denied calls logged in trace
- **Limitation**: Python-level blocking is not OS-level isolation. Use `--sandbox docker` (when implemented) for full isolation.

## Troubleshooting

**Report contains doctype/meta/CSS/JS garbage**: Usually caused by failed source downloads (captcha pages, redirects, dynamic sites). Run `make verify-run RUN=runs/<id>` to identify broken sources. Use `--allow-net true` with reliable domains, or fall back to offline mode for clean demo output.

## Development

```bash
pip install -e ".[dev]"
pytest -v
ruff check src/ tests/
make verify        # repo integrity checks
make verify-run RUN=runs/<id>  # run data integrity
```

## License

MIT
