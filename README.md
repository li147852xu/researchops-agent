# ResearchOps Agent

**Multi-agent research orchestration harness** — long-horizon task execution with traceable reports, verifiable artifacts, and reproducible runs.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## What it does

ResearchOps Agent takes a research topic and runs a full pipeline:

```
PLAN → COLLECT → READ → VERIFY → WRITE → QA → EVAL
```

Each run produces a **workspace** with structured artifacts: plan, sources, reading notes, verification scripts + outputs, report with traceable citations, evaluation metrics, and a replay-able trace log.

## Key capabilities

- **arXiv-first + hybrid ingestion** — arXiv Atom API for paper search/download, web sources as supplement; `--sources {demo,arxiv,web,hybrid}`
- **Evidence-first writing** — every report sentence must cite a source; unsupported sections trigger rollback
- **Run-scoped retrieval** — BM25 index over extracted claims, used by Writer/QA/Verifier
- **Sandbox verification** — scripts generated and executed in subprocess sandbox with self-correction on failure
- **Data quality governance** — magic-number detection, boilerplate filtering, code density checks, low-quality source gating
- **LLM ecosystem** — `--llm none` (rule-based) / `openai_compat` (DeepSeek/OpenAI/OpenRouter/vLLM) / `anthropic`
- **Checkpoint/resume/replay** — interrupt and resume from state; replay trace with `--no-tools` dry-run
- **Tool registry** — schema-validated tools with permission governance, caching, and auditing
- **Evaluation** — 15+ metrics in `eval.json`; batch evaluation via `evalset/`

## Quick start

```bash
pip install -e .
pip install -e ".[dev]"      # pytest + ruff
pip install -e ".[quality]"  # trafilatura for better HTML extraction

# Offline demo (no network, no LLM)
researchops run "demo topic" --mode fast --allow-net false --llm none --sources demo

# arXiv-only (network required)
researchops run "quantum computing" --mode deep --allow-net true --sources arxiv --llm none

# Hybrid with DeepSeek
researchops run "量子计算" --mode deep --allow-net true --sources hybrid \
  --llm openai_compat --llm-base-url https://api.deepseek.com/v1 \
  --llm-model deepseek-chat --llm-provider-label deepseek \
  --llm-api-key "$LLM_API_KEY"

# Resume / Replay / Evaluate
researchops resume runs/<run_id>
researchops replay runs/<run_id> --no-tools --json
researchops eval runs/<run_id>

# Verify
researchops verify-run runs/<run_id>
researchops verify-repo
```

## Run workspace structure

```
runs/<run_id>/
  plan.json              # Research questions + outline
  sources.jsonl          # Collected sources (success only)
  failures.json          # Failed source attempts
  downloads/             # Raw HTML/PDF files
  notes/                 # Structured reading cards (JSON)
  code/                  # Verification scripts
  code/logs/             # Sandbox stdout/stderr
  artifacts/             # Verification outputs (JSON/CSV)
  report.md              # Generated report with citations
  report_index.json      # Sentence → claim/source mapping
  retrieval_index.json   # BM25 claim index metadata
  qa_report.json         # QA check results
  qa_conflicts.json      # Detected claim conflicts
  trace.jsonl            # Full event trace
  eval.json              # Evaluation metrics
  state.json             # Pipeline state (for resume)
  cache.json             # Tool cache
```

## CLI reference

| Command | Description |
|---------|-------------|
| `run` | Full pipeline execution |
| `resume` | Continue from checkpoint |
| `replay` | Replay trace events |
| `eval` | Recompute eval.json |
| `verify-run` | Check run artifact integrity |
| `verify-repo` | Check repository constraints |

### Key `run` options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | fast | `fast` or `deep` |
| `--sources` | hybrid | `demo`, `arxiv`, `web`, `hybrid` |
| `--retrieval` | bm25 | `none`, `bm25` |
| `--llm` | none | `none`, `openai_compat`, `anthropic` |
| `--allow-net` | true | Network access control |
| `--sandbox` | subprocess | `subprocess`, `docker` |

## Architecture

```
CLI → Orchestrator → Agents (Planner, Collector, Reader, Verifier, Writer, QA)
                  ↓                    ↓
           State Machine        Tool Registry (arxiv_search, fetch, parse, sandbox_exec, ...)
                  ↓                    ↓
           Checkpoint/Resume     Permission + Cache + Audit
                  ↓
           Retrieval (BM25) ← claims from Reader
                  ↓
           Writer/QA/Verifier consume ranked claims
```

## Evaluation

Each run outputs `eval.json` with metrics including:

- `citation_coverage` — ratio of paragraphs with citations
- `unsupported_claim_rate` — sentences without evidence backing
- `papers_per_rq` — arXiv papers per research question
- `reproduction_rate` — verification script success rate
- `section_nonempty_rate` — report sections with content
- `low_quality_source_rate` — sources flagged as low quality
- `conflict_count` — detected claim conflicts across sources

Batch evaluation: `python scripts/run_evalset.py` runs all topics in `evalset/topics.jsonl` and produces `runs_batch/aggregate_metrics.json`.

## Security

- **Network blocking**: `--allow-net false` blocks all network access via monkeypatch (socket, http.client, urllib, requests, httpx). Strategy: best-effort; see trace for details.
- **Sandbox**: subprocess with timeout and resource limits; Docker sandbox available as opt-in.
- **Tool governance**: all external capabilities go through ToolRegistry with permission checks and audit logging.
- **LLM keys**: never logged; resolved via `--llm-api-key` > env vars.

## Development

```bash
make dev          # Install with dev deps
make test         # Run pytest
make lint         # Run ruff
make fmt          # Auto-format
make verify       # Repo verification
make demo         # Quick offline demo
make evalset      # Batch evaluation
```

## Troubleshooting

**Report contains doctype/meta/css/js garbage**: Usually caused by failed source fetches (captcha pages, paywalls). Use `make verify-run RUN=runs/<id>` to diagnose. Consider using `--sources arxiv` for cleaner inputs.

**LLM not being used**: Check trace.jsonl for `llm.call` events. Verify API key is set. Run `python scripts/verify_llm_path.py runs/<id>` to audit.

## License

MIT
