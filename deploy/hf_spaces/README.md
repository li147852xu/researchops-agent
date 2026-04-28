---
title: ResearchOps
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Multi-agent research workflow with evidence-grounded reports
---

# ResearchOps — Live Demo

A reusable multi-agent workflow core for **General Research** and **Market Intelligence**, packaged as a single Gradio Space. The same codebase powers a CLI, a FastAPI service, and an MCP stdio server — this UI is just one front-end on top of the shared pipeline (plan → collect → read → verify → write → QA → eval, with a supervisor agent for rollback).

> Source code: https://github.com/Tiantanghuaxiao/researchops-ai

## Try it

The Space ships with three tabs:

| Tab | What it does |
|---|---|
| **General Research** | Pick a demo topic (or type your own), run a fast 7-stage pipeline, and read the cited Markdown report. |
| **Market Intelligence** | Same engine, configured with a ticker + analysis type to produce a source-grounded market memo. |
| **Architecture** | A static page describing the multi-agent pipeline, the App-as-config-layer pattern, and how to add a new vertical. |

Each run produces a numbered report, the agents' plan, the source list (deduped + bucketed), and an automated evaluation card (citation coverage, bucket coverage, relevance). All artifacts are written to `runs/<run_id>/` inside the container; they survive only for the lifetime of the Space replica.

## What you get without an API key

The Space defaults to `LLM_BACKEND=none`, which switches the pipeline into its **rule-based** mode: every agent uses deterministic templates instead of an LLM, the report is a fully-structured skeleton, and the entire UI — stages, tabs, evaluation panel — works end-to-end. This makes the demo reviewable in seconds, with zero configuration on your part.

## Bring your own key

To upgrade to a real LLM, configure these as **Secrets** in the Space settings (Settings → Variables and secrets). Reviewers without a key never see this — the rule-based mode keeps working.

| Variable | Required when | Notes |
|---|---|---|
| `LLM_BACKEND` | Switching off rule-based mode | `openai_compat` (DeepSeek / OpenRouter / vLLM) · `openai` · `anthropic` |
| `LLM_MODEL` | Optional | e.g. `deepseek-chat`, `gpt-4o-mini`, `claude-sonnet-4-20250514` |
| `LLM_BASE_URL` | `openai_compat` w/ non-OpenAI provider | e.g. `https://api.deepseek.com/v1` |
| `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Match the backend you chose | Standard provider keys |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | Optional | Mirrors every run as a trace + spans + generations to your Langfuse project |

A copy-paste reference of every variable lives in `.env.template`.

## How the container is built

This Space uses the **Docker SDK**. The `Dockerfile` at the Space root:

1. Starts from `python:3.11-slim` and creates a non-root `user` (uid 1000), per HF Spaces conventions.
2. Copies the entire ResearchOps source into `/home/user/app`.
3. Installs only the `[web]` extra (Gradio + transitive deps), keeping the image around 600 MB and well under the free-tier 16 GB RAM ceiling.
4. Boots the UI via `python deploy/hf_spaces/app.py`, which binds to `0.0.0.0:7860` so the Space can route traffic in.

The `embeddings` extra (sentence-transformers, ~1.3 GB model download at first use) is intentionally NOT installed, which keeps the demo snappy on free hardware. BM25 retrieval is the active default and is plenty for the demo workload.

## Deploying your own copy

See [`DEPLOY.md`](DEPLOY.md) for the five-step walkthrough (create Space → push → set Secrets → wait for build → verify), plus the local Docker smoke-test command.
