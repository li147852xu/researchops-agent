# Changelog

## v1.0.0 — Initial Release

Local-first AI workflow platform for research, market intelligence, and structured analytics.

### Core
- Multi-agent pipeline with seven specialized agents (Planner, Collector, Reader, Verifier, Writer, QA, Supervisor) orchestrated via LangGraph
- Supervisor-driven rollback and quality gating with LLM-assisted remediation
- Configuration-driven app architecture — new domains require only config, prompts, tools, and evaluator
- Evidence protocol with source-traceable claims and explicit support status
- Hybrid retrieval (BM25 + SentenceTransformers + Reciprocal Rank Fusion)
- Tool registry with schema validation, permissions, caching, and audit logging
- Checkpoint, resume, and replay support
- Evaluation harness with core metrics and app-specific evaluators

### Apps
- **General Research**: topic research, literature surveys, technology analysis
- **Market Intelligence**: company analysis, sector intelligence, competitive landscape

### Interfaces
- Gradio Web UI with demo examples, pipeline progress visualization, and auto-loaded LLM config from `.env`
- Typer CLI with full pipeline control
- FastAPI REST API

### LLM Support
- OpenAI-compatible (DeepSeek, OpenRouter, vLLM, Ollama)
- OpenAI direct
- Anthropic (Claude)
- Rule-based fallback for offline development
