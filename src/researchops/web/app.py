"""ResearchOps Web UI — unified Gradio interface for all apps on the multi-agent platform."""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Generator

from dotenv import load_dotenv

load_dotenv()

import gradio as gr

from researchops.web.components import (
    STAGES,
    detect_stages_from_trace,
    resolve_api_key,
    stage_html,
)

_RESEARCH_TOPIC_DEMOS = [
    "transformer architecture evolution",
    "quantum computing applications",
    "CRISPR gene editing advances",
    "renewable energy storage",
    "large language model alignment",
    "brain-computer interfaces",
]

_MARKET_TOPIC_DEMOS = [
    "NVDA competitive position in AI chips",
    "Apple financial health and growth outlook",
    "Tesla risk assessment and regulatory challenges",
    "Microsoft cloud computing market share",
    "Amazon e-commerce and AWS revenue analysis",
]

_MARKET_TICKER_DEMOS = ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN"]

_CUSTOM_CSS = """
.gradio-container {
    max-width: 1280px !important;
    margin: auto !important;
}
.header-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 16px;
    color: white;
}
.header-banner h1 {
    color: white !important;
    font-size: 28px !important;
    margin-bottom: 4px !important;
}
.header-banner p {
    color: #b0bec5 !important;
    font-size: 14px !important;
    margin: 0 !important;
}
.arch-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 12px;
    color: #e0e0e0;
    margin-right: 6px;
    margin-top: 8px;
}
.stage-indicator {
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    display: inline-block;
    margin: 3px 4px;
}
.stage-active {
    background: #e3f2fd;
    color: #1565c0;
    border: 1px solid #90caf9;
}
.stage-done {
    background: #e8f5e9;
    color: #2e7d32;
    border: 1px solid #a5d6a7;
}
.stage-pending {
    background: #f5f5f5;
    color: #9e9e9e;
    border: 1px solid #e0e0e0;
}
.metric-card {
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}
.metric-card h3 {
    font-size: 24px !important;
    margin: 0 !important;
    color: #1565c0;
}
.metric-card p {
    font-size: 12px !important;
    color: #757575;
    margin: 4px 0 0 0 !important;
}
footer {
    border-top: 1px solid #e0e0e0;
    margin-top: 24px;
    padding-top: 12px;
}
"""


def _header_html() -> str:
    version = "1.0.0"
    try:
        from researchops import __version__
        version = __version__
    except ImportError:
        pass

    badges = "".join(
        f'<span class="arch-badge">{b}</span>'
        for b in [
            "Multi-Agent Core",
            "LangGraph",
            "Config-Driven Apps",
            "Evidence-Grounded",
            "Source-Traceable",
        ]
    )
    return (
        '<div class="header-banner">'
        f"<h1>ResearchOps <small>v{version}</small></h1>"
        "<p>Reusable multi-agent workflow core for research and decision support</p>"
        f"<div style='margin-top:10px'>{badges}</div>"
        "</div>"
    )


def _render_citations(report: str, sources: list[dict]) -> str:
    """Transform [@source_id] markers into numbered references for clean display."""
    source_map = {s.get("source_id", ""): s for s in sources if s.get("source_id")}

    cite_ids: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"\[@(\w+)\]", report):
        sid = m.group(1)
        if sid not in seen:
            cite_ids.append(sid)
            seen.add(sid)

    if not cite_ids:
        return report

    num_map = {sid: i + 1 for i, sid in enumerate(cite_ids)}

    ref_re = re.compile(r"^##\s+(?:References|参考来源)\s*$", re.MULTILINE)
    ref_match = ref_re.search(report)
    body = report[: ref_match.start()].rstrip() if ref_match else report.rstrip()

    def _replace(m: re.Match) -> str:
        n = num_map.get(m.group(1))
        return f"**[{n}]**" if n else m.group(0)

    body = re.sub(r"\[@(\w+)\]", _replace, body)

    ref_lines = ["\n\n---\n\n## References\n"]
    for sid in cite_ids:
        n = num_map[sid]
        src = source_map.get(sid, {})
        title = src.get("title") or sid
        domain = src.get("domain", "")
        url = src.get("url", "")
        if url and url.startswith("http"):
            ref_lines.append(f"**[{n}]** [{title}]({url}) ({domain})\n")
        else:
            suffix = f" ({domain})" if domain else ""
            ref_lines.append(f"**[{n}]** {title}{suffix}\n")

    return body + "\n".join(ref_lines)


def _run_pipeline(
    app_name: str,
    topic: str,
    mode: str,
    sources: str,
    progress_log: list[str],
    status: dict,
    extra_config: dict | None = None,
) -> dict[str, Any]:
    """Run any app's pipeline — shared implementation for all tabs."""
    from researchops.apps.registry import get_app
    from researchops.core.pipeline import build_initial_state, build_pipeline_graph, clear_ctx_cache

    spec = get_app(app_name)

    run_id = f"{app_name}_{uuid.uuid4().hex[:8]}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status["run_dir"] = str(run_dir)

    llm = os.environ.get("LLM_BACKEND", "openai_compat")
    model = os.environ.get("LLM_MODEL", "")
    base_url = os.environ.get("LLM_BASE_URL", "")
    key = resolve_api_key(llm, "")

    config_data: dict[str, Any] = {
        "topic": topic, "mode": mode, "allow_net": True,
        "sources": sources, "run_dir": run_dir,
        "llm": llm if llm != "none" else "openai_compat",
        "llm_model": model, "llm_base_url": base_url, "llm_api_key": key,
    }
    if extra_config:
        config_data.update(extra_config)

    config = spec.config_class.model_validate(config_data)

    progress_log.append(f"[ResearchOps] Starting {spec.display_name} pipeline")
    progress_log.append(f"  Topic: {topic}")
    progress_log.append(f"  Run ID: {run_id} | Mode: {mode} | Sources: {sources}")
    progress_log.append(f"  LLM: {llm} | Model: {model or '(default)'}")
    progress_log.append("")

    clear_ctx_cache()
    graph = build_pipeline_graph(spec)
    initial_state = build_initial_state(run_id, run_dir, topic, config)

    try:
        graph.invoke(initial_state, {"recursion_limit": 100})
        progress_log.append("Pipeline complete. Computing evaluation...")
        if spec.compute_eval:
            eval_result = spec.compute_eval(run_dir, config=config)
            for key_name in ("citation_coverage", "bucket_coverage_rate", "relevance_avg"):
                val = getattr(eval_result, key_name, None)
                if val is not None:
                    progress_log.append(f"  {key_name}: {val:.1%}" if val <= 1 else f"  {key_name}: {val}")
        progress_log.append("")
        progress_log.append(f"Run complete -> {run_dir}")
    except Exception as exc:
        progress_log.append(f"Error: {exc}")
        return {"error": str(exc), "run_dir": str(run_dir)}

    report_path = run_dir / "report.md"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else "No report generated."

    eval_path = run_dir / "eval.json"
    eval_data = {}
    if eval_path.exists():
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))

    plan_data = {}
    plan_path = run_dir / "plan.json"
    if plan_path.exists():
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))

    sources_data: list[dict] = []
    sources_path = run_dir / "sources.jsonl"
    if sources_path.exists():
        for line in sources_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                with contextlib.suppress(Exception):
                    sources_data.append(json.loads(line))

    report = _render_citations(report, sources_data)

    return {
        "report": report, "eval": eval_data, "run_dir": str(run_dir),
        "plan": plan_data, "sources": sources_data,
    }


def _build_app_tab(
    app_name: str,
    display_name: str,
    description: str,
    topic_choices: list[str] | None = None,
    extra_fields: dict | None = None,
) -> None:
    """Build a standardized app tab with shared pipeline UI components."""
    gr.Markdown(f"### {display_name}\n{description}")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Configuration")
            topic_input = gr.Dropdown(
                choices=topic_choices or [],
                allow_custom_value=True,
                label="Topic / Query",
                info="Select a demo topic or type your own",
            )

            extra_inputs = {}
            if extra_fields:
                for field_name, field_spec in extra_fields.items():
                    ftype = field_spec.get("type", "text")
                    if ftype == "dropdown":
                        extra_inputs[field_name] = gr.Dropdown(
                            choices=field_spec["choices"],
                            value=field_spec.get("default", field_spec["choices"][0]),
                            label=field_spec.get("label", field_name),
                            scale=field_spec.get("scale", 1),
                        )
                    elif ftype == "combo":
                        extra_inputs[field_name] = gr.Dropdown(
                            choices=field_spec.get("choices", []),
                            allow_custom_value=True,
                            label=field_spec.get("label", field_name),
                            info=field_spec.get("info", "Select or type your own"),
                            scale=field_spec.get("scale", 1),
                        )
                    else:
                        extra_inputs[field_name] = gr.Textbox(
                            label=field_spec.get("label", field_name),
                            placeholder=field_spec.get("placeholder", ""),
                            scale=field_spec.get("scale", 1),
                        )

            mode_input = gr.Dropdown(
                choices=["fast", "deep"], value="fast", label="Mode",
                info="Fast: quick survey. Deep: thorough with more sources.",
            )
            sources_input = gr.Dropdown(
                choices=["arxiv", "web", "hybrid"], value="hybrid",
                label="Source Strategy",
                info="Hybrid: arXiv + web + Wikipedia combined.",
            )

            run_btn = gr.Button(f"Run {display_name}", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.Markdown("#### Results")
            stage_display = gr.HTML(value=stage_html([], ""), label="Pipeline Stages")

            with gr.Tabs():
                with gr.Tab("Report"):
                    report_output = gr.Markdown(label="Generated Report")
                with gr.Tab("Plan"):
                    plan_output = gr.JSON(label="Plan")
                with gr.Tab("Sources"):
                    sources_output = gr.JSON(label="Sources")
                with gr.Tab("Evaluation"):
                    eval_output = gr.JSON(label="Metrics")
                with gr.Tab("Log"):
                    progress_output = gr.Textbox(label="Pipeline Log", lines=12, interactive=False)

    all_inputs = [topic_input]
    extra_keys = list(extra_inputs.keys()) if extra_inputs else []
    all_inputs.extend(extra_inputs[k] for k in extra_keys)
    all_inputs.extend([mode_input, sources_input])

    def run_handler(*args) -> Generator:
        n_extra = len(extra_keys)
        topic_val = args[0]
        extra_vals = args[1: 1 + n_extra]
        mode_val, sources_val = args[1 + n_extra:]

        if not topic_val or not str(topic_val).strip():
            yield stage_html([], ""), "Please enter a topic.", {}, [], {}, "No topic provided."
            return

        extra_config = dict(zip(extra_keys, extra_vals)) if extra_keys else None
        progress_log: list[str] = []
        result: dict[str, Any] = {}
        status: dict[str, Any] = {"run_dir": None}

        def worker():
            nonlocal result
            result = _run_pipeline(
                app_name, str(topic_val).strip(), mode_val, sources_val,
                progress_log, status, extra_config,
            )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while thread.is_alive():
            time.sleep(0.8)
            run_dir_str = status.get("run_dir")
            if run_dir_str:
                completed, current, trace_lines = detect_stages_from_trace(Path(run_dir_str))
            else:
                completed, current, trace_lines = [], "PLAN", []
            log_text = "\n".join(progress_log + ["", "--- Pipeline Trace ---"] + trace_lines)
            yield stage_html(completed, current), gr.skip(), gr.skip(), gr.skip(), gr.skip(), log_text

        thread.join(timeout=10)

        run_dir_str = status.get("run_dir")
        trace_lines = detect_stages_from_trace(Path(run_dir_str))[2] if run_dir_str else []
        log_text = "\n".join(progress_log + ["", "--- Pipeline Trace ---"] + trace_lines)

        if "error" in result:
            yield stage_html(["PLAN"], ""), f"**Error**: {result.get('error')}", {}, [], {}, log_text
            return

        yield (
            stage_html(list(STAGES), ""),
            result.get("report", ""),
            result.get("plan", {}),
            result.get("sources", []),
            result.get("eval", {}),
            log_text,
        )

    run_btn.click(
        fn=run_handler,
        inputs=all_inputs,
        outputs=[stage_display, report_output, plan_output, sources_output, eval_output, progress_output],
    )


def _build_architecture_tab() -> None:
    gr.Markdown(
        "## Platform Architecture\n\n"
        "ResearchOps is a **reusable multi-agent workflow core**.  Apps are thin "
        "configuration layers that plug into the shared pipeline.\n\n"
        "### Multi-Agent Pipeline (shared by all apps)\n\n"
        "```\n"
        "PLAN --> COLLECT --> READ --> VERIFY --> WRITE --> QA --> EVAL\n"
        "  ^                    |                    |           |\n"
        "  |                    +-(coverage low)-----+           |\n"
        "  +---- Supervisor (rollback / retry) <----------------+\n"
        "```\n\n"
        "### How Apps Work\n\n"
        "Each app provides:\n"
        "- **Config** — domain-specific fields (e.g. `ticker` for market)\n"
        "- **Prompts** — task-specific prompt templates\n"
        "- **Tools** — tool registration policy\n"
        "- **Evaluator** — domain-specific quality metrics\n\n"
        "Everything else (agents, orchestration, state, checkpoint, retrieval, "
        "sandbox, tracing) comes from the **shared core**.\n\n"
        "### Adding a New App\n\n"
        "1. Create `apps/myapp/config.py` extending `BaseAppConfig`\n"
        "2. Create `apps/myapp/prompts.py` with domain prompts\n"
        "3. Create `apps/myapp/__init__.py` registering an `AppSpec`\n"
        "4. Run: `researchops run 'my query' --app myapp`\n\n"
        "### Technology Stack\n\n"
        "| Component | Technology |\n"
        "|-----------|------------|\n"
        "| Orchestration | LangGraph (StateGraph + conditional edges) |\n"
        "| API / Service | FastAPI + Uvicorn |\n"
        "| CLI | Typer + Rich |\n"
        "| Data Models | Pydantic v2 |\n"
        "| HTTP | httpx |\n"
        "| Retrieval | BM25 + SentenceTransformers + RRF fusion |\n"
        "| Persistence | SQLite + SQLAlchemy |\n"
        "| Testing | pytest |\n"
        "| LLM | OpenAI-compatible, Anthropic, rule-based fallback |\n"
    )


def create_app() -> gr.Blocks:
    """Create the unified Gradio web UI."""
    with gr.Blocks(
        title="ResearchOps — Multi-Agent Workflow Platform",
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ),
        css=_CUSTOM_CSS,
    ) as demo:
        gr.HTML(_header_html())

        with gr.Tabs():
            with gr.Tab("General Research", id="research"):
                _build_app_tab(
                    app_name="research",
                    display_name="General Research",
                    description=(
                        "Evidence-backed research and report generation across any domain — "
                        "technology, policy, science, industry trends, and more."
                    ),
                    topic_choices=_RESEARCH_TOPIC_DEMOS,
                )

            with gr.Tab("Market Intelligence", id="market"):
                _build_app_tab(
                    app_name="market",
                    display_name="Market Intelligence",
                    description=(
                        "Source-grounded financial research — company analysis, sector intelligence, "
                        "competitive landscape, risk assessment, and structured market memos."
                    ),
                    topic_choices=_MARKET_TOPIC_DEMOS,
                    extra_fields={
                        "ticker": {
                            "type": "combo",
                            "label": "Ticker Symbol",
                            "choices": _MARKET_TICKER_DEMOS,
                            "info": "Select a demo ticker or type your own",
                            "scale": 1,
                        },
                        "analysis_type": {
                            "type": "dropdown",
                            "label": "Analysis Type",
                            "choices": ["fundamental", "competitive", "risk", "technical"],
                            "default": "fundamental",
                            "scale": 1,
                        },
                    },
                )

            with gr.Tab("Architecture", id="arch"):
                _build_architecture_tab()

        gr.HTML(
            '<footer style="text-align:center;color:#9e9e9e;font-size:12px">'
            "ResearchOps — Reusable multi-agent workflow core. "
            "Core: orchestration, agents, tools, sandbox, checkpoint, evaluation. "
            "Apps: configuration-driven thin layers."
            "</footer>"
        )

    demo.queue()
    return demo


def launch(host: str = "0.0.0.0", port: int = 7860, share: bool = False) -> None:
    """Launch the web UI."""
    for env_key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(env_key, "")
        loopback = "localhost,127.0.0.1,0.0.0.0"
        if existing:
            new_val = ",".join(
                {v.strip() for v in existing.split(",") if v.strip()} | set(loopback.split(","))
            )
        else:
            new_val = loopback
        os.environ[env_key] = new_val

    demo = create_app()
    bind_host = "127.0.0.1" if host == "0.0.0.0" else host
    demo.launch(server_name=bind_host, server_port=port, share=share)


if __name__ == "__main__":
    launch()
