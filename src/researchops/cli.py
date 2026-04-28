"""Unified CLI — single entry point for all apps on the multi-agent platform."""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="researchops",
    help="ResearchOps — reusable multi-agent workflow core for research and decision support",
    add_completion=False,
)
console = Console()


def _resolve_api_key(llm: str, llm_api_key: str) -> str:
    if llm_api_key:
        return llm_api_key
    if llm in ("openai", "openai_compat"):
        for env in ("OPENAI_API_KEY", "LLM_API_KEY", "DEEPSEEK_API_KEY"):
            key = os.environ.get(env, "")
            if key:
                return key
    elif llm == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("LLM_API_KEY", "")
    return ""


# ── Main run command ───────────────────────────────────────────────────

@app.command()
def run(
    topic: str = typer.Argument(..., help="Research topic or query"),
    app_name: str = typer.Option("research", "--app", "-a", help="App to run (research, market)"),
    mode: str = typer.Option("fast", "--mode", "-m", help="Run mode: fast or deep"),
    ticker: str = typer.Option("", "--ticker", help="Stock ticker (market app)"),
    analysis_type: str = typer.Option("fundamental", "--analysis-type", help="Analysis type (market app)"),
    allow_net: str = typer.Option("true", "--allow-net", help="Allow network access"),
    sources: str = typer.Option("hybrid", "--sources", help="Source strategy"),
    retrieval: str = typer.Option("bm25", "--retrieval", help="Retrieval mode: none, bm25, hybrid"),
    embedder_model: str = typer.Option("all-MiniLM-L6-v2", "--embedder-model", help="Embedding model"),
    graph: bool = typer.Option(True, "--graph/--no-graph", help="Use LangGraph orchestrator"),
    judge: bool = typer.Option(False, "--judge", help="Enable LLM-as-Judge evaluation"),
    llm: str = typer.Option("openai_compat", "--llm", help="LLM backend: openai, openai_compat, anthropic, none"),
    llm_model: str = typer.Option("", "--llm-model", help="LLM model name"),
    llm_base_url: str = typer.Option("", "--llm-base-url", help="LLM API base URL"),
    llm_api_key: str = typer.Option("", "--llm-api-key", help="LLM API key (or use env var)"),
    llm_provider_label: str = typer.Option("", "--llm-provider-label", help="Provider label for trace"),
    llm_headers: str = typer.Option("", "--llm-headers", help="Extra request headers as JSON"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Run a multi-agent pipeline on the given topic."""
    from researchops.apps.registry import get_app
    from researchops.core.orchestration.engine import GraphOrchestrator, Orchestrator
    from researchops.core.pipeline import build_initial_state, build_pipeline_graph, clear_ctx_cache

    spec = get_app(app_name)

    net_enabled = allow_net.lower() in ("true", "1", "yes")
    api_key = _resolve_api_key(llm, llm_api_key)

    if llm in ("openai", "openai_compat", "anthropic") and not api_key:
        console.print(f"[red]--llm {llm} requires an API key. Set --llm-api-key or the appropriate env var.[/]")
        raise SystemExit(1)

    if llm_headers:
        try:
            json.loads(llm_headers)
        except json.JSONDecodeError as exc:
            console.print("[red]--llm-headers must be valid JSON[/]")
            raise SystemExit(1) from exc

    config_data: dict = {
        "topic": topic, "mode": mode, "allow_net": net_enabled,
        "sources": sources, "retrieval": retrieval,
        "embedder_model": embedder_model, "use_graph": graph,
        "judge": judge, "llm": llm, "llm_model": llm_model,
        "llm_base_url": llm_base_url, "llm_api_key": api_key,
        "llm_provider_label": llm_provider_label, "llm_headers": llm_headers,
        "seed": seed,
    }
    if app_name == "market":
        config_data["ticker"] = ticker
        config_data["analysis_type"] = analysis_type

    run_id = f"{app_name}_{uuid.uuid4().hex[:8]}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_data["run_dir"] = run_dir

    config = spec.config_class.model_validate(config_data)

    console.print(f"[bold]ResearchOps v{__import__('researchops').__version__}[/] — [{spec.display_name}] run [cyan]{run_id}[/]")
    console.print(f"  Topic: {topic}")
    orch_label = "LangGraph" if graph else "sequential"
    console.print(
        f"  Mode: {mode} | Net: {net_enabled} | Sources: {sources} | "
        f"Retrieval: {retrieval} | Orchestrator: {orch_label}"
    )
    if llm != "none":
        console.print(f"  LLM: {llm} model={llm_model or '(default)'} key={'yes' if api_key else 'NO KEY'}")
    console.print()

    if graph:
        orch = GraphOrchestrator(spec, config, run_dir)
    else:
        orch = Orchestrator(spec, config, run_dir)
    orch.run()

    if spec.compute_eval and not graph:
        console.print("\n[bold]Computing evaluation...[/]")
        eval_result = spec.compute_eval(run_dir, config=config)
        for field_name in ("citation_coverage", "bucket_coverage_rate", "relevance_avg"):
            val = getattr(eval_result, field_name, None)
            if val is not None:
                console.print(f"  {field_name}: {val:.1%}" if val <= 1 else f"  {field_name}: {val}")


# ── Utility commands ───────────────────────────────────────────────────

@app.command(name="list-apps")
def list_apps_cmd() -> None:
    """List all registered apps on the platform."""
    from researchops.apps.registry import list_apps

    table = Table(title="Registered Apps", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Display Name", style="bold")
    table.add_column("Description")

    for spec in list_apps():
        table.add_row(spec.name, spec.display_name, spec.description)

    console.print(table)


@app.command(name="inspect-app")
def inspect_app_cmd(
    name: str = typer.Argument(..., help="App name to inspect"),
) -> None:
    """Show detailed specification for an app."""
    from researchops.apps.registry import get_app

    spec = get_app(name)
    console.print(f"[bold]{spec.display_name}[/] ({spec.name})")
    console.print(f"  Description: {spec.description}")
    console.print(f"  Config class: {spec.config_class.__name__}")
    console.print(f"  Tool registrar: {spec.register_tools.__module__}.{spec.register_tools.__name__}")
    console.print(f"  Custom planner: {'yes' if spec.custom_planner else 'no (uses default PlannerAgent)'}")
    console.print(f"  Evaluator: {'yes' if spec.compute_eval else 'no'}")
    if spec.extra_cli_options:
        console.print(f"  Extra CLI options: {list(spec.extra_cli_options.keys())}")


@app.command(name="inspect-config")
def inspect_config_cmd(
    name: str = typer.Argument(..., help="App name"),
) -> None:
    """Show default configuration schema for an app."""
    from researchops.apps.registry import get_app

    spec = get_app(name)
    schema = spec.config_class.model_json_schema()
    console.print_json(json.dumps(schema, indent=2))


@app.command()
def resume(
    run_dir: Path = typer.Argument(..., help="Path to existing run directory"),
) -> None:
    """Resume an interrupted run from its last checkpoint."""
    from researchops.core.orchestration.engine import resume_run

    console.print(f"[bold]Resuming[/] from {run_dir}")
    resume_run(run_dir)


@app.command()
def replay(
    run_dir: Path = typer.Argument(..., help="Path to run directory to replay"),
    from_step: int = typer.Option(0, "--from-step", help="Start replay from this step"),
    no_tools: bool = typer.Option(False, "--no-tools", help="Dry-run: show what would execute"),
    json_out: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Replay a completed run's trace for inspection."""
    from researchops.core.replay import replay_run

    replay_run(run_dir, from_step=from_step, no_tools=no_tools, json_output=json_out)


@app.command(name="eval")
def eval_cmd(
    run_dir: Path = typer.Argument(..., help="Path to run directory to evaluate"),
    app_name: str = typer.Option("research", "--app", "-a", help="App that produced the run"),
) -> None:
    """Recompute evaluation metrics for a completed run."""
    from researchops.apps.registry import get_app

    spec = get_app(app_name)
    console.print(f"[bold]Evaluating[/] {run_dir} (app={app_name})")
    if spec.compute_eval:
        result = spec.compute_eval(run_dir)
        console.print_json(result.model_dump_json(indent=2))
    else:
        console.print("[yellow]No evaluator registered for this app.[/]")


@app.command(name="verify-run")
def verify_run_cmd(
    run_dir: Path = typer.Argument(..., help="Path to run directory to verify"),
) -> None:
    """Verify integrity of a completed run's artifacts."""
    import subprocess

    script = Path("scripts/verify_run_integrity.py")
    if not script.exists():
        console.print("[red]scripts/verify_run_integrity.py not found[/]")
        raise SystemExit(1)
    result = subprocess.run([sys.executable, str(script), str(run_dir)])
    raise SystemExit(result.returncode)


@app.command(name="verify-repo")
def verify_repo_cmd() -> None:
    """Verify repository structure and constraints."""
    import subprocess

    script = Path("scripts/verify_repo.py")
    if not script.exists():
        console.print("[red]scripts/verify_repo.py not found[/]")
        raise SystemExit(1)
    result = subprocess.run([sys.executable, str(script)])
    raise SystemExit(result.returncode)


@app.command()
def web(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
    port: int = typer.Option(7860, "--port", help="Port to bind"),
    share: bool = typer.Option(False, "--share", help="Create public Gradio link"),
) -> None:
    """Launch the unified web UI (all apps in one interface)."""
    from researchops.web.app import launch

    local_url = f"http://localhost:{port}" if host in ("0.0.0.0", "") else f"http://{host}:{port}"
    console.print("[bold]ResearchOps Web UI[/]")
    console.print(f"  [green]Local URL :[/]  {local_url}")
    if host == "0.0.0.0":
        console.print(f"  [dim]Network   :  http://0.0.0.0:{port}  (all interfaces)[/]")
    console.print(f"  [dim]Tip: open [bold]{local_url}[/bold] in your browser[/]")
    launch(host=host, port=port, share=share)


@app.command()
def api(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
    port: int = typer.Option(8000, "--port", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
) -> None:
    """Launch the unified REST API (FastAPI + Uvicorn)."""
    import uvicorn

    from researchops.api.app import create_api

    console.print(f"[bold]ResearchOps API[/] — http://{host}:{port}")
    console.print(f"  Docs: http://{host}:{port}/docs")
    api_app = create_api()
    uvicorn.run(api_app, host=host, port=port, reload=reload)


@app.command(name="mcp")
def mcp_cmd(
    runs_dir: Path = typer.Option(
        Path("runs"),
        "--runs-dir",
        help="Directory containing run artifacts to expose as MCP resources",
    ),
) -> None:
    """Run ResearchOps as a Model Context Protocol stdio server.

    Exposes every registered tool (web_search, arxiv_search, parse, sandbox_exec, ...)
    and every runs/<id>/{plan.json,sources.jsonl,report.md} artifact to any MCP
    client (Claude Desktop, Cursor, Cline). Requires the `mcp` extra:
    `pip install -e ".[mcp]"`.
    """
    try:
        from researchops.mcp.server import run as run_mcp
    except ImportError as exc:  # pragma: no cover - extra not installed
        console.print(
            "[red]MCP SDK not installed.[/] Install with: "
            "[bold]pip install -e \".[mcp]\"[/]"
        )
        raise SystemExit(1) from exc
    run_mcp(runs_dir=runs_dir)


if __name__ == "__main__":
    app()
