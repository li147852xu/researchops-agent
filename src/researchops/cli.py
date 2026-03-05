from __future__ import annotations

import os
import uuid
from pathlib import Path

import typer
from rich.console import Console

from researchops.config import RunConfig, RunMode, SandboxBackend

app = typer.Typer(
    name="researchops",
    help="ResearchOps Agent — multi-agent research orchestration harness",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    topic: str = typer.Argument(..., help="Research topic to investigate"),
    mode: RunMode = typer.Option(RunMode.FAST, "--mode", "-m", help="Run mode: fast or deep"),
    checkpoint: Path | None = typer.Option(None, "--checkpoint", help="Checkpoint path"),
    budget: float = typer.Option(10.0, "--budget", help="Budget limit"),
    max_steps: int = typer.Option(50, "--max-steps", help="Maximum pipeline steps"),
    allow_net: str = typer.Option("true", "--allow-net", help="Allow network access (true/false)"),
    net_allowlist: str = typer.Option("", "--net-allowlist", help="Comma-separated allowed domains"),
    sandbox: SandboxBackend = typer.Option(
        SandboxBackend.SUBPROCESS, "--sandbox", help="Sandbox backend"
    ),
    llm: str = typer.Option("none", "--llm", help="LLM backend: none, openai, anthropic"),
    llm_model: str = typer.Option("", "--llm-model", help="LLM model name"),
    llm_base_url: str = typer.Option("", "--llm-base-url", help="LLM API base URL"),
    llm_api_key: str = typer.Option("", "--llm-api-key", help="LLM API key (or use env var)"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Run a full research pipeline on the given topic."""
    from researchops.evaluator import compute_eval
    from researchops.orchestrator import Orchestrator

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    net_enabled = allow_net.lower() in ("true", "1", "yes")
    domains = [d.strip() for d in net_allowlist.split(",") if d.strip()] if net_allowlist else []

    api_key = llm_api_key
    if not api_key and llm == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
    elif not api_key and llm == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    config = RunConfig(
        topic=topic,
        mode=mode,
        checkpoint=checkpoint,
        budget=budget,
        max_steps=max_steps,
        allow_net=net_enabled,
        net_allowlist=domains,
        sandbox=sandbox,
        run_dir=run_dir,
        llm=llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=api_key,
        seed=seed,
    )

    console.print(f"[bold]ResearchOps Agent[/] — run [cyan]{run_id}[/]")
    console.print(f"  Topic: {topic}")
    console.print(
        f"  Mode: {mode.value} | Net: {net_enabled} | Sandbox: {sandbox.value} | LLM: {llm}\n"
    )

    orch = Orchestrator(config, run_dir)
    orch.run()

    console.print("\n[bold]Computing evaluation...[/]")
    eval_result = compute_eval(run_dir, llm_enabled=llm != "none")
    console.print(f"  Citation coverage: {eval_result.citation_coverage:.1%}")
    console.print(f"  Source diversity: {eval_result.source_diversity}")
    console.print(f"  Reproduction rate: {eval_result.reproduction_rate:.1%}")
    console.print(f"  Unsupported claim rate: {eval_result.unsupported_claim_rate:.1%}")
    console.print(f"  Tool calls: {eval_result.tool_calls} | Cache hit rate: {eval_result.cache_hit_rate:.1%}")
    console.print(f"  Latency: {eval_result.latency_sec}s")


@app.command()
def resume(
    run_dir: Path = typer.Argument(..., help="Path to existing run directory"),
) -> None:
    """Resume an interrupted run from its last checkpoint."""
    from researchops.orchestrator import resume_run

    console.print(f"[bold]Resuming[/] from {run_dir}")
    resume_run(run_dir)

    from researchops.evaluator import compute_eval

    compute_eval(run_dir)


@app.command()
def replay(
    run_dir: Path = typer.Argument(..., help="Path to run directory to replay"),
    from_step: int = typer.Option(0, "--from-step", help="Start replay from this step"),
    no_tools: bool = typer.Option(False, "--no-tools", help="Dry-run: show what would execute"),
    json_out: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Replay a completed run's trace for inspection."""
    from researchops.orchestrator import replay_run

    replay_run(run_dir, from_step=from_step, no_tools=no_tools, json_output=json_out)


@app.command(name="eval")
def eval_cmd(
    run_dir: Path = typer.Argument(..., help="Path to run directory to evaluate"),
) -> None:
    """Recompute eval.json for a completed run."""
    from researchops.evaluator import compute_eval

    console.print(f"[bold]Evaluating[/] {run_dir}")
    result = compute_eval(run_dir)
    console.print(f"  Citation coverage: {result.citation_coverage:.1%}")
    console.print(f"  Source diversity: {result.source_diversity}")
    console.print(f"  Reproduction rate: {result.reproduction_rate:.1%}")
    console.print(f"  Unsupported claim rate: {result.unsupported_claim_rate:.1%}")
    console.print(f"  Tool calls: {result.tool_calls} | Cache hit rate: {result.cache_hit_rate:.1%}")
    console.print(f"  Latency: {result.latency_sec}s | Steps: {result.steps}")
    console.print(f"\n[green]Written to {run_dir / 'eval.json'}[/]")


if __name__ == "__main__":
    app()
