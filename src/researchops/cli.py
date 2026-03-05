from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import typer
from rich.console import Console

from researchops.config import RetrievalMode, RunConfig, RunMode, SandboxBackend, SourceStrategy

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
    sources: SourceStrategy = typer.Option(SourceStrategy.HYBRID, "--sources", help="Source strategy: demo, arxiv, web, hybrid"),
    retrieval: RetrievalMode = typer.Option(RetrievalMode.BM25, "--retrieval", help="Retrieval mode: none, bm25"),
    embedder: str = typer.Option("none", "--embedder", help="Embedder backend: none, openai_compat"),
    llm: str = typer.Option("none", "--llm", help="LLM backend: none, openai, openai_compat, anthropic"),
    llm_model: str = typer.Option("", "--llm-model", help="LLM model name"),
    llm_base_url: str = typer.Option("", "--llm-base-url", help="LLM API base URL"),
    llm_api_key: str = typer.Option("", "--llm-api-key", help="LLM API key (or use env var)"),
    llm_provider_label: str = typer.Option("", "--llm-provider-label", help="Provider label for trace/eval (e.g. deepseek, openrouter)"),
    llm_headers: str = typer.Option("", "--llm-headers", help="Extra request headers as JSON string"),
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

    if not net_enabled and sources not in (SourceStrategy.DEMO,):
        console.print(f"[yellow]allow-net=false: overriding sources={sources.value} -> demo[/]")
        sources = SourceStrategy.DEMO

    api_key = llm_api_key
    if not api_key:
        if llm in ("openai", "openai_compat"):
            for _env in ("OPENAI_API_KEY", "LLM_API_KEY", "DEEPSEEK_API_KEY"):
                api_key = os.environ.get(_env, "")
                if api_key:
                    break
        elif llm == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("LLM_API_KEY", "")

    if llm in ("openai", "openai_compat", "anthropic") and not api_key:
        console.print(f"[red]--llm {llm} requires an API key. Set --llm-api-key or the appropriate env var.[/]")
        console.print("[dim]Use --llm none for rule-based mode.[/]")
        raise SystemExit(1)

    if llm_headers:
        try:
            json.loads(llm_headers)
        except json.JSONDecodeError as exc:
            console.print("[red]--llm-headers must be valid JSON[/]")
            raise SystemExit(1) from exc

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
        sources=sources,
        retrieval=retrieval,
        embedder=embedder,
        llm=llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=api_key,
        llm_provider_label=llm_provider_label,
        llm_headers=llm_headers,
        seed=seed,
    )

    console.print(f"[bold]ResearchOps Agent v1.0.0[/] — run [cyan]{run_id}[/]")
    console.print(f"  Topic: {topic}")
    console.print(
        f"  Mode: {mode.value} | Net: {net_enabled} | Sandbox: {sandbox.value} | "
        f"Sources: {sources.value} | Retrieval: {retrieval.value}"
    )

    if llm != "none":
        label = llm_provider_label or "auto"
        base = llm_base_url or "(default)"
        model_name = llm_model or "(default)"
        has_key = "yes" if api_key else "NO KEY"
        console.print(f"  LLM provider={label} base_url={base} model={model_name} key={has_key}")
    else:
        console.print("  LLM disabled (rule-based mode)")
    console.print()

    orch = Orchestrator(config, run_dir)
    orch.run()

    console.print("\n[bold]Computing evaluation...[/]")
    eval_result = compute_eval(run_dir, config=config)
    console.print(f"  Citation coverage: {eval_result.citation_coverage:.1%}")
    console.print(f"  Source diversity: {eval_result.source_diversity}")
    console.print(f"  Reproduction rate: {eval_result.reproduction_rate:.1%}")
    console.print(f"  Unsupported claim rate: {eval_result.unsupported_claim_rate:.1%}")
    console.print(f"  Conflicts: {eval_result.conflict_count} | Refinements: {eval_result.plan_refinement_count}")
    console.print(f"  Tool calls: {eval_result.tool_calls} | Cache hit rate: {eval_result.cache_hit_rate:.1%}")
    console.print(f"  Latency: {eval_result.latency_sec}s | Artifacts: {eval_result.artifacts_count}")
    if eval_result.papers_per_rq > 0:
        console.print(f"  Papers/RQ: {eval_result.papers_per_rq:.1f} | Low-quality rate: {eval_result.low_quality_source_rate:.1%}")
    console.print(f"  Section non-empty rate: {eval_result.section_nonempty_rate:.1%}")


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
    console.print(f"  Conflicts: {result.conflict_count} | Refinements: {result.plan_refinement_count}")
    console.print(f"  Tool calls: {result.tool_calls} | Cache hit rate: {result.cache_hit_rate:.1%}")
    console.print(f"  Latency: {result.latency_sec}s | Steps: {result.steps}")
    console.print(f"\n[green]Written to {run_dir / 'eval.json'}[/]")


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


if __name__ == "__main__":
    app()
