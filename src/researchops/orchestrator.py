from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console

from researchops.agents.base import AgentBase, RunContext
from researchops.agents.collector import CollectorAgent
from researchops.agents.planner import PlannerAgent
from researchops.agents.qa import QAAgent
from researchops.agents.reader import ReaderAgent
from researchops.agents.verifier import VerifierAgent
from researchops.agents.writer import WriterAgent
from researchops.checkpoint import advance_stage, load_state, save_state, should_skip
from researchops.config import RunConfig, SandboxBackend
from researchops.models import STAGE_ORDER, Stage, StateSnapshot
from researchops.reasoning import create_reasoner
from researchops.registry.builtin import register_builtin_tools
from researchops.registry.manager import ToolRegistry
from researchops.sandbox.proc import SubprocessSandbox
from researchops.trace import TraceLogger

console = Console()

_STAGE_AGENTS: dict[Stage, type[AgentBase]] = {
    Stage.PLAN: PlannerAgent,
    Stage.COLLECT: CollectorAgent,
    Stage.READ: ReaderAgent,
    Stage.VERIFY: VerifierAgent,
    Stage.WRITE: WriterAgent,
    Stage.QA: QAAgent,
}


class Orchestrator:
    def __init__(self, config: RunConfig, run_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.trace = TraceLogger(run_dir / "trace.jsonl")
        self.registry = self._build_registry()
        self.sandbox = self._build_sandbox()
        self.reasoner = create_reasoner(config)
        self.state = self._init_state()

    def _build_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        registry.set_persistent_cache_path(self.run_dir / "cache.json")

        perms: set[str] = {"sandbox"}
        if self.config.allow_net:
            perms.add("net")
        registry.grant_permissions(perms)

        self.trace.log(
            stage="ORCHESTRATOR",
            action="registry_init",
            meta={
                "granted_permissions": list(perms),
                "sandbox_strategy": "best_effort" if not self.config.allow_net else "open",
            },
        )
        return registry

    def _build_sandbox(self) -> SubprocessSandbox:
        if self.config.sandbox == SandboxBackend.DOCKER:
            console.print("[yellow]Docker sandbox not implemented, falling back to subprocess[/]")
        return SubprocessSandbox()

    def _init_state(self) -> StateSnapshot:
        existing = load_state(self.run_dir)
        if existing is not None:
            return existing
        state = StateSnapshot(
            run_id=self.run_dir.name,
            config_snapshot=self.config.model_dump(mode="json"),
        )
        save_state(state, self.run_dir)
        return state

    def _make_context(self) -> RunContext:
        return RunContext(
            run_dir=self.run_dir,
            config=self.config,
            state=self.state,
            registry=self.registry,
            trace=self.trace,
            sandbox=self.sandbox,
            reasoner=self.reasoner,
        )

    def run(self) -> None:
        self._ensure_dirs()
        t0 = time.monotonic()
        self.trace.log(stage="ORCHESTRATOR", action="run_start", input_summary=self.config.topic)

        pipeline = [s for s in STAGE_ORDER if s != Stage.DONE]

        for stage in pipeline:
            if should_skip(self.state, stage):
                console.print(f"  [dim]Skipping {stage.value} (already completed)[/]")
                continue

            if self.state.step >= self.config.max_steps:
                console.print("[red]Max steps reached, stopping[/]")
                break

            self.state.stage = stage
            save_state(self.state, self.run_dir)

            agent_cls = _STAGE_AGENTS[stage]
            agent = agent_cls()
            ctx = self._make_context()

            console.print(f"  [bold cyan]▶ {stage.value}[/] ({agent.name})")

            try:
                result = agent.execute(ctx)
            except Exception as exc:
                self.trace.log(
                    stage=stage.value,
                    agent=agent.name,
                    action="error",
                    error=str(exc),
                )
                console.print(f"  [red]Error in {stage.value}: {exc}[/]")
                save_state(self.state, self.run_dir)
                raise

            if result.rollback_to is not None:
                rollback_target = result.rollback_to
                console.print(
                    f"  [yellow]Rollback → {rollback_target.value}: {result.message}[/]"
                )
                self.trace.log(
                    stage=stage.value,
                    agent=agent.name,
                    action="rollback",
                    output_summary=f"Rolling back to {rollback_target.value}",
                )
                self.state.stage = rollback_target
                if rollback_target in self.state.completed_stages:
                    self.state.completed_stages.remove(rollback_target)
                for s in STAGE_ORDER:
                    if STAGE_ORDER.index(s) > STAGE_ORDER.index(rollback_target) and s in self.state.completed_stages:
                        self.state.completed_stages.remove(s)
                save_state(self.state, self.run_dir)
                return self.run()

            next_idx = STAGE_ORDER.index(stage) + 1
            next_stage = STAGE_ORDER[next_idx] if next_idx < len(STAGE_ORDER) else Stage.DONE
            advance_stage(self.state, stage, next_stage)
            save_state(self.state, self.run_dir)
            console.print(f"  [green]✓ {stage.value}[/] — {result.message}")

        elapsed = time.monotonic() - t0
        self.state.stage = Stage.DONE
        save_state(self.state, self.run_dir)

        reasoner_stats = self.reasoner.get_stats()
        self.trace.log(
            stage="ORCHESTRATOR",
            action="run_complete",
            duration_ms=elapsed * 1000,
            meta={"reasoner_stats": reasoner_stats},
        )
        console.print(f"\n[bold green]Run complete[/] in {elapsed:.1f}s → {self.run_dir}")

    def _ensure_dirs(self) -> None:
        for sub in ["notes", "code", "code/logs", "artifacts"]:
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)


def resume_run(run_dir: Path) -> None:
    state = load_state(run_dir)
    if state is None:
        console.print(f"[red]No state.json found in {run_dir}[/]")
        raise SystemExit(1)

    config = RunConfig.model_validate(state.config_snapshot)
    orch = Orchestrator(config, run_dir)
    orch.state = state
    console.print(f"[bold]Resuming from {state.stage.value} (step {state.step})[/]")
    orch.run()


def replay_run(
    run_dir: Path,
    from_step: int = 0,
    no_tools: bool = False,
    json_output: bool = False,
) -> None:
    trace_path = run_dir / "trace.jsonl"
    if not trace_path.exists():
        console.print(f"[red]No trace.jsonl found in {run_dir}[/]")
        raise SystemExit(1)

    logger = TraceLogger(trace_path)
    events = logger.read_all()

    if json_output:
        output = []
        for i, ev in enumerate(events):
            if i < from_step:
                continue
            entry = ev.model_dump()
            if no_tools and ev.action == "invoke":
                entry["dry_run"] = True
                entry["note"] = f"Would invoke tool={ev.tool}"
            output.append(entry)
        print(json.dumps(output, indent=2, default=str))
        return

    console.print(f"[bold]Replaying {len(events)} events from {run_dir}[/]")
    if no_tools:
        console.print("[yellow]Dry-run mode: tool invocations shown but not executed[/]")
    console.print()

    for i, ev in enumerate(events):
        if i < from_step:
            continue

        prefix = f"[{i:04d}]"
        stage_str = f"[cyan]{ev.stage}[/]" if ev.stage else ""
        agent_str = f"[magenta]{ev.agent}[/]" if ev.agent else ""
        tool_str = f"tool={ev.tool}" if ev.tool else ""
        err_str = f"[red]ERR: {ev.error}[/]" if ev.error else ""

        action_label = ev.action
        if no_tools and ev.action == "invoke":
            action_label = "[dim]WOULD_INVOKE[/]"

        parts = [p for p in [prefix, stage_str, agent_str, action_label, tool_str, err_str] if p]
        console.print(" ".join(parts))

        if ev.output_summary:
            console.print(f"       → {ev.output_summary[:120]}")
