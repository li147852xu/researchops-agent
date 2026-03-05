from __future__ import annotations

import json
import time
from pathlib import Path

from researchops.agents.base import AgentBase, RunContext
from researchops.agents.collector import CollectorAgent
from researchops.agents.planner import PlannerAgent
from researchops.agents.qa import QAAgent
from researchops.agents.reader import ReaderAgent
from researchops.agents.verifier import VerifierAgent
from researchops.agents.writer import WriterAgent
from researchops.checkpoint import advance_stage, load_state, save_state, should_skip
from researchops.config import RunConfig, SandboxBackend
from researchops.models import STAGE_ORDER, PlanOutput, SourceNotes, Stage, StateSnapshot
from researchops.reasoning import create_reasoner
from researchops.registry.builtin import register_builtin_tools
from researchops.registry.manager import ToolRegistry
from researchops.sandbox.proc import SubprocessSandbox
from researchops.trace import TraceLogger

try:
    from rich.console import Console
    _console = Console()
    def _print(msg: str) -> None:
        _console.print(msg)
except ImportError:
    def _print(msg: str) -> None:
        import re
        print(re.sub(r"\[/?[a-z ]+\]", "", msg))


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
            _print("[yellow]Docker sandbox not implemented, falling back to subprocess[/]")
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
        self.trace.log(
            stage="ORCHESTRATOR",
            action="llm_status",
            meta={
                "llm": self.config.llm,
                "is_llm": self.reasoner.is_llm,
                "provider_label": getattr(self.reasoner, "provider_label", "none"),
                "model": getattr(self.reasoner, "model", "none"),
            },
        )

        pipeline = [s for s in STAGE_ORDER if s != Stage.DONE]

        for stage in pipeline:
            if should_skip(self.state, stage):
                _print(f"  [dim]Skipping {stage.value} (already completed)[/]")
                continue

            if self.state.step >= self.config.max_steps:
                _print("[red]Max steps reached, stopping[/]")
                break

            self.state.stage = stage
            save_state(self.state, self.run_dir)

            agent_cls = _STAGE_AGENTS[stage]
            agent = agent_cls()
            ctx = self._make_context()

            _print(f"  [bold cyan]▶ {stage.value}[/] ({agent.name})")

            try:
                result = agent.execute(ctx)
            except Exception as exc:
                self.trace.log(
                    stage=stage.value,
                    agent=agent.name,
                    action="error",
                    error=str(exc),
                )
                _print(f"  [red]Error in {stage.value}: {exc}[/]")
                save_state(self.state, self.run_dir)
                raise

            if result.rollback_to is not None:
                rollback_target = result.rollback_to
                _print(f"  [yellow]Rollback → {rollback_target.value}: {result.message}[/]")
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

            if stage == Stage.READ:
                refinement_result = self._check_plan_coverage(ctx)
                if refinement_result is not None:
                    return self.run()

            next_idx = STAGE_ORDER.index(stage) + 1
            next_stage = STAGE_ORDER[next_idx] if next_idx < len(STAGE_ORDER) else Stage.DONE
            advance_stage(self.state, stage, next_stage)
            save_state(self.state, self.run_dir)
            _print(f"  [green]✓ {stage.value}[/] — {result.message}")

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
        _print(f"\n[bold green]Run complete[/] in {elapsed:.1f}s → {self.run_dir}")

    def _check_plan_coverage(self, ctx: RunContext) -> str | None:
        plan_path = ctx.run_dir / "plan.json"
        if not plan_path.exists():
            return None
        plan = PlanOutput.model_validate(json.loads(plan_path.read_text(encoding="utf-8")))
        notes_dir = ctx.run_dir / "notes"

        rq_claim_counts: dict[str, int] = {rq.rq_id: 0 for rq in plan.research_questions}
        if notes_dir.exists():
            for f in notes_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    notes = SourceNotes.model_validate(data)
                    for claim in notes.claims:
                        for rq_id in claim.supports_rq:
                            if rq_id in rq_claim_counts:
                                rq_claim_counts[rq_id] += 1
                except Exception:
                    continue

        total_rqs = len(plan.research_questions)
        covered = sum(1 for c in rq_claim_counts.values() if c > 0)
        coverage = covered / total_rqs if total_rqs > 0 else 1.0

        if coverage >= plan.acceptance_threshold:
            return None

        if not self.config.allow_net:
            ctx.shared["evidence_limited"] = True
            self.trace.log(
                stage="ORCHESTRATOR",
                action="plan.refine",
                output_summary=f"Coverage {coverage:.0%} below threshold, but allow_net=false; marking evidence_limited",
                meta={"coverage": coverage, "threshold": plan.acceptance_threshold},
            )
            return None

        max_ref = self.config.max_refinements
        if self.state.refinement_count >= max_ref:
            self.trace.log(
                stage="ORCHESTRATOR",
                action="plan.refine",
                output_summary=f"Coverage {coverage:.0%} still low after {self.state.refinement_count} refinements; proceeding",
                meta={"coverage": coverage, "refinement_count": self.state.refinement_count},
            )
            return None

        self.state.refinement_count += 1
        self.state.collect_rounds += 1
        self.trace.log(
            stage="ORCHESTRATOR",
            action="plan.refine",
            output_summary=f"Coverage {coverage:.0%} < {plan.acceptance_threshold:.0%}; rolling back to COLLECT (round {self.state.collect_rounds})",
            meta={
                "coverage": coverage,
                "threshold": plan.acceptance_threshold,
                "refinement_count": self.state.refinement_count,
                "collect_rounds": self.state.collect_rounds,
            },
        )
        _print(f"  [yellow]Plan refine: coverage {coverage:.0%}, rolling back to COLLECT[/]")

        self.state.stage = Stage.COLLECT
        if Stage.COLLECT in self.state.completed_stages:
            self.state.completed_stages.remove(Stage.COLLECT)
        if Stage.READ in self.state.completed_stages:
            self.state.completed_stages.remove(Stage.READ)
        save_state(self.state, self.run_dir)
        return "refinement"

    def _ensure_dirs(self) -> None:
        for sub in ["notes", "code", "code/logs", "artifacts", "downloads"]:
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)


def resume_run(run_dir: Path) -> None:
    state = load_state(run_dir)
    if state is None:
        _print(f"[red]No state.json found in {run_dir}[/]")
        raise SystemExit(1)

    config = RunConfig.model_validate(state.config_snapshot)
    orch = Orchestrator(config, run_dir)
    orch.state = state
    _print(f"[bold]Resuming from {state.stage.value} (step {state.step})[/]")
    orch.run()


def replay_run(
    run_dir: Path,
    from_step: int = 0,
    no_tools: bool = False,
    json_output: bool = False,
) -> None:
    trace_path = run_dir / "trace.jsonl"
    if not trace_path.exists():
        _print(f"[red]No trace.jsonl found in {run_dir}[/]")
        raise SystemExit(1)

    logger = TraceLogger(trace_path)
    events = logger.read_all()

    if json_output:
        output = []
        for i, ev in enumerate(events):
            if i < from_step:
                continue
            entry = {
                "step": i,
                "stage": ev.stage,
                "agent": ev.agent,
                "action": ev.action,
                "tool": ev.tool,
                "outcome": "error" if ev.error else "ok",
                "latency_ms": ev.duration_ms,
                "cache_hit": ev.action == "cache_hit",
                "error": ev.error,
            }
            if no_tools and ev.action == "invoke":
                entry["dry_run"] = True
                entry["note"] = f"Would invoke tool={ev.tool} with {ev.input_summary[:100]}"
            output.append(entry)
        print(json.dumps(output, indent=2, default=str))
        return

    _print(f"[bold]Replaying {len(events)} events from {run_dir}[/]")
    if no_tools:
        _print("[yellow]Dry-run mode: tool invocations shown but not executed[/]")

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
        _print(" ".join(parts))

        if ev.output_summary:
            _print(f"       → {ev.output_summary[:120]}")
