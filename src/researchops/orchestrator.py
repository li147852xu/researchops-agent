"""Pipeline orchestrator — both legacy sequential and LangGraph-based execution."""

from __future__ import annotations

import json
import logging
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
from researchops.models import STAGE_ORDER, Stage, StateSnapshot
from researchops.reasoning import create_reasoner
from researchops.registry.builtin import register_builtin_tools
from researchops.registry.manager import ToolRegistry
from researchops.sandbox.proc import SubprocessSandbox
from researchops.supervisor import Supervisor
from researchops.trace import TraceLogger
from researchops.utils import compute_coverage, load_claim_dicts, load_plan

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    _console = Console()
    def _print(msg: str) -> None:
        _console.print(msg)
except ImportError:
    import re as _re
    def _print(msg: str) -> None:
        print(_re.sub(r"\[/?[a-z ]+\]", "", msg))


_STAGE_AGENTS: dict[Stage, type[AgentBase]] = {
    Stage.PLAN: PlannerAgent,
    Stage.COLLECT: CollectorAgent,
    Stage.READ: ReaderAgent,
    Stage.VERIFY: VerifierAgent,
    Stage.WRITE: WriterAgent,
    Stage.QA: QAAgent,
}


# ---------------------------------------------------------------------------
# LangGraph-based orchestrator (new)
# ---------------------------------------------------------------------------

class GraphOrchestrator:
    """Execute the research pipeline via LangGraph's compiled graph."""

    def __init__(self, config: RunConfig, run_dir: Path):
        self.config = config
        self.run_dir = run_dir

    def run(self) -> None:
        from researchops.graph.builder import build_research_graph
        from researchops.graph.nodes import clear_ctx_cache

        _print("[bold cyan]Running via LangGraph orchestrator[/]")
        t0 = time.monotonic()

        clear_ctx_cache()
        graph = build_research_graph()
        initial_state = {
            "run_id": self.run_dir.name,
            "run_dir": str(self.run_dir),
            "topic": self.config.topic,
            "config": self.config.model_dump(mode="json"),
            "plan": None,
            "sources": [],
            "claims": [],
            "report": "",
            "qa_result": None,
            "eval_result": None,
            "diagnostics": {},
            "stage": "PLAN",
            "rollback_target": None,
            "collect_rounds": 1,
            "max_collect_rounds": self.config.max_collect_rounds,
            "write_rounds": 0,
            "refinement_count": 0,
            "decision_history": [],
            "completed_stages": [],
            "memory": {},
            "last_error": None,
        }

        _print(f"  Topic: {self.config.topic}")
        _print(f"  Mode: {self.config.mode.value} | Retrieval: {self.config.retrieval.value}")

        try:
            final_state = graph.invoke(initial_state)
        except Exception as exc:
            _print(f"[red]Graph execution failed: {exc}[/]")
            raise

        elapsed = time.monotonic() - t0
        _print(f"\n[bold green]Run complete[/] in {elapsed:.1f}s → {self.run_dir}")

        eval_result = final_state.get("eval_result", {})
        if eval_result:
            _print(f"  Citation coverage: {eval_result.get('citation_coverage', 0):.1%}")
            _print(f"  Bucket coverage: {eval_result.get('bucket_coverage_rate', 0):.1%}")
            _print(f"  Relevance avg: {eval_result.get('relevance_avg', 0):.2f}")


# ---------------------------------------------------------------------------
# Legacy sequential orchestrator (kept for backward compatibility)
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(self, config: RunConfig, run_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.trace = TraceLogger(run_dir / "trace.jsonl")
        self.registry = self._build_registry()
        self.sandbox = self._build_sandbox()
        self.reasoner = create_reasoner(config)
        self.supervisor = Supervisor(run_dir, self.reasoner)
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
            stage="ORCHESTRATOR", action="registry_init",
            meta={"granted_permissions": list(perms)},
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
            config_snapshot=self.config.safe_dump(),
        )
        save_state(state, self.run_dir)
        return state

    def _make_context(self) -> RunContext:
        return RunContext(
            run_dir=self.run_dir, config=self.config, state=self.state,
            registry=self.registry, trace=self.trace, sandbox=self.sandbox,
            reasoner=self.reasoner,
        )

    def run(self) -> None:
        self._ensure_dirs()
        t0 = time.monotonic()
        self.trace.log(stage="ORCHESTRATOR", action="run_start", input_summary=self.config.topic)
        self.trace.log(
            stage="ORCHESTRATOR", action="llm_status",
            meta={
                "llm": self.config.llm, "is_llm": self.reasoner.is_llm,
                "provider_label": getattr(self.reasoner, "provider_label", "none"),
                "model": getattr(self.reasoner, "model", "none"),
                "sources_strategy": self.config.sources.value,
                "retrieval_mode": self.config.retrieval.value,
            },
        )

        pipeline = [s for s in STAGE_ORDER if s != Stage.DONE]
        # Use a while-loop instead of recursion for rollbacks
        stage_idx = 0
        while stage_idx < len(pipeline):
            stage = pipeline[stage_idx]
            if should_skip(self.state, stage):
                _print(f"  [dim]Skipping {stage.value} (already completed)[/]")
                stage_idx += 1
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
                self.trace.log(stage=stage.value, agent=agent.name, action="error", error=str(exc))
                _print(f"  [red]Error in {stage.value}: {exc}[/]")
                save_state(self.state, self.run_dir)
                raise

            if stage == Stage.WRITE:
                self.state.write_rounds += 1

            if result.rollback_to is not None:
                rollback_target = result.rollback_to
                diagnostics = self._build_diagnostics()
                self.supervisor.decide(self.state, diagnostics, self.config, self.trace)

                if rollback_target == Stage.COLLECT:
                    if self.state.collect_rounds >= self.config.max_collect_rounds:
                        _print("  [yellow]Max collect rounds reached — degrade-completing[/]")
                        self._degrade_complete(result.message)
                        stage_idx += 1
                        continue
                    self.state.collect_rounds += 1

                _print(f"  [yellow]Rollback → {rollback_target.value}: {result.message}[/]")
                self.trace.log(
                    stage=stage.value, agent=agent.name, action="rollback",
                    output_summary=f"Rolling back to {rollback_target.value}",
                    meta={"collect_rounds": self.state.collect_rounds},
                )
                self.state.stage = rollback_target
                if rollback_target in self.state.completed_stages:
                    self.state.completed_stages.remove(rollback_target)
                for s in STAGE_ORDER:
                    if STAGE_ORDER.index(s) > STAGE_ORDER.index(rollback_target) and s in self.state.completed_stages:
                        self.state.completed_stages.remove(s)
                save_state(self.state, self.run_dir)
                stage_idx = pipeline.index(rollback_target)
                continue

            if stage == Stage.READ:
                self._build_retrieval_index(ctx)

            next_idx = STAGE_ORDER.index(stage) + 1
            next_stage = STAGE_ORDER[next_idx] if next_idx < len(STAGE_ORDER) else Stage.DONE
            advance_stage(self.state, stage, next_stage)
            save_state(self.state, self.run_dir)
            _print(f"  [green]✓ {stage.value}[/] — {result.message}")
            stage_idx += 1

        elapsed = time.monotonic() - t0
        self.state.stage = Stage.DONE
        save_state(self.state, self.run_dir)
        reasoner_stats = self.reasoner.get_stats()
        self.trace.log(
            stage="ORCHESTRATOR", action="run_complete",
            duration_ms=elapsed * 1000,
            meta={"reasoner_stats": reasoner_stats},
        )
        _print(f"\n[bold green]Run complete[/] in {elapsed:.1f}s → {self.run_dir}")

    def _build_retrieval_index(self, ctx: RunContext) -> None:
        from researchops.retrieval import create_retriever

        retriever = create_retriever(
            self.config.retrieval.value, self.run_dir,
            embedder_model=self.config.embedder_model,
        )
        all_claims = load_claim_dicts(self.run_dir)
        retriever.index(all_claims)
        ctx.shared["retriever"] = retriever
        self.trace.log(
            stage="ORCHESTRATOR", action="retrieval.index_built",
            output_summary=f"Indexed {len(all_claims)} claims with {self.config.retrieval.value}",
        )

    def _build_diagnostics(self) -> dict:
        diagnostics: dict = {"coverage_vector": self.state.coverage_vector}
        qa_report_path = self.run_dir / "qa_report.json"
        if qa_report_path.exists():
            try:
                qa_data = json.loads(qa_report_path.read_text(encoding="utf-8"))
                diagnostics["bucket_coverage_rate"] = qa_data.get("bucket_coverage_rate", 1.0)
                diagnostics["relevance_avg"] = qa_data.get("relevance_avg", 1.0)
            except Exception:
                pass
        return diagnostics

    def _degrade_complete(self, reason: str) -> None:
        plan = load_plan(self.run_dir)
        incomplete: list[str] = []
        if plan:
            rq_counts, _ = compute_coverage(self.run_dir)
            for rq in plan.research_questions:
                if rq_counts.get(rq.rq_id, 0) < self.config.min_claims_per_rq:
                    incomplete.append(rq.rq_id)
        self.state.incomplete_sections = incomplete
        self.trace.log(
            stage="ORCHESTRATOR", action="degrade_complete",
            output_summary=f"Degrading: {len(incomplete)} sections incomplete",
        )
        save_state(self.state, self.run_dir)

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
    run_dir: Path, from_step: int = 0, no_tools: bool = False, json_output: bool = False,
) -> None:
    trace_path = run_dir / "trace.jsonl"
    if not trace_path.exists():
        _print(f"[red]No trace.jsonl found in {run_dir}[/]")
        raise SystemExit(1)
    tl = TraceLogger(trace_path)
    events = tl.read_all()

    if json_output:
        output = []
        for i, ev in enumerate(events):
            if i < from_step:
                continue
            entry: dict = {
                "step": i, "stage": ev.stage, "agent": ev.agent,
                "action": ev.action, "tool": ev.tool,
                "outcome": "error" if ev.error else "ok",
                "latency_ms": ev.duration_ms,
                "cache_hit": ev.action == "cache_hit",
            }
            if no_tools and ev.tool:
                entry["dry_run"] = True
            output.append(entry)
        print(json.dumps(output, indent=2, default=str))
        return

    _print(f"[bold]Replaying {len(events)} events from {run_dir}[/]")
    for i, ev in enumerate(events):
        if i < from_step:
            continue
        prefix = f"[{i:04d}]"
        parts = [prefix]
        if ev.stage:
            parts.append(f"[cyan]{ev.stage}[/]")
        if ev.agent:
            parts.append(f"[magenta]{ev.agent}[/]")
        parts.append(ev.action)
        if ev.tool:
            parts.append(f"tool={ev.tool}")
        if ev.error:
            parts.append(f"[red]ERR: {ev.error}[/]")
        _print(" ".join(parts))
        if ev.output_summary:
            _print(f"       → {ev.output_summary[:120]}")
