"""FastAPI application — unified REST endpoints for the multi-agent platform."""

from __future__ import annotations

import contextlib
import json
import uuid
from pathlib import Path
from typing import Any

from researchops.api.schemas import (
    AppInfo,
    ArtifactResponse,
    HealthResponse,
    RunRequest,
    RunStatus,
    RunSummary,
)

_RUNS_DIR = Path("runs")


def create_api():
    """Create and return the FastAPI application."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="ResearchOps API",
        description="Unified REST API for the ResearchOps multi-agent workflow platform",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health():
        from researchops import __version__
        return HealthResponse(version=__version__)

    @app.get("/api/v1/apps", response_model=list[AppInfo])
    def list_apps():
        from researchops.apps.registry import list_apps as _list_apps
        return [
            AppInfo(name=s.name, display_name=s.display_name, description=s.description)
            for s in _list_apps()
        ]

    @app.post("/api/v1/apps/{app_name}/run", response_model=RunStatus)
    def run_app(app_name: str, req: RunRequest):
        from researchops.apps.registry import get_app
        from researchops.core.pipeline import build_initial_state, build_pipeline_graph, clear_ctx_cache

        try:
            spec = get_app(app_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        run_id = f"{app_name}_{uuid.uuid4().hex[:8]}"
        run_dir = _RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        config_data: dict[str, Any] = {
            "topic": req.topic,
            "mode": req.mode,
            "allow_net": True,
            "run_dir": run_dir,
            "sources": req.sources,
            "llm": req.llm.provider if req.llm else "openai_compat",
            "llm_model": req.llm.model if req.llm else "",
            "llm_base_url": req.llm.base_url if req.llm else "",
            "llm_api_key": req.llm.api_key if req.llm else "",
        }
        if req.extra:
            config_data.update(req.extra)

        config = spec.config_class.model_validate(config_data)

        clear_ctx_cache()
        graph = build_pipeline_graph(spec)
        initial_state = build_initial_state(run_id, run_dir, req.topic, config)

        try:
            graph.invoke(initial_state, {"recursion_limit": 100})
        except Exception:
            return RunStatus(
                run_id=run_id, app_name=app_name,
                status="failed", stage="ERROR", run_dir=str(run_dir),
            )

        if spec.compute_eval:
            try:
                spec.compute_eval(run_dir, config=config)
            except Exception:
                pass

        return RunStatus(
            run_id=run_id, app_name=app_name,
            status="completed", stage="DONE", run_dir=str(run_dir),
        )

    @app.get("/api/v1/runs", response_model=list[RunSummary])
    def list_runs():
        if not _RUNS_DIR.exists():
            return []
        summaries: list[RunSummary] = []
        for d in sorted(_RUNS_DIR.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            parts = d.name.split("_", 1)
            app_name = parts[0] if len(parts) > 1 else "research"
            if app_name in ("run",):
                app_name = "research"
            topic = ""
            plan_path = d / "plan.json"
            if plan_path.exists():
                try:
                    plan = json.loads(plan_path.read_text(encoding="utf-8"))
                    topic = plan.get("topic", plan.get("query", ""))
                except Exception:
                    pass
            status = "completed" if (d / "report.md").exists() else "partial"
            eval_summary = {}
            eval_path = d / "eval.json"
            if eval_path.exists():
                with contextlib.suppress(Exception):
                    eval_summary = json.loads(eval_path.read_text(encoding="utf-8"))
            summaries.append(RunSummary(
                run_id=d.name, app_name=app_name, status=status,
                topic=topic, eval_summary=eval_summary,
            ))
        return summaries[:50]

    @app.get("/api/v1/runs/{run_id}/artifacts/{artifact_type}", response_model=ArtifactResponse)
    def get_artifact(run_id: str, artifact_type: str):
        run_dir = _RUNS_DIR / run_id
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        file_map = {
            "plan": "plan.json",
            "report": "report.md",
            "eval": "eval.json",
            "sources": "sources.jsonl",
            "trace": "trace.jsonl",
            "qa": "qa_report.json",
            "state": "state.json",
        }

        filename = file_map.get(artifact_type)
        if not filename:
            raise HTTPException(status_code=400, detail=f"Unknown artifact type: {artifact_type}")

        path = run_dir / filename
        if not path.exists():
            return ArtifactResponse(
                run_id=run_id, artifact_type=artifact_type,
                error=f"{filename} not found",
            )

        text = path.read_text(encoding="utf-8")
        if filename.endswith(".json"):
            content = json.loads(text)
        elif filename.endswith(".jsonl"):
            content = [json.loads(line) for line in text.strip().splitlines() if line.strip()]
        else:
            content = text

        return ArtifactResponse(
            run_id=run_id, artifact_type=artifact_type, content=content,
        )

    return app
