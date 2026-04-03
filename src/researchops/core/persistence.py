"""Persistence layer — SQLite-backed run metadata index via SQLAlchemy.

Provides a searchable index of completed runs alongside the existing
file-based artifact store.  The database lives at ``runs/researchops.db``
by default and is created lazily on first access.
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DB_FILENAME = "researchops.db"

try:
    from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
    from sqlalchemy.orm import declarative_base, sessionmaker

    _HAS_SQLALCHEMY = True
except ImportError:
    _HAS_SQLALCHEMY = False

if _HAS_SQLALCHEMY:
    Base = declarative_base()

    class RunRecord(Base):  # type: ignore[valid-type]
        __tablename__ = "runs"

        id = Column(Integer, primary_key=True, autoincrement=True)
        run_id = Column(String(64), unique=True, nullable=False, index=True)
        app_type = Column(String(32), nullable=False, default="research")
        topic = Column(Text, default="")
        status = Column(String(32), default="pending")
        stage = Column(String(32), default="")
        run_dir = Column(Text, default="")
        created_at = Column(DateTime, default=lambda: datetime.now(UTC))
        completed_at = Column(DateTime, nullable=True)
        citation_coverage = Column(Float, default=0.0)
        evidence_density = Column(Float, default=0.0)
        source_count = Column(Integer, default=0)
        claim_count = Column(Integer, default=0)
        latency_sec = Column(Float, default=0.0)
        llm_provider = Column(String(64), default="")
        extra_json = Column(Text, default="{}")

    class ArtifactRecord(Base):  # type: ignore[valid-type]
        __tablename__ = "artifacts"

        id = Column(Integer, primary_key=True, autoincrement=True)
        run_id = Column(String(64), nullable=False, index=True)
        artifact_type = Column(String(64), nullable=False)
        file_path = Column(Text, default="")
        size_bytes = Column(Integer, default=0)
        created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class RunIndex:
    """SQLite-backed index for run metadata.

    Falls back gracefully to no-op when SQLAlchemy is not installed.
    """

    def __init__(self, db_path: Path | str | None = None):
        self._enabled = _HAS_SQLALCHEMY
        if not self._enabled:
            logger.info("SQLAlchemy not installed — persistence layer disabled")
            return

        if db_path is None:
            db_path = Path("runs") / _DB_FILENAME
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

    def record_run(
        self,
        run_id: str,
        app_type: str,
        topic: str,
        run_dir: str,
        status: str = "completed",
        eval_data: dict[str, Any] | None = None,
        llm_provider: str = "",
    ) -> None:
        """Insert or update a run record in the index."""
        if not self._enabled:
            return
        eval_data = eval_data or {}
        with self._Session() as session:
            existing = session.query(RunRecord).filter_by(run_id=run_id).first()
            if existing:
                existing.status = status
                existing.completed_at = datetime.now(UTC)
                existing.citation_coverage = eval_data.get("citation_coverage", 0.0)
                existing.evidence_density = eval_data.get("evidence_density", 0.0)
                existing.latency_sec = eval_data.get("latency_sec", 0.0)
                existing.llm_provider = llm_provider
            else:
                record = RunRecord(
                    run_id=run_id,
                    app_type=app_type,
                    topic=topic,
                    run_dir=run_dir,
                    status=status,
                    completed_at=datetime.now(UTC) if status == "completed" else None,
                    citation_coverage=eval_data.get("citation_coverage", 0.0),
                    evidence_density=eval_data.get("evidence_density", 0.0),
                    source_count=eval_data.get("source_count", 0),
                    claim_count=eval_data.get("claim_count", 0),
                    latency_sec=eval_data.get("latency_sec", 0.0),
                    llm_provider=llm_provider,
                    extra_json=json.dumps(eval_data),
                )
                session.add(record)
            session.commit()

    def record_artifact(
        self, run_id: str, artifact_type: str, file_path: str, size_bytes: int = 0,
    ) -> None:
        """Record an artifact file associated with a run."""
        if not self._enabled:
            return
        with self._Session() as session:
            session.add(ArtifactRecord(
                run_id=run_id, artifact_type=artifact_type,
                file_path=file_path, size_bytes=size_bytes,
            ))
            session.commit()

    def list_runs(
        self, app_type: str | None = None, limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List recent runs, optionally filtered by app type."""
        if not self._enabled:
            return []
        with self._Session() as session:
            query = session.query(RunRecord).order_by(RunRecord.created_at.desc())
            if app_type:
                query = query.filter_by(app_type=app_type)
            results = query.limit(limit).all()
            return [
                {
                    "run_id": r.run_id,
                    "app_type": r.app_type,
                    "topic": r.topic,
                    "status": r.status,
                    "created_at": r.created_at.isoformat() if r.created_at else "",
                    "citation_coverage": r.citation_coverage,
                    "evidence_density": r.evidence_density,
                    "latency_sec": r.latency_sec,
                    "llm_provider": r.llm_provider,
                }
                for r in results
            ]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Retrieve a single run record."""
        if not self._enabled:
            return None
        with self._Session() as session:
            r = session.query(RunRecord).filter_by(run_id=run_id).first()
            if not r:
                return None
            return {
                "run_id": r.run_id,
                "app_type": r.app_type,
                "topic": r.topic,
                "status": r.status,
                "run_dir": r.run_dir,
                "created_at": r.created_at.isoformat() if r.created_at else "",
                "completed_at": r.completed_at.isoformat() if r.completed_at else "",
                "citation_coverage": r.citation_coverage,
                "evidence_density": r.evidence_density,
                "source_count": r.source_count,
                "claim_count": r.claim_count,
                "latency_sec": r.latency_sec,
                "llm_provider": r.llm_provider,
                "extra": json.loads(r.extra_json) if r.extra_json else {},
            }

    def index_completed_run(self, run_dir: Path) -> None:
        """Scan a completed run directory and index its metadata."""
        if not self._enabled:
            return
        run_id = run_dir.name
        app_type = "market" if "quant" in run_id else "research"

        topic = ""
        plan_path = run_dir / "plan.json"
        if plan_path.exists():
            try:
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                topic = plan.get("topic", plan.get("query", ""))
            except Exception:
                pass

        eval_data: dict[str, Any] = {}
        eval_path = run_dir / "eval.json"
        if eval_path.exists():
            with contextlib.suppress(Exception):
                eval_data = json.loads(eval_path.read_text(encoding="utf-8"))

        sources_path = run_dir / "sources.jsonl"
        if sources_path.exists():
            try:
                lines = sources_path.read_text(encoding="utf-8").strip().splitlines()
                eval_data["source_count"] = len([line for line in lines if line.strip()])
            except Exception:
                pass

        self.record_run(
            run_id=run_id, app_type=app_type, topic=topic,
            run_dir=str(run_dir), status="completed", eval_data=eval_data,
        )

        for f in run_dir.iterdir():
            if f.is_file() and f.suffix in (".json", ".jsonl", ".md"):
                self.record_artifact(
                    run_id=run_id, artifact_type=f.stem,
                    file_path=str(f), size_bytes=f.stat().st_size,
                )
