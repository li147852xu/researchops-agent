"""Tests for the LangGraph pipeline components — state, edges, builder."""

from __future__ import annotations

import pytest

from researchops.core.pipeline import (
    PipelineState,
    WorkingMemory,
    _after_qa,
    _after_read,
    _after_supervisor,
    _after_write,
)


class TestPipelineState:
    def test_state_dict_creation(self):
        state: PipelineState = {
            "run_id": "test_run",
            "run_dir": "/tmp/test",
            "topic": "deep learning",
            "config": {},
            "stage": "PLAN",
        }
        assert state["topic"] == "deep learning"
        assert state["stage"] == "PLAN"

    def test_working_memory_defaults(self):
        mem: WorkingMemory = {}
        assert mem.get("evidence_limited") is None

    def test_state_with_full_fields(self):
        state: PipelineState = {
            "run_id": "r1",
            "run_dir": "/tmp/r1",
            "topic": "topic",
            "config": {},
            "plan": None,
            "sources": [],
            "claims": [],
            "report": "",
            "qa_result": None,
            "diagnostics": {},
            "stage": "PLAN",
            "rollback_target": None,
            "collect_rounds": 1,
            "max_collect_rounds": 6,
            "write_rounds": 0,
            "refinement_count": 0,
            "decision_history": [],
            "completed_stages": [],
            "memory": {},
            "last_error": None,
        }
        assert state["max_collect_rounds"] == 6


class TestConditionalEdges:
    def test_after_read_high_coverage(self):
        state: PipelineState = {
            "diagnostics": {"coverage": 0.9},
            "config": {"acceptance_threshold": 0.7},
            "collect_rounds": 1,
            "max_collect_rounds": 6,
        }
        assert _after_read(state) == "verify"

    def test_after_read_low_coverage(self):
        state: PipelineState = {
            "diagnostics": {"coverage": 0.3},
            "config": {"acceptance_threshold": 0.7},
            "collect_rounds": 1,
            "max_collect_rounds": 6,
        }
        assert _after_read(state) == "supervisor"

    def test_after_read_low_coverage_max_rounds(self):
        state: PipelineState = {
            "diagnostics": {"coverage": 0.3},
            "config": {"acceptance_threshold": 0.7},
            "collect_rounds": 6,
            "max_collect_rounds": 6,
        }
        assert _after_read(state) == "verify"

    def test_after_write_no_rollback(self):
        state: PipelineState = {"rollback_target": None}
        assert _after_write(state) == "qa"

    def test_after_write_with_rollback(self):
        state: PipelineState = {"rollback_target": "COLLECT"}
        assert _after_write(state) == "supervisor"

    def test_after_qa_passed(self):
        state: PipelineState = {"qa_result": {"passed": True}}
        assert _after_qa(state) == "evaluate"

    def test_after_qa_failed_write(self):
        state: PipelineState = {
            "qa_result": {"passed": False},
            "rollback_target": "WRITE",
        }
        assert _after_qa(state) == "write"

    def test_after_qa_failed_collect(self):
        state: PipelineState = {
            "qa_result": {"passed": False},
            "rollback_target": "COLLECT",
        }
        assert _after_qa(state) == "supervisor"

    def test_after_supervisor_collect(self):
        state: PipelineState = {"rollback_target": "collect"}
        assert _after_supervisor(state) == "collect"

    def test_after_supervisor_read(self):
        state: PipelineState = {"rollback_target": "READ"}
        assert _after_supervisor(state) == "read"

    def test_after_supervisor_invalid(self):
        state: PipelineState = {"rollback_target": "INVALID"}
        assert _after_supervisor(state) == "collect"


class TestGraphBuilder:
    def test_graph_compiles(self):
        """Verify the generic pipeline can be built for any app."""
        try:
            from researchops.apps.registry import get_app
            from researchops.core.pipeline import build_pipeline_graph

            spec = get_app("research")
            graph = build_pipeline_graph(spec)
            assert graph is not None
        except ImportError:
            pytest.skip("langgraph not installed")
