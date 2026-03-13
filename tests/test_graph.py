"""Tests for the LangGraph components — state, edges, builder."""

from __future__ import annotations

import pytest

from researchops.graph.edges import after_qa, after_read, after_supervisor, after_write
from researchops.graph.state import ResearchState, WorkingMemory


class TestResearchState:
    def test_state_dict_creation(self):
        state: ResearchState = {
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
        state: ResearchState = {
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
        state: ResearchState = {
            "diagnostics": {"coverage": 0.9},
            "config": {"acceptance_threshold": 0.7},
            "collect_rounds": 1,
            "max_collect_rounds": 6,
        }
        assert after_read(state) == "verify"

    def test_after_read_low_coverage(self):
        state: ResearchState = {
            "diagnostics": {"coverage": 0.3},
            "config": {"acceptance_threshold": 0.7},
            "collect_rounds": 1,
            "max_collect_rounds": 6,
        }
        assert after_read(state) == "supervisor"

    def test_after_read_low_coverage_max_rounds(self):
        state: ResearchState = {
            "diagnostics": {"coverage": 0.3},
            "config": {"acceptance_threshold": 0.7},
            "collect_rounds": 6,
            "max_collect_rounds": 6,
        }
        assert after_read(state) == "verify"

    def test_after_write_no_rollback(self):
        state: ResearchState = {"rollback_target": None}
        assert after_write(state) == "qa"

    def test_after_write_with_rollback(self):
        state: ResearchState = {"rollback_target": "COLLECT"}
        assert after_write(state) == "supervisor"

    def test_after_qa_passed(self):
        state: ResearchState = {"qa_result": {"passed": True}}
        assert after_qa(state) == "evaluate"

    def test_after_qa_failed_write(self):
        state: ResearchState = {
            "qa_result": {"passed": False},
            "rollback_target": "WRITE",
        }
        assert after_qa(state) == "write"

    def test_after_qa_failed_collect(self):
        state: ResearchState = {
            "qa_result": {"passed": False},
            "rollback_target": "COLLECT",
        }
        assert after_qa(state) == "supervisor"

    def test_after_supervisor_collect(self):
        state: ResearchState = {"rollback_target": "collect"}
        assert after_supervisor(state) == "collect"

    def test_after_supervisor_read(self):
        state: ResearchState = {"rollback_target": "READ"}
        assert after_supervisor(state) == "read"

    def test_after_supervisor_invalid(self):
        state: ResearchState = {"rollback_target": "INVALID"}
        assert after_supervisor(state) == "collect"


class TestGraphBuilder:
    def test_graph_compiles(self):
        """Verify the graph can be built and compiled without error."""
        try:
            from researchops.graph.builder import build_research_graph
            graph = build_research_graph()
            assert graph is not None
        except ImportError:
            pytest.skip("langgraph not installed")
