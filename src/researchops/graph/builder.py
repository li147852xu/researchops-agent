"""Build and compile the LangGraph research pipeline."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from researchops.graph.edges import after_qa, after_read, after_supervisor, after_write
from researchops.graph.nodes import (
    collect_node,
    eval_node,
    plan_node,
    qa_node,
    read_node,
    supervisor_node,
    verify_node,
    write_node,
)
from researchops.graph.state import ResearchState

logger = logging.getLogger(__name__)


def build_research_graph() -> Any:
    """Construct and compile the research agent graph.

    Graph topology::

        plan -> collect -> read --(coverage ok)--> verify -> write --(ok)--> qa --(pass)--> evaluate -> END
                  ^                |                                   |          |
                  |                +-(coverage low)-> supervisor <-----+          |
                  |                                      |                       |
                  +--------------------------------------+                       |
                                                         +-- (rewrite) -> write -+

    Returns a compiled LangGraph ``CompiledGraph``.
    """
    g: StateGraph = StateGraph(ResearchState)

    # --- Nodes ---
    g.add_node("plan", plan_node)
    g.add_node("collect", collect_node)
    g.add_node("read", read_node)
    g.add_node("verify", verify_node)
    g.add_node("write", write_node)
    g.add_node("qa", qa_node)
    g.add_node("evaluate", eval_node)
    g.add_node("supervisor", supervisor_node)

    # --- Edges ---
    g.add_edge("plan", "collect")
    g.add_edge("collect", "read")
    g.add_conditional_edges("read", after_read, {
        "verify": "verify",
        "supervisor": "supervisor",
    })
    g.add_edge("verify", "write")
    g.add_conditional_edges("write", after_write, {
        "qa": "qa",
        "supervisor": "supervisor",
    })
    g.add_conditional_edges("qa", after_qa, {
        "evaluate": "evaluate",
        "supervisor": "supervisor",
        "write": "write",
    })
    g.add_conditional_edges("supervisor", after_supervisor, {
        "collect": "collect",
        "read": "read",
        "write": "write",
        "verify": "verify",
    })
    g.add_edge("evaluate", END)

    g.set_entry_point("plan")

    return g.compile()
