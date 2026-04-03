"""Market Intelligence source adapters — register finance-focused tools into the core ToolRegistry.

Reuses the same web_search, fetch, parse, sandbox, and citation tools from the research
adapters, demonstrating how different apps share the same core tool infrastructure.
"""

from __future__ import annotations

from researchops.apps.research.adapters import (
    arxiv_download_pdf,
    arxiv_search,
    fetch_page,
    parse_doc,
    semantic_scholar_search,
    web_search,
    wikipedia_search,
)
from researchops.core.tools.builtins import cite, sandbox_exec
from researchops.core.tools.registry import ToolRegistry
from researchops.core.tools.schema import ToolDefinition


def register_market_tools(registry: ToolRegistry) -> None:
    """Register market-intelligence tools into the shared ToolRegistry."""
    registry.register(
        ToolDefinition(
            name="web_search", version="1.0.0",
            description="Search the web for financial news, market data, and analyst reports",
            input_schema={"query": "str", "max_results": "int"},
            output_schema={"results": "list[dict]"},
            risk_level="medium", permissions=["net"],
            timeout_default=15, cache_policy="session",
        ),
        web_search,
    )
    registry.register(
        ToolDefinition(
            name="fetch", version="1.0.0",
            description="Download a URL to local storage",
            input_schema={"url": "str", "dest_dir": "str"},
            output_schema={"status": "str", "local_path": "str", "detected_type": "str"},
            risk_level="medium", permissions=["net"],
            timeout_default=30, cache_policy="session",
        ),
        fetch_page,
    )
    registry.register(
        ToolDefinition(
            name="parse", version="1.0.0",
            description="Parse HTML or PDF to extract text",
            input_schema={"file_path": "str", "format": "str"},
            output_schema={"text": "str", "title": "str", "quality_score": "float"},
            risk_level="low", permissions=[],
            timeout_default=15, cache_policy="session",
        ),
        parse_doc,
    )
    registry.register(
        ToolDefinition(
            name="sandbox_exec", version="1.0.0",
            description="Execute Python in sandbox",
            input_schema={"script_path": "str", "timeout": "int", "allow_net": "bool"},
            output_schema={"exit_code": "int", "stdout": "str", "stderr": "str"},
            risk_level="high", permissions=["sandbox"],
            timeout_default=60, cache_policy="none",
        ),
        sandbox_exec,
    )
    registry.register(
        ToolDefinition(
            name="cite", version="1.0.0",
            description="Map source/claim IDs to citation markers",
            input_schema={"source_id": "str", "claim_id": "str"},
            output_schema={"marker": "str"},
            risk_level="low", permissions=[],
            timeout_default=5, cache_policy="persistent",
        ),
        cite,
    )
    registry.register(
        ToolDefinition(
            name="arxiv_search", version="1.0.0",
            description="Search arXiv via Atom API for academic papers",
            input_schema={"query": "str", "max_results": "int"},
            output_schema={"results": "list[dict]"},
            risk_level="medium", permissions=["net"],
            timeout_default=25, cache_policy="session",
        ),
        arxiv_search,
    )
    registry.register(
        ToolDefinition(
            name="arxiv_download_pdf", version="1.0.0",
            description="Download PDF from arXiv",
            input_schema={"pdf_url": "str", "dest_dir": "str"},
            output_schema={"status": "str", "local_path": "str", "detected_type": "str"},
            risk_level="medium", permissions=["net"],
            timeout_default=40, cache_policy="session",
        ),
        arxiv_download_pdf,
    )
    registry.register(
        ToolDefinition(
            name="semantic_scholar_search", version="1.0.0",
            description="Search Semantic Scholar for academic papers",
            input_schema={"query": "str", "max_results": "int"},
            output_schema={"results": "list[dict]"},
            risk_level="medium", permissions=["net"],
            timeout_default=30, cache_policy="session",
        ),
        semantic_scholar_search,
    )
    registry.register(
        ToolDefinition(
            name="wikipedia_search", version="1.0.0",
            description="Search Wikipedia for background knowledge summaries",
            input_schema={"query": "str", "max_results": "int"},
            output_schema={"results": "list[dict]"},
            risk_level="low", permissions=["net"],
            timeout_default=20, cache_policy="session",
        ),
        wikipedia_search,
    )
