from __future__ import annotations

from researchops.registry.manager import ToolRegistry
from researchops.registry.schema import ToolDefinition
from researchops.tools.arxiv_download import arxiv_download_pdf
from researchops.tools.arxiv_search import arxiv_search
from researchops.tools.cite import cite
from researchops.tools.fetch_page import fetch_page
from researchops.tools.parse_doc import parse_doc
from researchops.tools.sandbox_exec import sandbox_exec
from researchops.tools.web_search import web_search


def register_builtin_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolDefinition(
            name="web_search",
            version="1.0.0",
            description="Search the web for a query string",
            input_schema={"query": "str", "max_results": "int"},
            output_schema={"results": "list[dict]"},
            risk_level="medium",
            permissions=["net"],
            timeout_default=15,
            cache_policy="session",
        ),
        web_search,
    )

    registry.register(
        ToolDefinition(
            name="fetch",
            version="1.0.0",
            description="Download a URL to local storage with magic-number detection",
            input_schema={"url": "str", "dest_dir": "str"},
            output_schema={"status": "str", "local_path": "str", "detected_type": "str"},
            risk_level="medium",
            permissions=["net"],
            timeout_default=30,
            cache_policy="session",
        ),
        fetch_page,
    )

    registry.register(
        ToolDefinition(
            name="parse",
            version="1.0.0",
            description="Parse HTML or PDF to extract text with quality scoring",
            input_schema={"file_path": "str", "format": "str"},
            output_schema={"text": "str", "title": "str", "quality_score": "float"},
            risk_level="low",
            permissions=[],
            timeout_default=15,
            cache_policy="session",
        ),
        parse_doc,
    )

    registry.register(
        ToolDefinition(
            name="sandbox_exec",
            version="1.0.0",
            description="Execute a Python script in a sandboxed environment",
            input_schema={"script_path": "str", "timeout": "int", "allow_net": "bool"},
            output_schema={"exit_code": "int", "stdout": "str", "stderr": "str"},
            risk_level="high",
            permissions=["sandbox"],
            timeout_default=60,
            cache_policy="none",
        ),
        sandbox_exec,
    )

    registry.register(
        ToolDefinition(
            name="cite",
            version="1.0.0",
            description="Map source/claim IDs to citation markers",
            input_schema={"source_id": "str", "claim_id": "str"},
            output_schema={"marker": "str"},
            risk_level="low",
            permissions=[],
            timeout_default=5,
            cache_policy="persistent",
        ),
        cite,
    )

    registry.register(
        ToolDefinition(
            name="arxiv_search",
            version="1.0.0",
            description="Search arXiv for papers matching a query via Atom API",
            input_schema={"query": "str", "max_results": "int"},
            output_schema={"results": "list[dict]"},
            risk_level="medium",
            permissions=["net"],
            timeout_default=20,
            cache_policy="session",
        ),
        arxiv_search,
    )

    registry.register(
        ToolDefinition(
            name="arxiv_download_pdf",
            version="1.0.0",
            description="Download a PDF from arXiv with magic-number verification",
            input_schema={"pdf_url": "str", "dest_dir": "str"},
            output_schema={"status": "str", "local_path": "str", "detected_type": "str"},
            risk_level="medium",
            permissions=["net"],
            timeout_default=30,
            cache_policy="session",
        ),
        arxiv_download_pdf,
    )
