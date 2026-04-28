"""Optional Langfuse observability layer for ResearchOps.

This sub-package is loaded eagerly only by ``TraceLogger``; the heavyweight
``langfuse`` SDK itself is imported lazily so projects that don't install the
``observability`` extra never pay the import cost or risk an ``ImportError``.

Public API:

- :func:`get_client` — cached singleton facade (v2 / v3 / no-op auto-dispatch)
- :func:`reset_client` — drop the cache so tests can re-resolve env or version
- :class:`LangfuseFacade` — abstract base shared by every dispatch variant
"""

from __future__ import annotations

from researchops.core.observability.langfuse_client import (
    LangfuseFacade,
    get_client,
    reset_client,
)

__all__ = ["LangfuseFacade", "get_client", "reset_client"]
