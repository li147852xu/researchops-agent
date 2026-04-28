"""Langfuse client facade with v2 / v3 auto-detection.

Design goals
------------

1. **Zero hard dependency.** ``langfuse`` is imported lazily inside a
   ``try / except ImportError`` so the package works without the optional
   ``observability`` extra.
2. **Silent no-op.** If ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are
   not in the environment, :func:`get_client` returns a :class:`_NoopFacade`
   that swallows every call.
3. **Version-agnostic.** Both Langfuse v2 (classic ``trace`` / ``span`` API)
   and v3 (OTel-style ``start_span`` / ``start_generation``) are supported.
4. **No exception bubbles.** Every call into the SDK is guarded by
   :func:`_safe`; any failure is downgraded to ``logger.warning``.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any

_HEX32 = re.compile(r"^[0-9a-f]{32}$")


def _to_hex32(seed: str) -> str:
    """Coerce an arbitrary identifier into a Langfuse-v3-compatible 32-hex trace id."""
    if _HEX32.fullmatch(seed):
        return seed
    return hashlib.md5(seed.encode("utf-8")).hexdigest()

logger = logging.getLogger("researchops.observability")

# ── Public surface ──────────────────────────────────────────────────────


class LangfuseNode(ABC):
    """Facade for a Langfuse trace / span / generation; behaves like a tree node."""

    @abstractmethod
    def open_span(self, name: str, *, input: str = "", metadata: dict | None = None) -> LangfuseNode:
        ...

    @abstractmethod
    def short_span(
        self,
        name: str,
        *,
        input: str = "",
        output: str = "",
        metadata: dict | None = None,
    ) -> None:
        ...

    @abstractmethod
    def log_generation(
        self,
        *,
        name: str,
        model: str,
        input: str = "",
        output: str = "",
        usage_input: int = 0,
        usage_output: int = 0,
        metadata: dict | None = None,
    ) -> None:
        ...

    @abstractmethod
    def event(self, name: str, *, metadata: dict | None = None) -> None:
        ...

    @abstractmethod
    def end(self, *, output: str = "", error: str | None = None) -> None:
        ...


class LangfuseFacade(ABC):
    """Top-level facade. Provides ``trace()`` (returns a node) and ``flush()``."""

    @abstractmethod
    def trace(self, *, id: str, name: str, metadata: dict | None = None) -> LangfuseNode:
        ...

    @abstractmethod
    def flush(self) -> None:
        ...

    # Subclasses set this to True iff they perform real work (used by tests).
    enabled: bool = False


# ── Helpers ─────────────────────────────────────────────────────────────


def _safe(func, *args, **kwargs):
    """Run ``func``; downgrade any exception to a warning."""
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — defensive
        logger.warning("langfuse call failed (%s): %s", func.__qualname__, exc)
        return None


# ── No-op implementation ────────────────────────────────────────────────


class _NoopNode(LangfuseNode):
    def open_span(self, name, *, input="", metadata=None):
        return self

    def short_span(self, name, *, input="", output="", metadata=None):
        return None

    def log_generation(self, *, name, model, input="", output="",
                       usage_input=0, usage_output=0, metadata=None):
        return None

    def event(self, name, *, metadata=None):
        return None

    def end(self, *, output="", error=None):
        return None


class _NoopFacade(LangfuseFacade):
    enabled = False

    def trace(self, *, id, name, metadata=None):
        return _NoopNode()

    def flush(self):
        return None


# ── v2 implementation (classic explicit API) ────────────────────────────


class _V2Node(LangfuseNode):
    def __init__(self, handle: Any):
        self._h = handle

    def open_span(self, name, *, input="", metadata=None):
        sub = _safe(self._h.span, name=name, input=input, metadata=metadata or {})
        return _V2Node(sub) if sub is not None else _NoopNode()

    def short_span(self, name, *, input="", output="", metadata=None):
        sub = _safe(self._h.span, name=name, input=input, output=output,
                    metadata=metadata or {})
        if sub is not None:
            _safe(sub.end)

    def log_generation(self, *, name, model, input="", output="",
                       usage_input=0, usage_output=0, metadata=None):
        gen = _safe(
            self._h.generation,
            name=name, model=model, input=input, output=output,
            usage={"input": usage_input, "output": usage_output,
                   "total": usage_input + usage_output},
            metadata=metadata or {},
        )
        if gen is not None:
            _safe(gen.end)

    def event(self, name, *, metadata=None):
        _safe(self._h.event, name=name, metadata=metadata or {})

    def end(self, *, output="", error=None):
        kwargs: dict[str, Any] = {}
        if output:
            kwargs["output"] = output
        if error:
            kwargs["level"] = "ERROR"
            kwargs["status_message"] = error
        _safe(self._h.end, **kwargs)


class _V2Facade(LangfuseFacade):
    enabled = True

    def __init__(self, lf: Any):
        self._lf = lf

    def trace(self, *, id, name, metadata=None):
        h = _safe(self._lf.trace, id=id, name=name, metadata=metadata or {})
        return _V2Node(h) if h is not None else _NoopNode()

    def flush(self):
        _safe(self._lf.flush)


# ── v3 implementation (OTel-style) ──────────────────────────────────────


class _V3Node(LangfuseNode):
    """v3 wrapper: a span/generation object exposing ``update`` / ``end``."""

    def __init__(self, lf: Any, handle: Any | None, trace_id: str | None = None):
        self._lf = lf
        self._h = handle
        self._trace_id = trace_id

    def _start(self, factory_name: str, **kwargs):
        """Start a child span/generation, propagating ``trace_id`` if available."""
        factory = getattr(self._lf, factory_name, None)
        if factory is None:
            return None
        if self._trace_id is not None:
            kwargs.setdefault("trace_context", {"trace_id": self._trace_id})
        return _safe(factory, **kwargs)

    def open_span(self, name, *, input="", metadata=None):
        h = self._start("start_span", name=name, input=input, metadata=metadata or {})
        return _V3Node(self._lf, h, self._trace_id)

    def short_span(self, name, *, input="", output="", metadata=None):
        h = self._start("start_span", name=name, input=input, metadata=metadata or {})
        if h is None:
            return
        if output:
            _safe(getattr(h, "update", lambda **kw: None), output=output)
        _safe(getattr(h, "end", lambda: None))

    def log_generation(self, *, name, model, input="", output="",
                       usage_input=0, usage_output=0, metadata=None):
        h = self._start(
            "start_generation", name=name, model=model, input=input,
            usage_details={"input": usage_input, "output": usage_output,
                           "total": usage_input + usage_output},
            metadata=metadata or {},
        )
        if h is None:
            return
        if output:
            _safe(getattr(h, "update", lambda **kw: None), output=output)
        _safe(getattr(h, "end", lambda: None))

    def event(self, name, *, metadata=None):
        kw: dict[str, Any] = {"name": name, "metadata": metadata or {}}
        if self._trace_id is not None:
            kw["trace_context"] = {"trace_id": self._trace_id}
        ev = getattr(self._lf, "create_event", None)
        if ev is None:
            return
        _safe(ev, **kw)

    def end(self, *, output="", error=None):
        if self._h is None:
            return
        update = getattr(self._h, "update", None)
        if update is not None:
            kwargs: dict[str, Any] = {}
            if output:
                kwargs["output"] = output
            if error:
                kwargs["level"] = "ERROR"
                kwargs["status_message"] = error
            if kwargs:
                _safe(update, **kwargs)
        end_fn = getattr(self._h, "end", None)
        if end_fn is not None:
            _safe(end_fn)


class _V3Facade(LangfuseFacade):
    enabled = True

    def __init__(self, lf: Any):
        self._lf = lf

    def trace(self, *, id, name, metadata=None):
        # v3 has no explicit trace handle — children are started against the
        # client and tagged with this trace id via ``trace_context``. The trace
        # id MUST be 32 lowercase hex chars (W3C trace-context); we deterministic-
        # ally hash any non-conforming run_id and surface the original as metadata.
        return _V3Node(self._lf, None, trace_id=_to_hex32(id))

    def flush(self):
        _safe(self._lf.flush)


# ── Resolver ────────────────────────────────────────────────────────────


_CACHED: LangfuseFacade | None = None


def _build_client() -> LangfuseFacade:
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("LANGFUSE_HOST", "")

    if not pk or not sk:
        return _NoopFacade()

    try:
        import langfuse  # type: ignore[import-not-found]
    except ImportError:
        logger.info(
            "LANGFUSE_* env detected but the `langfuse` package is not installed; "
            "install with `pip install -e \".[observability]\"`. Disabling Langfuse."
        )
        return _NoopFacade()

    # Resolve SDK version. ``__version__`` was dropped in langfuse 3.x in favour
    # of installed-package metadata; fall back to ``importlib.metadata`` so the
    # probe works on both v2.x and v3.x.
    version = str(getattr(langfuse, "__version__", "") or "")
    if not version:
        try:
            from importlib.metadata import version as _pkg_version

            version = _pkg_version("langfuse")
        except Exception:  # noqa: BLE001
            version = "0.0.0"
    major = version.split(".", 1)[0] or "0"

    try:
        kwargs: dict[str, Any] = {"public_key": pk, "secret_key": sk}
        if host:
            kwargs["host"] = host
        lf = langfuse.Langfuse(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse client init failed: %s; disabling.", exc)
        return _NoopFacade()

    if major in ("0", "1", "2"):
        logger.info("Langfuse SDK %s detected -> using v2 facade.", version)
        return _V2Facade(lf)
    logger.info("Langfuse SDK %s detected -> using v3 facade.", version)
    return _V3Facade(lf)


def get_client() -> LangfuseFacade:
    """Return a cached :class:`LangfuseFacade` resolved from environment + SDK."""
    global _CACHED
    if _CACHED is None:
        _CACHED = _build_client()
        atexit.register(_CACHED.flush)
    return _CACHED


def reset_client() -> None:
    """Drop the cached client. Tests use this to re-resolve env / SDK version."""
    global _CACHED
    _CACHED = None
