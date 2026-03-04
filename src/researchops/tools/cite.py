from __future__ import annotations

_COUNTER: dict[str, int] = {}


def cite(source_id: str = "", claim_id: str = "") -> dict:
    """Map source_id / claim_id to a citation marker like [1], [2], etc."""
    key = source_id or claim_id
    if not key:
        return {"marker": ""}
    if key not in _COUNTER:
        _COUNTER[key] = len(_COUNTER) + 1
    return {"marker": f"[{_COUNTER[key]}]"}


def reset_citations() -> None:
    _COUNTER.clear()
