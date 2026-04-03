from __future__ import annotations

from abc import ABC, abstractmethod


class RetrieverBase(ABC):
    @abstractmethod
    def index(self, claims: list[dict]) -> None:
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        ...

    @abstractmethod
    def retrieve_for_rq(self, rq_id: str, top_k: int = 10) -> list[dict]:
        ...


class NullRetriever(RetrieverBase):
    def index(self, claims: list[dict]) -> None:
        self._claims = claims

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        return self._claims[:top_k] if hasattr(self, "_claims") else []

    def retrieve_for_rq(self, rq_id: str, top_k: int = 10) -> list[dict]:
        if not hasattr(self, "_claims"):
            return []
        return [c for c in self._claims if rq_id in c.get("supports_rq", [])][:top_k]
