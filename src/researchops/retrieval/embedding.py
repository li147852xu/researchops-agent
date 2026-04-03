"""Embedding-based retriever using sentence-transformers and cosine similarity."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from researchops.retrieval.base import RetrieverBase

logger = logging.getLogger(__name__)


class EmbeddingRetriever(RetrieverBase):
    """Dense retriever backed by a local SentenceTransformer model."""

    def __init__(self, run_dir: Path, model_name: str = "all-MiniLM-L6-v2"):
        self._run_dir = run_dir
        self._model_name = model_name
        self._claims: list[dict[str, Any]] = []
        self._embeddings: Any = None  # np.ndarray once indexed
        self._model: Any = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embedding retrieval. "
                "Install with: pip install 'researchops[embeddings]'"
            ) from exc

    def index(self, claims: list[dict[str, Any]]) -> None:
        import numpy as np

        self._claims = claims
        if not claims:
            self._embeddings = np.empty((0, 0))
            return
        self._ensure_model()
        texts = [c.get("text", "") for c in claims]
        self._embeddings = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        )
        self._save_meta()

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        import numpy as np

        if not self._claims or self._embeddings is None or self._embeddings.size == 0:
            return []
        self._ensure_model()
        q_emb = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        scores = (self._embeddings @ q_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self._claims[int(i)] for i in top_indices]

    def retrieve_for_rq(self, rq_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        filtered = [c for c in self._claims if rq_id in c.get("supports_rq", [])]
        if not filtered:
            return []
        if self._model is None:
            return filtered[:top_k]
        import numpy as np

        texts = [c.get("text", "") for c in filtered]
        embs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        q_emb = self._model.encode([rq_id], normalize_embeddings=True, show_progress_bar=False)
        scores = (embs @ q_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [filtered[int(i)] for i in top_indices]

    def encode_query(self, query: str) -> Any:
        """Encode a single query — useful for external relevance scoring."""
        self._ensure_model()
        return self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]

    def encode_texts(self, texts: list[str]) -> Any:
        """Encode a batch of texts."""
        self._ensure_model()
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def _save_meta(self) -> None:
        meta_path = self._run_dir / "embedding_index_meta.json"
        meta_path.write_text(
            json.dumps({
                "model": self._model_name,
                "claims_count": len(self._claims),
            }, indent=2),
            encoding="utf-8",
        )
