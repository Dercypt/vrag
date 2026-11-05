"""Local vector indexing and top-k passage retrieval."""

from __future__ import annotations

import math
from collections.abc import Callable

from vrag.models import Chunk, RetrievedPassage

EmbedderFn = Callable[[list[str]], list[list[float]]]


def _cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class LocalVectorRetriever:
    """Indexes chunks locally (using FAISS when available, or exact cosine fallback)."""

    def __init__(self, embed_fn: EmbedderFn | None = None) -> None:
        self._embed_fn = embed_fn
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self._embed_fn is not None:
            return self._embed_fn(texts)
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        arr = model.encode(texts, convert_to_numpy=True)
        return [row.tolist() for row in arr]

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks into local memory."""
        self._chunks = list(chunks)
        if not self._chunks:
            self._embeddings = []
            return
        self._embeddings = self._encode([c.text for c in self._chunks])

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPassage]:
        """Retrieve top-k passages with their Chunk_ID and page numbers."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self._chunks:
            return []

        query_vec = self._encode([query])[0]
        scored = [
            RetrievedPassage(chunk=chunk, score=_cosine_sim(query_vec, emb))
            for chunk, emb in zip(self._chunks, self._embeddings)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
