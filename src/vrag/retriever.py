"""Local vector indexing and top-k passage retrieval."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from vrag.models import Chunk, RetrievedPassage

EmbedderFn = Callable[[list[str]], list[list[float]]]


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return [0.0 for _ in vec]
    return [v / norm for v in vec]


def _dot(vec_a: list[float], vec_b: list[float]) -> float:
    return sum(a * b for a, b in zip(vec_a, vec_b))


class LocalVectorRetriever:
    """Indexes chunks locally into FAISS IndexFlatIP or normalized inner-product memory."""

    def __init__(self, embed_fn: EmbedderFn | None = None) -> None:
        self._embed_fn = embed_fn
        self._chunks: list[Chunk] = []
        self._chunk_map: dict[str, Chunk] = {}
        self._normalized_embeddings: list[list[float]] = []
        self._faiss_index: Any = None

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)

    def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        """Retrieve an indexed chunk by its Chunk_ID (e.g. 'C1')."""
        return self._chunk_map.get(chunk_id)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self._embed_fn is not None:
            return self._embed_fn(texts)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise ImportError(
                f"Failed to import sentence-transformers / transformers: {err}. "
                "Ensure tokenizers and transformers versions are compatible "
                "(e.g., pip install 'tokenizers>=0.22.0,<=0.23.0' 'huggingface-hub>=0.34.0,<1.0' "
                "or pass an injectable embed_fn / use Simulation Mode)."
            ) from err

        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        arr = model.encode(texts, convert_to_numpy=True)
        return [row.tolist() for row in arr]

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks into local FAISS / normalized vector memory."""
        self._chunks = list(chunks)
        self._chunk_map = {c.chunk_id: c for c in self._chunks}
        self._faiss_index = None
        if not self._chunks:
            self._normalized_embeddings = []
            return

        raw_embeddings = self._encode([c.text for c in self._chunks])
        self._normalized_embeddings = [_l2_normalize(vec) for vec in raw_embeddings]

        try:
            import faiss  # type: ignore[import-untyped]
            import numpy as np

            mat = np.asarray(self._normalized_embeddings, dtype=np.float32)
            index = faiss.IndexFlatIP(mat.shape[1])
            index.add(mat)
            self._faiss_index = index
        except ImportError:
            self._faiss_index = None

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPassage]:
        """Retrieve top-k passages with their Chunk_ID and page numbers."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self._chunks:
            return []

        k = min(top_k, len(self._chunks))
        query_vec = _l2_normalize(self._encode([query])[0])

        if self._faiss_index is not None:
            import numpy as np

            q_mat = np.asarray([query_vec], dtype=np.float32)
            scores, indices = self._faiss_index.search(q_mat, k)
            return [
                RetrievedPassage(chunk=self._chunks[int(idx)], score=float(score))
                for score, idx in zip(scores[0], indices[0])
                if int(idx) >= 0
            ]

        scored = [
            RetrievedPassage(chunk=chunk, score=_dot(query_vec, emb))
            for chunk, emb in zip(self._chunks, self._normalized_embeddings)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]
