"""Local vector indexing and top-k passage retrieval."""

from __future__ import annotations

from typing import Any

from vrag.chunker import EmbedderFn, dot, embed_texts, l2_normalize
from vrag.models import Chunk, RetrievedPassage

# --- Local Vector Index & Retrieval ---


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
        """Encode passages or queries using the injected embedder or local MiniLM."""
        return embed_texts(texts, self._embed_fn)

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks into local FAISS IndexFlatIP or normalized vector memory."""
        self._chunks = list(chunks)
        self._chunk_map = {c.chunk_id: c for c in self._chunks}
        self._faiss_index = None
        if not self._chunks:
            self._normalized_embeddings = []
            return

        # L2-normalize once so inner-product search equals cosine similarity.
        self._normalized_embeddings = [
            l2_normalize(vec) for vec in self._encode([c.text for c in self._chunks])
        ]

        try:
            import faiss  # type: ignore[import-untyped]
            import numpy as np

            if hasattr(faiss, "omp_set_num_threads"):
                faiss.omp_set_num_threads(1)
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
        query_vec = l2_normalize(self._encode([query])[0])

        if self._faiss_index is not None:
            import numpy as np

            scores, indices = self._faiss_index.search(
                np.asarray([query_vec], dtype=np.float32),
                k,
            )
            return [
                RetrievedPassage(chunk=self._chunks[int(idx)], score=float(score))
                for score, idx in zip(scores[0], indices[0])
                if int(idx) >= 0
            ]

        scored = [
            RetrievedPassage(chunk=chunk, score=dot(query_vec, emb))
            for chunk, emb in zip(self._chunks, self._normalized_embeddings)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]
