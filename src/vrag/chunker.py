"""Document chunking strategies for Condition A (fixed window) and Condition B (semantic cosine)."""

from __future__ import annotations

import math
import re
from collections.abc import Callable

from vrag.models import Chunk

EmbedderFn = Callable[[list[str]], list[list[float]]]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using pysbd if installed, falling back to regex."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    try:
        import pysbd  # type: ignore[import-untyped]

        segmenter = pysbd.Segmenter(language="en", clean=False)
        return [s.strip() for s in segmenter.segment(cleaned) if s.strip()]
    except ImportError:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", cleaned)
        return [p.strip() for p in parts if p.strip()]


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def fixed_window_chunk(
    pdf_text: str,
    token_size: int = 256,
    page_number: int = 1,
) -> list[Chunk]:
    """Condition A (Baseline Standard RAG): Fixed-size whitespace token chunking."""
    if token_size <= 0:
        raise ValueError("token_size must be positive")
    tokens = pdf_text.split()
    if not tokens:
        return []

    chunks: list[Chunk] = []
    for idx, start in enumerate(range(0, len(tokens), token_size), start=1):
        window = " ".join(tokens[start : start + token_size])
        chunks.append(Chunk(chunk_id=f"C{idx}", text=window, page_number=page_number))
    return chunks


def semantic_cosine_chunk(
    pdf_text: str,
    similarity_threshold: float = 0.75,
    page_number: int = 1,
    embed_fn: EmbedderFn | None = None,
) -> list[Chunk]:
    """Condition B (Verification-Driven RAG): Dynamic sentence-level cosine-similarity chunking."""
    sentences = split_sentences(pdf_text)
    if not sentences:
        return []

    if embed_fn is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        raw_embeddings = model.encode(sentences, convert_to_numpy=True)
        embeddings: list[list[float]] = [row.tolist() for row in raw_embeddings]
    else:
        embeddings = embed_fn(sentences)

    chunks: list[Chunk] = []
    current_group: list[str] = [sentences[0]]

    for idx in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[idx - 1], embeddings[idx])
        if sim >= similarity_threshold:
            current_group.append(sentences[idx])
        else:
            chunk_id = f"C{len(chunks) + 1}"
            chunks.append(
                Chunk(chunk_id=chunk_id, text=" ".join(current_group), page_number=page_number)
            )
            current_group = [sentences[idx]]

    if current_group:
        chunk_id = f"C{len(chunks) + 1}"
        chunks.append(
            Chunk(chunk_id=chunk_id, text=" ".join(current_group), page_number=page_number)
        )

    return chunks
