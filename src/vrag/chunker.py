"""Document chunking strategies for Condition A (fixed window) and Condition B (semantic cosine)."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from pathlib import Path

from vrag.models import Chunk, PDFPage

EmbedderFn = Callable[[list[str]], list[list[float]]]


def extract_pdf_pages(source: bytes | str | Path) -> list[PDFPage]:
    """Extract per-page text from PDF bytes/path via PyMuPDF (fitz) or form-feed text fallback."""
    if isinstance(source, Path):
        raw_bytes = source.read_bytes()
    elif isinstance(source, str):
        path_candidate = Path(source)
        if path_candidate.exists() and path_candidate.is_file():
            raw_bytes = path_candidate.read_bytes()
        else:
            raw_bytes = source.encode("utf-8")
    else:
        raw_bytes = source

    if raw_bytes.startswith(b"%PDF"):
        import fitz  # type: ignore[import-untyped]

        pages: list[PDFPage] = []
        with fitz.open(stream=raw_bytes, filetype="pdf") as doc:
            for idx, page in enumerate(doc, start=1):
                txt = page.get_text("text").strip()
                if txt:
                    pages.append(PDFPage(page_number=idx, text=txt))
        return pages

    decoded = raw_bytes.decode("utf-8", errors="replace")
    raw_pages = decoded.split("\f")
    return [
        PDFPage(page_number=idx, text=page_text.strip())
        for idx, page_text in enumerate(raw_pages, start=1)
        if page_text.strip()
    ]


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
    start_index: int = 1,
) -> list[Chunk]:
    """Condition A (Baseline Standard RAG): Fixed-size whitespace token chunking."""
    if token_size <= 0:
        raise ValueError("token_size must be positive")
    tokens = pdf_text.split()
    if not tokens:
        return []

    chunks: list[Chunk] = []
    for offset, start in enumerate(range(0, len(tokens), token_size)):
        window = " ".join(tokens[start : start + token_size])
        chunk_id = f"C{start_index + offset}"
        chunks.append(Chunk(chunk_id=chunk_id, text=window, page_number=page_number))
    return chunks


def semantic_cosine_chunk(
    pdf_text: str,
    similarity_threshold: float = 0.75,
    page_number: int = 1,
    embed_fn: EmbedderFn | None = None,
    start_index: int = 1,
) -> list[Chunk]:
    """Condition B (Verification-Driven RAG): Dynamic sentence-level cosine-similarity chunking."""
    if not (0.0 < similarity_threshold <= 1.0):
        raise ValueError("similarity_threshold must be in (0.0, 1.0]")

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
            chunk_id = f"C{start_index + len(chunks)}"
            chunks.append(
                Chunk(chunk_id=chunk_id, text=" ".join(current_group), page_number=page_number)
            )
            current_group = [sentences[idx]]

    if current_group:
        chunk_id = f"C{start_index + len(chunks)}"
        chunks.append(
            Chunk(chunk_id=chunk_id, text=" ".join(current_group), page_number=page_number)
        )

    return chunks


def chunk_pages_fixed(
    pages: list[PDFPage],
    token_size: int = 256,
) -> list[Chunk]:
    """Chunk a list of PDFPages using Condition A fixed-window token chunking."""
    all_chunks: list[Chunk] = []
    for page in pages:
        page_chunks = fixed_window_chunk(
            pdf_text=page.text,
            token_size=token_size,
            page_number=page.page_number,
            start_index=len(all_chunks) + 1,
        )
        all_chunks.extend(page_chunks)
    return all_chunks


def chunk_pages_semantic(
    pages: list[PDFPage],
    similarity_threshold: float = 0.75,
    embed_fn: EmbedderFn | None = None,
) -> list[Chunk]:
    """Chunk a list of PDFPages using Condition B sentence-level semantic cosine chunking."""
    all_chunks: list[Chunk] = []
    for page in pages:
        page_chunks = semantic_cosine_chunk(
            pdf_text=page.text,
            similarity_threshold=similarity_threshold,
            page_number=page.page_number,
            embed_fn=embed_fn,
            start_index=len(all_chunks) + 1,
        )
        all_chunks.extend(page_chunks)
    return all_chunks
