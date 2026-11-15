"""Document chunking strategies for Condition A (fixed window) and Condition B (semantic cosine)."""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vrag.models import Chunk, PDFPage

EmbedderFn = Callable[[list[str]], list[list[float]]]

# --- Cached NLP & Embedding Singletons ---


@functools.lru_cache(maxsize=1)
def get_cached_minilm_model(cls: Any) -> Any:
    """Cache all-MiniLM-L6-v2 singleton on CPU per PRINCIPLES.md."""
    return cls("all-MiniLM-L6-v2", device="cpu")


@functools.lru_cache(maxsize=1)
def _get_cached_segmenter(pysbd_mod: Any) -> Any:
    """Cache pysbd Segmenter so ~50 boundary regexes are compiled once per process."""
    return pysbd_mod.Segmenter(language="en", clean=False)


def l2_normalize(vec: list[float]) -> list[float]:
    """Return unit-length L2 vector, or zero vector when norm is 0."""
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0.0 else [0.0 for _ in vec]


def dot(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute inner product between two dense vectors."""
    return sum(a * b for a, b in zip(vec_a, vec_b))


def embed_texts(texts: list[str], embed_fn: EmbedderFn | None = None) -> list[list[float]]:
    """Encode texts with an injectable embedder or the local CPU MiniLM singleton."""
    if embed_fn is not None:
        return embed_fn(texts)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as err:
        raise ImportError(
            f"Failed to import sentence-transformers / transformers: {err}. "
            "Ensure tokenizers and transformers versions are compatible "
            "(e.g., pip install 'tokenizers>=0.22.0,<=0.23.0' 'huggingface-hub>=0.34.0,<1.0' "
            "or pass an injectable embed_fn / use Simulation Mode)."
        ) from err

    model = get_cached_minilm_model(SentenceTransformer)
    return [row.tolist() for row in model.encode(texts, convert_to_numpy=True)]


# --- PDF Extraction & Sentence Segmentation ---


def _read_source_bytes(source: bytes | str | Path) -> bytes:
    """Normalize PDF bytes, filesystem paths, or raw text into bytes."""
    if isinstance(source, Path):
        return source.read_bytes()
    if isinstance(source, str):
        candidate = Path(source)
        return candidate.read_bytes() if candidate.is_file() else source.encode("utf-8")
    return source


def extract_pdf_pages(source: bytes | str | Path) -> list[PDFPage]:
    """Extract per-page text from PDF bytes/path via PyMuPDF (fitz) or form-feed text fallback."""
    raw_bytes = _read_source_bytes(source)

    if raw_bytes.startswith(b"%PDF"):
        try:
            import fitz  # type: ignore[import-untyped]
        except (ImportError, ModuleNotFoundError) as err:
            raise ModuleNotFoundError(
                "PyMuPDF ('fitz') is required to parse PDF documents. "
                "Please install it using 'pip install pymupdf' or 'pip install -e .'."
            ) from err

        try:
            with fitz.open(stream=raw_bytes, filetype="pdf") as doc:
                return [
                    PDFPage(page_number=idx, text=txt)
                    for idx, page in enumerate(doc, start=1)
                    if (txt := page.get_text("text").strip())
                ]
        except Exception as err:
            if isinstance(err, (RuntimeError, ValueError)) or "broken document" in str(err).lower():
                raise ValueError(f"Failed to parse corrupt PDF document: {err}") from err
            raise

    # Form-feed (\f) fallback enables lightweight offline testing without PDF binaries.
    raw_pages = raw_bytes.decode("utf-8", errors="replace").split("\f")
    return [
        PDFPage(page_number=idx, text=cleaned)
        for idx, page_text in enumerate(raw_pages, start=1)
        if (cleaned := page_text.strip())
    ]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using cached pysbd if installed, falling back to regex."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    try:
        import pysbd  # type: ignore[import-untyped]

        return [s.strip() for s in _get_cached_segmenter(pysbd).segment(cleaned) if s.strip()]
    except ImportError:
        return [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", cleaned) if p.strip()]


# --- Chunking Strategies (Condition A & Condition B) ---


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
    return [
        Chunk(
            chunk_id=f"C{start_index + offset}",
            text=" ".join(tokens[start : start + token_size]),
            page_number=page_number,
        )
        for offset, start in enumerate(range(0, len(tokens), token_size))
    ]


def _group_sentences_by_similarity(
    tagged_sentences: list[tuple[int, str]],
    similarity_threshold: float,
    embed_fn: EmbedderFn | None,
    start_index: int = 1,
) -> list[Chunk]:
    """Group adjacent (page_number, sentence) pairs whose cosine similarity meets threshold."""
    if not (0.0 < similarity_threshold <= 1.0):
        raise ValueError("similarity_threshold must be in (0.0, 1.0]")
    if not tagged_sentences:
        return []

    # Pre-normalize embeddings once so each boundary comparison is a single dot product.
    normed = [l2_normalize(v) for v in embed_texts([s for _, s in tagged_sentences], embed_fn)]

    chunks: list[Chunk] = []
    current_page, first_sent = tagged_sentences[0]
    current_group = [first_sent]

    for idx in range(1, len(tagged_sentences)):
        if dot(normed[idx - 1], normed[idx]) >= similarity_threshold:
            current_group.append(tagged_sentences[idx][1])
            continue
        chunks.append(
            Chunk(
                chunk_id=f"C{start_index + len(chunks)}",
                text=" ".join(current_group),
                page_number=current_page,
            )
        )
        current_page, sent = tagged_sentences[idx]
        current_group = [sent]

    chunks.append(
        Chunk(
            chunk_id=f"C{start_index + len(chunks)}",
            text=" ".join(current_group),
            page_number=current_page,
        )
    )
    return chunks


def semantic_cosine_chunk(
    pdf_text: str,
    similarity_threshold: float = 0.75,
    page_number: int = 1,
    embed_fn: EmbedderFn | None = None,
    start_index: int = 1,
) -> list[Chunk]:
    """Condition B (Verification-Driven RAG): Dynamic sentence-level cosine-similarity chunking."""
    tagged = [(page_number, sent) for sent in split_sentences(pdf_text)]
    return _group_sentences_by_similarity(
        tagged_sentences=tagged,
        similarity_threshold=similarity_threshold,
        embed_fn=embed_fn,
        start_index=start_index,
    )


def chunk_pages_fixed(pages: list[PDFPage], token_size: int = 256) -> list[Chunk]:
    """Chunk a list of PDFPages using Condition A fixed-window token chunking."""
    all_chunks: list[Chunk] = []
    for page in pages:
        all_chunks.extend(
            fixed_window_chunk(
                pdf_text=page.text,
                token_size=token_size,
                page_number=page.page_number,
                start_index=len(all_chunks) + 1,
            )
        )
    return all_chunks


def chunk_pages_semantic(
    pages: list[PDFPage],
    similarity_threshold: float = 0.75,
    embed_fn: EmbedderFn | None = None,
    cross_page: bool = True,
) -> list[Chunk]:
    """Chunk PDFPages using Condition B sentence-level semantic cosine chunking."""
    if not (0.0 < similarity_threshold <= 1.0):
        raise ValueError("similarity_threshold must be in (0.0, 1.0]")

    if cross_page:
        tagged = [(p.page_number, sent) for p in pages for sent in split_sentences(p.text)]
        return _group_sentences_by_similarity(
            tagged_sentences=tagged,
            similarity_threshold=similarity_threshold,
            embed_fn=embed_fn,
        )

    all_chunks: list[Chunk] = []
    for page in pages:
        all_chunks.extend(
            semantic_cosine_chunk(
                pdf_text=page.text,
                similarity_threshold=similarity_threshold,
                page_number=page.page_number,
                embed_fn=embed_fn,
                start_index=len(all_chunks) + 1,
            )
        )
    return all_chunks
