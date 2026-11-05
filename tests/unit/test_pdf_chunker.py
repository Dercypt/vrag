"""Unit tests for PDF page extraction and multi-page document chunking."""

from __future__ import annotations

import pytest

from vrag.chunker import (
    chunk_pages_fixed,
    chunk_pages_semantic,
    extract_pdf_pages,
    fixed_window_chunk,
    semantic_cosine_chunk,
)
from vrag.models import PDFPage


def test_extract_pdf_pages_from_formfeed_bytes_preserves_page_numbers() -> None:
    raw_doc = b"Page one introduction text.\fPage two methodology text.\f   \fPage four results."
    pages = extract_pdf_pages(raw_doc)
    assert [p.page_number for p in pages] == [1, 2, 4]
    assert pages[0].text == "Page one introduction text."
    assert pages[1].text == "Page two methodology text."
    assert pages[2].text == "Page four results."


def test_chunk_pages_fixed_assigns_global_sequential_chunk_ids_and_pages() -> None:
    pages = [
        PDFPage(page_number=1, text="one two three four five"),
        PDFPage(page_number=2, text="six seven eight"),
    ]
    chunks = chunk_pages_fixed(pages, token_size=3)
    assert [(c.chunk_id, c.page_number, c.text) for c in chunks] == [
        ("C1", 1, "one two three"),
        ("C2", 1, "four five"),
        ("C3", 2, "six seven eight"),
    ]


def test_chunk_pages_semantic_assigns_global_chunk_ids_across_pages() -> None:
    pages = [
        PDFPage(page_number=1, text="First sentence on page one. Second dissimilar sentence."),
        PDFPage(page_number=2, text="Page two sentence A. Page two similar sentence B."),
    ]

    def fake_embed(sents: list[str]) -> list[list[float]]:
        if len(sents) == 2 and "First" in sents[0]:
            return [[1.0, 0.0], [0.0, 1.0]]
        return [[1.0, 0.0], [0.99, 0.01]]

    chunks = chunk_pages_semantic(pages, similarity_threshold=0.75, embed_fn=fake_embed)
    assert [(c.chunk_id, c.page_number) for c in chunks] == [
        ("C1", 1),
        ("C2", 1),
        ("C3", 2),
    ]
    assert "Page two sentence A. Page two similar sentence B." == chunks[2].text


def test_chunker_validation_edge_cases() -> None:
    with pytest.raises(ValueError, match="token_size must be positive"):
        fixed_window_chunk("some text", token_size=0)

    with pytest.raises(ValueError, match="similarity_threshold must be in"):
        semantic_cosine_chunk("Sentence one. Sentence two.", similarity_threshold=1.5)

    with pytest.raises(ValueError, match="similarity_threshold must be in"):
        semantic_cosine_chunk("Sentence one. Sentence two.", similarity_threshold=0.0)
