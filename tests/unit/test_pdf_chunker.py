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


def test_extract_pdf_pages_missing_fitz_raises_descriptive_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    orig_import = builtins.__import__

    def mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "fitz":
            raise ModuleNotFoundError("No module named 'fitz'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(
        ModuleNotFoundError,
        match="PyMuPDF.*fitz.*pip install pymupdf",
    ):
        extract_pdf_pages(b"%PDF-1.4 dummy header")


def test_extract_pdf_pages_with_fitz_parses_real_pdf(tmp_path: pytest.TempPathFactory) -> None:
    import fitz

    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "First page academic thesis text.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Second page evaluation findings.")
    pdf_bytes = doc.tobytes()
    doc.close()

    # Test raw bytes
    pages_from_bytes = extract_pdf_pages(pdf_bytes)
    assert len(pages_from_bytes) == 2
    assert pages_from_bytes[0].page_number == 1
    assert "First page academic thesis text." in pages_from_bytes[0].text
    assert pages_from_bytes[1].page_number == 2
    assert "Second page evaluation findings." in pages_from_bytes[1].text

    # Test file path input
    test_pdf_file = tmp_path / "sample.pdf"  # type: ignore[operator]
    test_pdf_file.write_bytes(pdf_bytes)
    pages_from_path = extract_pdf_pages(test_pdf_file)
    assert len(pages_from_path) == 2
    assert pages_from_path[0].text == pages_from_bytes[0].text


def test_extract_pdf_pages_handles_corrupt_pdf_bytes() -> None:
    with pytest.raises(ValueError, match="Failed to parse corrupt PDF document"):
        extract_pdf_pages(b"%PDF-corrupted-stream-garbage-bytes")


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
