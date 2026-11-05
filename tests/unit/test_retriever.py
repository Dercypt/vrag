"""Unit tests for LocalVectorRetriever indexing, top-k ranking, and edge cases."""

from __future__ import annotations

import pytest

from vrag.models import Chunk
from vrag.retriever import LocalVectorRetriever


def test_retriever_ranks_by_normalized_cosine_and_preserves_page_metadata() -> None:
    chunks = [
        Chunk(chunk_id="C1", text="Transformer attention mechanism", page_number=1),
        Chunk(chunk_id="C2", text="Retrieval augmented generation", page_number=4),
        Chunk(chunk_id="C3", text="NLI cross encoder entailment", page_number=7),
    ]
    vectors = {
        "Transformer attention mechanism": [3.0, 0.0],
        "Retrieval augmented generation": [2.0, 2.0],
        "NLI cross encoder entailment": [0.0, 5.0],
        "query nli": [0.0, 1.0],
    }

    retriever = LocalVectorRetriever(embed_fn=lambda texts: [vectors[t] for t in texts])
    retriever.index_chunks(chunks)

    assert retriever.size == 3
    assert retriever.get_chunk_by_id("C2") == chunks[1]
    assert retriever.get_chunk_by_id("C99") is None

    top_passages = retriever.retrieve("query nli", top_k=2)
    assert len(top_passages) == 2
    assert top_passages[0].chunk.chunk_id == "C3"
    assert top_passages[0].chunk.page_number == 7
    assert pytest.approx(top_passages[0].score, rel=1e-5) == 1.0
    assert top_passages[1].chunk.chunk_id == "C2"
    assert pytest.approx(top_passages[1].score, rel=1e-5) == 0.70710678


def test_retriever_handles_empty_index_zero_vectors_and_invalid_top_k() -> None:
    retriever = LocalVectorRetriever(embed_fn=lambda texts: [[0.0, 0.0] for _ in texts])
    assert retriever.retrieve("any query", top_k=3) == []

    retriever.index_chunks([Chunk(chunk_id="C1", text="zero vec chunk", page_number=1)])
    results = retriever.retrieve("any query", top_k=10)
    assert len(results) == 1
    assert results[0].score == 0.0

    with pytest.raises(ValueError, match="top_k must be positive"):
        retriever.retrieve("any query", top_k=0)


def test_retriever_import_error_provides_actionable_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    orig_import = builtins.__import__

    def mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if "sentence_transformers" in name:
            raise ImportError(
                "tokenizers>=0.22.0,<=0.23.0 is required for a normal functioning of this module, "
                "but found tokenizers==0.21.1."
            )
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    retriever = LocalVectorRetriever(embed_fn=None)
    with pytest.raises(
        ImportError,
        match="Ensure tokenizers and transformers versions are compatible",
    ):
        retriever._encode(["test text"])
