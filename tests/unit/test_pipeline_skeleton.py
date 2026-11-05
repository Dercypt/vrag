"""Unit tests for vrag chunker and retriever skeleton modules."""

from __future__ import annotations

from vrag.chunker import fixed_window_chunk, semantic_cosine_chunk
from vrag.retriever import LocalVectorRetriever


def test_fixed_window_chunk_splits_by_token_budget() -> None:
    text = " ".join(f"word{i}" for i in range(10))
    chunks = fixed_window_chunk(text, token_size=4, page_number=3)
    assert len(chunks) == 3
    assert chunks[0].chunk_id == "C1"
    assert chunks[0].page_number == 3
    assert len(chunks[0].text.split()) == 4
    assert len(chunks[2].text.split()) == 2


def test_semantic_cosine_chunk_groups_similar_sentences() -> None:
    text = (
        "Transformers rely on self-attention mechanisms. "
        "Attention weights compute token interactions in parallel. "
        "By contrast, database indexing uses B-trees on disk."
    )
    # Provide deterministic mock embeddings: first two sentences colinear, third orthogonal
    mock_vectors = [
        [1.0, 0.0],
        [0.98, 0.02],
        [0.0, 1.0],
    ]
    chunks = semantic_cosine_chunk(
        text,
        similarity_threshold=0.75,
        page_number=2,
        embed_fn=lambda _sents: mock_vectors,
    )
    assert len(chunks) == 2
    assert "self-attention" in chunks[0].text
    assert "token interactions" in chunks[0].text
    assert "B-trees" in chunks[1].text


def test_local_vector_retriever_returns_top_k_ranked() -> None:
    chunks = fixed_window_chunk("alpha beta. gamma delta.", token_size=2)

    def fake_embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            if "gamma" in t:
                out.append([0.0, 1.0])
            else:
                out.append([1.0, 0.0])
        return out

    retriever = LocalVectorRetriever(embed_fn=fake_embed)
    retriever.index_chunks(chunks)
    results = retriever.retrieve("gamma query", top_k=1)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "C2"
