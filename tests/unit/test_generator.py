"""Unit tests for OllamaGenerator HTTP request payloads, bounded prompts, and model unloading."""

from __future__ import annotations

from typing import Any

import pytest

from vrag.generator import OllamaGenerator, build_rag_prompt
from vrag.models import Chunk, PipelineMode, RetrievedPassage


def test_build_rag_prompt_differs_by_condition_mode() -> None:
    passages = [
        RetrievedPassage(
            chunk=Chunk(
                chunk_id="C1", text="Vaswani et al. (2017) proposed attention.", page_number=2
            ),
            score=0.95,
        )
    ]
    prompt_a = build_rag_prompt("What is attention?", passages, PipelineMode.BASELINE_A)
    prompt_b = build_rag_prompt("What is attention?", passages, PipelineMode.VERIFIED_B)

    assert "[C1] (Page 2)" in prompt_a
    assert "[C1]" in prompt_a and "(Author, Year)" in prompt_a
    assert "SUMMARY:" in prompt_a
    assert "Use ONLY the provided context chunks below" not in prompt_a
    assert "Use ONLY the provided context chunks below" in prompt_b
    assert "Every single sentence MUST end with at least one inline chunk citation" in prompt_b
    assert "VERIFIABLE SUMMARY:" in prompt_b


def test_ollama_generator_sends_deterministic_options_and_unloads_previous_model() -> None:
    recorded_requests: list[tuple[str, dict[str, Any]]] = []

    def fake_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        recorded_requests.append((url, payload))
        if payload.get("keep_alive") == 0:
            return {"done": True}
        return {"response": "Vaswani et al. (2017) introduced attention [C1].", "done": True}

    gen = OllamaGenerator(
        base_url="http://localhost:11434",
        seed=42,
        temperature=0.0,
        http_post_fn=fake_post,
    )
    passages = [
        RetrievedPassage(
            chunk=Chunk(
                chunk_id="C1", text="Vaswani et al. (2017) introduced attention.", page_number=1
            ),
            score=0.91,
        )
    ]

    out1 = gen.generate(
        model_name="llama3.1:8b-instruct-q4_K_M",
        query="Summarize attention.",
        passages=passages,
        mode=PipelineMode.VERIFIED_B,
    )
    assert out1 == "Vaswani et al. (2017) introduced attention [C1]."
    assert len(recorded_requests) == 1
    assert recorded_requests[0][0] == "http://localhost:11434/api/generate"
    assert recorded_requests[0][1]["options"] == {"temperature": 0.0, "seed": 42}
    assert recorded_requests[0][1]["stream"] is False

    # Switching to mistral must trigger keep_alive=0 unload of llama3.1 first
    out2 = gen.generate(
        model_name="mistral:7b-instruct-v0.3-q4_K_M",
        query="Summarize attention.",
        passages=passages,
        mode=PipelineMode.BASELINE_A,
    )
    assert out2 == "Vaswani et al. (2017) introduced attention [C1]."
    assert len(recorded_requests) == 3
    unload_req = recorded_requests[1][1]
    assert unload_req["model"] == "llama3.1:8b-instruct-q4_K_M"
    assert unload_req["keep_alive"] == 0
    assert recorded_requests[2][1]["model"] == "mistral:7b-instruct-v0.3-q4_K_M"


def test_ollama_generator_rejects_unapproved_model() -> None:
    gen = OllamaGenerator(http_post_fn=lambda _u, _p: {"response": "ok"})
    with pytest.raises(ValueError, match="not in APPROVED_MODELS"):
        gen.ensure_single_model_loaded("gpt-4o-mini")


def test_ollama_generator_persistent_session_and_context_manager() -> None:
    gen = OllamaGenerator(base_url="http://127.0.0.1:11434")
    client1 = gen.get_client()
    assert client1 is not None
    assert gen.get_client() is client1  # Reuses same client instance (connection pooling)

    with gen:
        assert not client1.is_closed

    assert client1.is_closed


def test_ollama_generator_client_configuration_limits_and_timeout() -> None:
    gen = OllamaGenerator(
        base_url="http://127.0.0.1:11434",
        timeout=120.0,
        connect_timeout=4.0,
    )
    client = gen.get_client()
    try:
        assert client.timeout.connect == 4.0
        assert client.timeout.read == 120.0
    finally:
        gen.close()
    assert client.is_closed


def test_ollama_generator_generate_uses_persistent_client() -> None:
    import json

    import httpx

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"response": "Attention summary [C1]."})

    mock_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://127.0.0.1:11434",
    )
    gen = OllamaGenerator(base_url="http://127.0.0.1:11434", client=mock_client)
    passages = [
        RetrievedPassage(
            chunk=Chunk(chunk_id="C1", text="Attention mechanism details.", page_number=1),
            score=0.9,
        )
    ]

    res1 = gen.generate("llama3.1:8b-instruct-q4_K_M", "query 1", passages, PipelineMode.BASELINE_A)
    res2 = gen.generate("llama3.1:8b-instruct-q4_K_M", "query 2", passages, PipelineMode.VERIFIED_B)

    assert res1 == "Attention summary [C1]."
    assert res2 == "Attention summary [C1]."
    assert len(calls) == 2
    assert calls[0].url.path == "/api/generate"
    assert calls[1].url.path == "/api/generate"
    body1 = json.loads(calls[0].content.decode("utf-8"))
    assert body1["model"] == "llama3.1:8b-instruct-q4_K_M"
    assert gen.get_client() is mock_client


def test_ollama_generator_unload_uses_persistent_client() -> None:
    import json

    import httpx

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"done": True})

    mock_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://127.0.0.1:11434",
    )
    gen = OllamaGenerator(base_url="http://127.0.0.1:11434", client=mock_client)
    gen.ensure_single_model_loaded("llama3.1:8b-instruct-q4_K_M")
    gen.unload_model("llama3.1:8b-instruct-q4_K_M")

    assert len(calls) == 1
    assert calls[0].url.path == "/api/generate"
    data = json.loads(calls[0].content.decode("utf-8"))
    assert data["keep_alive"] == 0
    assert data["model"] == "llama3.1:8b-instruct-q4_K_M"
