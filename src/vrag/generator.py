"""Offline local Ollama LLM client enforcing LAW 2, LAW 3, and LAW 4."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from vrag.models import APPROVED_MODELS, PipelineMode, RetrievedPassage

ALLOWED_LOCAL_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})

HttpPostFn = Callable[[str, dict[str, Any]], dict[str, Any]]


def validate_offline_ollama_url(base_url: str) -> str:
    """Enforce LAW 2: Reject any non-loopback LLM endpoint."""
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").lower()
    if hostname not in ALLOWED_LOCAL_HOSTS:
        raise ValueError(
            f"LAW 2 violation: External host '{hostname}' is forbidden. "
            "Only localhost / 127.0.0.1 is permitted for offline execution."
        )
    return base_url.rstrip("/")


def check_ollama_status(base_url: str = "http://localhost:11434", timeout: float = 1.5) -> bool:
    """Enforce LAW 2 and check whether the local Ollama daemon is reachable."""
    clean_url = validate_offline_ollama_url(base_url)
    try:
        import httpx

        with httpx.Client(timeout=timeout, trust_env=False) as client:
            resp = client.get(f"{clean_url}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


def _default_http_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send synchronous JSON POST to local Ollama daemon."""
    validate_offline_ollama_url(url)
    import httpx

    with httpx.Client(timeout=180.0, trust_env=False) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data


def build_rag_prompt(
    query: str,
    passages: list[RetrievedPassage],
    mode: PipelineMode,
) -> str:
    """Construct bounded prompt for Condition A or Condition B."""
    context_blocks = "\n\n".join(
        f"[{p.chunk.chunk_id}] (Page {p.chunk.page_number}): {p.chunk.text}" for p in passages
    )
    if mode == PipelineMode.VERIFIED_B:
        return (
            "You are a factual academic summarization assistant. "
            "Use ONLY the provided context chunks below. "
            "Every single sentence MUST end with at least one inline chunk citation "
            "formatted as [C1], [C2], etc., and preserve exact (Author, Year) citations.\n\n"
            f"CONTEXT:\n{context_blocks}\n\n"
            f"QUERY: {query}\n\n"
            "VERIFIABLE SUMMARY:"
        )
    return (
        "Summarize the following context to answer the query concisely.\n\n"
        f"CONTEXT:\n{context_blocks}\n\n"
        f"QUERY: {query}\n\n"
        "SUMMARY:"
    )


class OllamaGenerator:
    """Sequential single-model local Ollama client (LAW 2, LAW 3, LAW 4)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        seed: int = 42,
        temperature: float = 0.0,
        http_post_fn: HttpPostFn | None = None,
    ) -> None:
        self.base_url = validate_offline_ollama_url(base_url)
        self.seed = seed
        self.temperature = temperature
        self._http_post_fn = http_post_fn
        self.active_model: str | None = None
        self.unload_history: list[str] = []

    def ensure_single_model_loaded(self, target_model: str) -> None:
        """Enforce LAW 4: Unload any previously active model before loading a new one."""
        if target_model not in APPROVED_MODELS:
            raise ValueError(f"Model '{target_model}' is not in APPROVED_MODELS: {APPROVED_MODELS}")
        if self.active_model is not None and self.active_model != target_model:
            self.unload_model(self.active_model)
        self.active_model = target_model

    def unload_model(self, model_name: str) -> None:
        """Send keep_alive=0 to free RAM on the 2019 MacBook Pro (LAW 4)."""
        self.unload_history.append(model_name)
        if self._http_post_fn is not None:
            self._http_post_fn(
                f"{self.base_url}/api/generate",
                {"model": model_name, "prompt": "", "keep_alive": 0, "stream": False},
            )
        if self.active_model == model_name:
            self.active_model = None

    def generate(
        self,
        model_name: str,
        query: str,
        passages: list[RetrievedPassage],
        mode: PipelineMode,
    ) -> str:
        """Generate a RAG summary from local Ollama using deterministic seed & temperature."""
        self.ensure_single_model_loaded(model_name)
        prompt = build_rag_prompt(query=query, passages=passages, mode=mode)
        post_fn = self._http_post_fn or _default_http_post
        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
            },
        }
        result = post_fn(f"{self.base_url}/api/generate", payload)
        return str(result.get("response", "")).strip()
