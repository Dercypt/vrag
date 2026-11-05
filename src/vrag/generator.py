"""Offline local Ollama LLM client enforcing LAW 2, LAW 3, and LAW 4."""

from __future__ import annotations

from urllib.parse import urlparse

from vrag.models import APPROVED_MODELS, PipelineMode, RetrievedPassage

ALLOWED_LOCAL_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})


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
    ) -> None:
        self.base_url = validate_offline_ollama_url(base_url)
        self.seed = seed
        self.temperature = temperature
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
        """Record model unload (keep_alive=0) to free RAM on the 2019 MacBook Pro."""
        self.unload_history.append(model_name)
        if self.active_model == model_name:
            self.active_model = None
