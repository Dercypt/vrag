"""Offline local Ollama LLM client enforcing LAW 2, LAW 3, and LAW 4."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from vrag.models import APPROVED_MODELS, PipelineMode, RetrievedPassage

ALLOWED_LOCAL_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})

HttpPostFn = Callable[[str, dict[str, Any]], dict[str, Any]]
HttpGetFn = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class OllamaHealthReport:
    """Diagnostic report on local Ollama daemon connectivity and available models."""

    is_online: bool
    base_url: str
    installed_models: tuple[str, ...]
    missing_approved_models: tuple[str, ...]
    message: str
    troubleshooting_steps: tuple[str, ...]


def _model_matches(approved: str, installed: str) -> bool:
    app_clean = approved.strip().lower()
    inst_clean = installed.strip().lower()
    return (
        app_clean == inst_clean
        or inst_clean.startswith(f"{app_clean}:")
        or (app_clean + ":latest") == inst_clean
    )


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


def diagnose_ollama(
    base_url: str = "http://localhost:11434",
    timeout: float = 1.5,
    http_get_fn: HttpGetFn | None = None,
) -> OllamaHealthReport:
    """Perform a comprehensive diagnostic check of the local Ollama service.

    Enforces LAW 2 offline loopback execution, checks daemon connectivity,
    and inspects installed models against the thesis-approved models.
    """
    clean_url = validate_offline_ollama_url(base_url)

    try:
        if http_get_fn is not None:
            data = http_get_fn(f"{clean_url}/api/tags")
        else:
            import httpx

            with httpx.Client(timeout=timeout, trust_env=False) as client:
                resp = client.get(f"{clean_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()

        raw_models = data.get("models", [])
        installed_names = tuple(
            str(m.get("name", "")) for m in raw_models if isinstance(m, dict) and m.get("name")
        )

        missing = tuple(
            app
            for app in APPROVED_MODELS
            if not any(_model_matches(app, inst) for inst in installed_names)
        )

        if not missing:
            msg = (
                f"Ollama is online at {clean_url} and all {len(APPROVED_MODELS)} "
                "thesis models are ready."
            )
            steps: tuple[str, ...] = ()
        else:
            msg = (
                f"Ollama is online at {clean_url}, but {len(missing)} of {len(APPROVED_MODELS)} "
                "approved models are missing."
            )
            steps = tuple(f"ollama pull {m}" for m in missing)

        return OllamaHealthReport(
            is_online=True,
            base_url=clean_url,
            installed_models=installed_names,
            missing_approved_models=missing,
            message=msg,
            troubleshooting_steps=steps,
        )

    except Exception as err:
        steps = (
            "Install Ollama: brew install --cask ollama (or download from https://ollama.com)",
            "Start the daemon in a terminal: ollama serve",
            f"Pull the required models: ollama pull {APPROVED_MODELS[0]}",
            (
                "Or tick 'Use Deterministic Simulation Mode (Fast Offline)' in the UI "
                "to run without Ollama."
            ),
        )
        return OllamaHealthReport(
            is_online=False,
            base_url=clean_url,
            installed_models=(),
            missing_approved_models=APPROVED_MODELS,
            message=f"Local Ollama daemon is unreachable at {clean_url} ({err}).",
            troubleshooting_steps=steps,
        )


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
