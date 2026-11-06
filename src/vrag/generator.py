"""Offline local Ollama LLM client enforcing LAW 2, LAW 3, and LAW 4."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    import httpx

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


_SHARED_CLIENT: httpx.Client | None = None


def get_shared_client() -> httpx.Client:
    """Module-level shared persistent client session for high-throughput local HTTP calls."""
    global _SHARED_CLIENT
    if _SHARED_CLIENT is None or getattr(_SHARED_CLIENT, "is_closed", False):
        import httpx

        transport = httpx.HTTPTransport(
            retries=3,
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
                keepalive_expiry=120.0,
            ),
        )
        _SHARED_CLIENT = httpx.Client(
            timeout=httpx.Timeout(180.0, connect=5.0),
            transport=transport,
            trust_env=False,
        )
    return _SHARED_CLIENT


def close_shared_client() -> None:
    """Close module-level shared persistent client."""
    global _SHARED_CLIENT
    if _SHARED_CLIENT is not None and not getattr(_SHARED_CLIENT, "is_closed", False):
        _SHARED_CLIENT.close()
        _SHARED_CLIENT = None


def _default_http_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send synchronous JSON POST to local Ollama daemon using persistent pooled connection."""
    validate_offline_ollama_url(url)
    client = get_shared_client()
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
            "Do not extrapolate or invent external facts. "
            "Every single sentence MUST end with at least one inline chunk citation "
            "formatted as [C1], [C2], etc., and preserve exact (Author, Year) citations.\n\n"
            f"CONTEXT:\n{context_blocks}\n\n"
            f"QUERY: {query}\n\n"
            "VERIFIABLE SUMMARY:"
        )
    return (
        "Summarize the following context to answer the query concisely. "
        "Cite supporting context chunks inline as [C1], [C2], etc., and include "
        "relevant (Author, Year) citations.\n\n"
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
        client: httpx.Client | None = None,
        timeout: float = 180.0,
        connect_timeout: float = 5.0,
    ) -> None:
        self.base_url = validate_offline_ollama_url(base_url)
        self.seed = seed
        self.temperature = temperature
        self._http_post_fn = http_post_fn
        self._timeout = timeout
        self._connect_timeout = connect_timeout
        self._client = client
        self._owns_client = client is None
        self.active_model: str | None = None
        self.unload_history: list[str] = []

    def get_client(self) -> httpx.Client:
        """Return or lazily initialize the persistent httpx.Client with connection pooling."""
        if self._client is None or getattr(self._client, "is_closed", False):
            import httpx

            transport = httpx.HTTPTransport(
                retries=3,
                limits=httpx.Limits(
                    max_keepalive_connections=10,
                    max_connections=20,
                    keepalive_expiry=120.0,
                ),
            )
            client_timeout = httpx.Timeout(self._timeout, connect=self._connect_timeout)
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=client_timeout,
                transport=transport,
                trust_env=False,
            )
            self._owns_client = True
        return self._client

    def close(self) -> None:
        """Close the persistent HTTP client session if owned by this instance."""
        if self._client is not None and not getattr(self._client, "is_closed", False):
            if self._owns_client:
                self._client.close()

    def __enter__(self) -> OllamaGenerator:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send JSON POST using this instance's persistent connection pool."""
        validate_offline_ollama_url(url)
        client = self.get_client()
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data

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
        payload = {"model": model_name, "prompt": "", "keep_alive": 0, "stream": False}
        if self._http_post_fn is not None:
            self._http_post_fn(f"{self.base_url}/api/generate", payload)
        elif self._client is not None or getattr(self, "_owns_client", False):
            # Only issue live HTTP unload if daemon connection is active/configured
            try:
                self._post_json(f"{self.base_url}/api/generate", payload)
            except Exception:
                pass
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
        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
            },
        }
        if self._http_post_fn is not None:
            result = self._http_post_fn(f"{self.base_url}/api/generate", payload)
        else:
            result = self._post_json(f"{self.base_url}/api/generate", payload)
        return str(result.get("response", "")).strip()
