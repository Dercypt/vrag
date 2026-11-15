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

# --- Offline Loopback Validation & Diagnostics ---


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
        or f"{app_clean}:latest" == inst_clean
    )


def validate_offline_ollama_url(base_url: str) -> str:
    """Enforce LAW 2: Reject any non-loopback LLM endpoint."""
    hostname = (urlparse(base_url).hostname or "").lower()
    if hostname not in ALLOWED_LOCAL_HOSTS:
        raise ValueError(
            f"LAW 2 violation: External host '{hostname}' is forbidden. "
            "Only localhost / 127.0.0.1 is permitted for offline execution."
        )
    return base_url.rstrip("/")


def _create_httpx_client(
    *,
    base_url: str = "",
    timeout: float = 180.0,
    connect_timeout: float = 5.0,
    retries: int = 3,
    pooled: bool = True,
) -> httpx.Client:
    """Create a proxy-disabled httpx.Client with optional keep-alive connection pooling."""
    import httpx

    limits = (
        httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=120.0)
        if pooled
        else httpx.Limits()
    )
    kwargs: dict[str, Any] = {
        "timeout": httpx.Timeout(timeout, connect=connect_timeout),
        "transport": httpx.HTTPTransport(retries=retries, limits=limits),
        "trust_env": False,  # Ignore OS proxy env vars so loopback calls stay offline (LAW 2).
    }
    if base_url:
        kwargs["base_url"] = base_url
    return httpx.Client(**kwargs)


def _fetch_tags_response(
    clean_url: str,
    timeout: float,
    client: httpx.Client | None,
) -> httpx.Response:
    """Query local Ollama /api/tags using a provided client or a short-lived probe client."""
    import httpx

    timeout_cfg = httpx.Timeout(timeout, connect=min(timeout, 0.75))
    tags_url = f"{clean_url}/api/tags"
    if client is not None:
        return client.get(tags_url, timeout=timeout_cfg)
    with _create_httpx_client(
        timeout=timeout,
        connect_timeout=min(timeout, 0.75),
        retries=1,
        pooled=False,
    ) as probe:
        return probe.get(tags_url)


def check_ollama_status(
    base_url: str = "http://localhost:11434",
    timeout: float = 1.5,
    client: httpx.Client | None = None,
) -> bool:
    """Enforce LAW 2 and check whether the local Ollama daemon is reachable."""
    clean_url = validate_offline_ollama_url(base_url)
    try:
        return _fetch_tags_response(clean_url, timeout, client).status_code == 200
    except Exception:
        return False


def diagnose_ollama(
    base_url: str = "http://localhost:11434",
    timeout: float = 1.5,
    http_get_fn: HttpGetFn | None = None,
    client: httpx.Client | None = None,
) -> OllamaHealthReport:
    """Perform a diagnostic check of the local Ollama service and installed thesis models."""
    clean_url = validate_offline_ollama_url(base_url)

    try:
        if http_get_fn is not None:
            data = http_get_fn(f"{clean_url}/api/tags")
        else:
            resp = _fetch_tags_response(clean_url, timeout, client)
            resp.raise_for_status()
            data = resp.json()

        installed_names = tuple(
            str(m["name"]) for m in data.get("models", []) if isinstance(m, dict) and m.get("name")
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
        return OllamaHealthReport(
            is_online=False,
            base_url=clean_url,
            installed_models=(),
            missing_approved_models=APPROVED_MODELS,
            message=f"Local Ollama daemon is unreachable at {clean_url} ({err}).",
            troubleshooting_steps=(
                "Install Ollama: brew install --cask ollama (or download from https://ollama.com)",
                "Start the daemon in a terminal: ollama serve",
                f"Pull the required models: ollama pull {APPROVED_MODELS[0]}",
                (
                    "Or tick 'Use Deterministic Simulation Mode (Fast Offline)' in the UI "
                    "to run without Ollama."
                ),
            ),
        )


# --- Pooled HTTP Client Lifecycle ---

_SHARED_CLIENT: httpx.Client | None = None


def get_shared_client() -> httpx.Client:
    """Module-level shared persistent client session for high-throughput local HTTP calls."""
    global _SHARED_CLIENT
    if _SHARED_CLIENT is None or getattr(_SHARED_CLIENT, "is_closed", False):
        _SHARED_CLIENT = _create_httpx_client()
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
    resp = get_shared_client().post(url, json=payload)
    resp.raise_for_status()
    return dict(resp.json())


# --- Prompt Builder & Sequential Ollama Client ---


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
            self._client = _create_httpx_client(
                base_url=self.base_url,
                timeout=self._timeout,
                connect_timeout=self._connect_timeout,
            )
            self._owns_client = True
        return self._client

    def close(self) -> None:
        """Close the persistent HTTP client session if owned by this instance."""
        if (
            self._client is not None
            and not getattr(self._client, "is_closed", False)
            and self._owns_client
        ):
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
        resp = self.get_client().post(url, json=payload)
        resp.raise_for_status()
        return dict(resp.json())

    def _send_generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Route /api/generate calls through injected mock transport or pooled HTTP client."""
        endpoint = f"{self.base_url}/api/generate"
        if self._http_post_fn is not None:
            return self._http_post_fn(endpoint, payload)
        return self._post_json(endpoint, payload)

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
        if self._http_post_fn is not None or self._client is not None or self._owns_client:
            try:
                self._send_generate(payload)
            except Exception:
                # Best-effort unload: never block cleanup if daemon was stopped mid-sweep.
                if self._http_post_fn is not None:
                    raise
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
        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": build_rag_prompt(query=query, passages=passages, mode=mode),
            "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed},
        }
        result = self._send_generate(payload)
        return str(result.get("response", "")).strip()
