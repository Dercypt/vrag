# PRINCIPLES.md — Engineering Principles, Defaults & Escalation Triggers

> **Purpose**: Define engineering defaults and clear boundaries for autonomous agent decisions versus mandatory user escalation.

---

## 1. Escalation Triggers (Stop & Ask the User)

An agent MUST pause and ask for explicit approval before making any of the following changes:

1. **Dependencies & Stack**:
   - Adding, removing, or replacing any third-party package in `pyproject.toml` beyond the approved thesis stack (`pymupdf`, `pysbd`, `sentence-transformers`, `faiss-cpu`, `scipy`, `numpy`, `typer`, `rich`, `httpx`, `psutil`, `pytest`, `ruff`).
2. **Experimental Constants & Thresholds**:
   - Altering default experimental parameters:
     - Baseline chunk size (`token_size = 256`)
     - Semantic cosine similarity threshold (`similarity_threshold = 0.75`)
     - NLI entailment threshold (`entailment_threshold = 0.70`)
     - Evaluated Ollama models (`llama3.1:8b-instruct-q4_K_M`, `mistral:7b-instruct-v0.3-q4_K_M`, `qwen2.5:7b-instruct-q4_K_M`)
3. **Public Module Contracts & Telemetry Schema**:
   - Changing public function signatures in `chunker.py`, `retriever.py`, `generator.py`, `verifier.py`, or `evaluator.py`.
   - Modifying CSV/JSON column headers used for thesis statistical tables.
4. **Domain Laws**:
   - Modifying or relaxing any invariant in `LAWS.md` or disabling any test in `tests/laws/`.

---

## 2. Code & Architectural Defaults

### Deterministic & Pure Pipeline Design
- **Dependency Injection for Heavy Models**: All functions touching HuggingFace models (`all-MiniLM-L6-v2`, `cross-encoder/nli-deberta-v3-small`) or the local Ollama server (`http://localhost:11434`) must accept injectable protocol/callable overrides so that `tests/laws/` and `tests/unit/` run deterministically in milliseconds without requiring live 8B weights on every edit.
- **Immutable Data Contracts**: Use `@dataclass(frozen=True)` in `src/vrag/models.py` for passing state across pipeline stages (`Chunk`, `RetrievedPassage`, `ClaimVerificationResult`, `ConfusionMatrix`, `StageLatencies`, `TrialTelemetry`).

### Style & Static Verification
- **Formatting & Linting**: Enforced via `ruff check` and `ruff format --check` (line length `100`).
- **Type Annotations**: Every public function and method must carry complete Python type annotations (`from __future__ import annotations`).
- **Error Handling**: Fail loudly with descriptive `ValueError` or custom domain exceptions when invariants (e.g., non-local Ollama URL, negative latency, invalid metric bounds) are breached.

### Resource Hygiene (2019 MacBook Pro)
- Avoid loading both `all-MiniLM-L6-v2` and `cross-encoder/nli-deberta-v3-small` until needed, or cache singletons cleanly on CPU (`device="cpu"`).
- Always set `keep_alive=0` when unloading an Ollama model after completing its benchmark sweep.
