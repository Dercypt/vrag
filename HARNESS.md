# HARNESS.md — Verification Pipeline & Execution Guide

> **Single Entrypoint**: `./verify.sh`
>
> All code changes must pass `./verify.sh` with exit code `0` before a task is considered complete.

---

## 1. Environment & Toolchain Strategy

To guarantee reproducible isolated execution on macOS without polluting system packages, `./verify.sh` uses a local virtual environment (`.venv`) managed via `uv` (if installed) or standard `python3 -m venv` fallback, with `PYTHONPATH=src`.

### Verification Stages in `./verify.sh`

| Stage | Command | Purpose |
| :--- | :--- | :--- |
| **1. Environment Bootstrap** | `.venv` activation / check | Ensures `pytest` and `ruff` are available in `.venv` or user `PATH`. |
| **2. Static Analysis (Ruff)** | `ruff check src/ tests/` | Catches unused imports, undefined names, bugbear issues, and style violations. |
| **3. Format Check (Ruff)** | `ruff format --check src/ tests/` | Ensures consistent code formatting across all modules. |
| **4. Domain Law Suite** | `pytest tests/laws/ -v -m law` | Verifies all 5 non-negotiable thesis invariants in `LAWS.md`. |
| **5. Unit Test Suite** | `pytest tests/unit/ -v` | Tests individual components (`chunker`, `retriever`, `generator`, `verifier`, `evaluator`). |

---

## 2. Running the Verification Harness

```bash
# Run full verification suite (lint + format + law tests + unit tests)
./verify.sh
```

### Installing Full Runtime Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,ui]"
```

> [!NOTE]
> PyMuPDF is imported via `import fitz`. It is part of the approved thesis stack in `pyproject.toml` (`pymupdf>=1.23.0`) for offline PDF ingestion (LAW 2). Ensure `pymupdf` is installed in your active environment.
> If using `transformers>=4.57`, ensure compatible tokenizer versions via `pip install "tokenizers>=0.22.0,<=0.23.0" "huggingface-hub>=0.34.0,<1.0"`.

### Offline Execution & Internet Connectivity

The entire system is designed for **100% offline execution** (LAW 2):
- **Streamlit Web UI**: Serves strictly over local loopback (`http://localhost:8501`) and requires zero internet. Features proactive daemon status indicators (`🟢 Online` / `⚠️ Offline`), light/dark theme toggle, custom metric cards with directional delta chips, side-by-side condition output comparison panels, and interactive claim verification logs.
- **Deterministic Simulation Mode (Fast Offline)**: Runs completely without internet or external model downloads, using purely local hash embeddings and mock verification.
- **Automatic Fallback on Connection Refusal**: If local Ollama is offline or returns `[Errno 61] Connection refused` while running the benchmark, execution automatically falls back to Deterministic Simulation Mode with a diagnostic notice rather than aborting.
- **Live Ollama + Local HF Mode**: Once Ollama models are pulled and HuggingFace weights (`all-MiniLM-L6-v2`, `cross-encoder/nli-deberta-v3-small`) are cached in `~/.cache/huggingface`, execution runs 100% locally on CPU/loopback with zero internet connectivity.

---

## 3. Module Architecture Map

| Module | Primary Responsibility | Enforced Laws |
| :--- | :--- | :--- |
| `src/vrag/models.py` | Immutable dataclasses (`Chunk`, `ClaimVerificationResult`, `ConfusionMatrix`, `TrialTelemetry`) | LAW 1, LAW 5 |
| `src/vrag/chunker.py` | `fixed_window_chunk(pdf_text, token_size=256)` & `semantic_cosine_chunk(pdf_text, similarity_threshold=0.75)` | LAW 3 |
| `src/vrag/retriever.py` | Local FAISS index construction and top-$k$ retrieval with `[Chunk_ID]` & page metadata | LAW 2, LAW 3 |
| `src/vrag/generator.py` | Offline Ollama HTTP client (`http://localhost:11434`), bounded citation prompts, sequential `keep_alive=0` unload | LAW 2, LAW 3, LAW 4 |
| `src/vrag/verifier.py` | `pysbd` sentence splitting, regex citation & entity check, `cross-encoder/nli-deberta-v3-small` pruning gate | LAW 1, LAW 2 |
| `src/vrag/evaluator.py` | Paired Condition A vs Condition B runner, stage timers ($t_{\text{chunk}}, t_{\text{retrieve}}, t_{\text{generate}}, t_{\text{verify}}$), 4 core metrics, `scipy.stats.ttest_rel` | LAW 3, LAW 4, LAW 5 |
| `src/vrag/cli.py` | CLI entrypoint for comparative tables & CSV/JSON export | LAW 5 |
| `src/vrag/ui.py` | Minimal Streamlit UI with light/dark theme toggle, custom 4-metric cards, side-by-side output comparison, and atomic claim verification logs | LAW 1, LAW 5 |
