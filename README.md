# vrag — Verification-Driven RAG Comparative Benchmarking Harness

> **Thesis**: *A Comparative Study on Reducing Hallucinations in Lightweight Open-Source Large Language Models Using a Verification-Driven RAG Pipeline*

`vrag` is an offline comparative benchmarking harness (CLI + lightweight Streamlit UI) designed for a 2019 MacBook Pro memory profile. It evaluates **Condition A (Base RAG)** against **Condition B (Verification-Driven RAG)** on uploaded student/academic PDFs across three 4-bit quantized 7B–8B local Ollama models:
- `llama3.1:8b-instruct-q4_K_M`
- `mistral:7b-instruct-v0.3-q4_K_M`
- `qwen2.5:7b-instruct-q4_K_M`

---

## Core Comparative Metrics (Base RAG vs. Verification-Driven RAG)

| Metric | Unit | Description |
| :--- | :---: | :--- |
| **1. Factual Precision (FActScore)** | `%` | Percentage of generated atomic claims actually supported by the uploaded PDF. |
| **2. Entailment Consistency (NLI Score)** | `%` | Accuracy of the verification gate in classifying sentences as supported vs. hallucinated (`TP` vs. `FP`). |
| **3. Citation Accuracy (Precision/Recall)** | `%` | Whether the model cites real authors and chunk/page numbers from the text without inventing fake ones. |
| **4. Computational Latency** | `s` | End-to-end execution time (in seconds) proving local efficiency on a 2019 MacBook Pro without freezing. |

---

## Architecture & Governance

1. [`LAWS.md`](LAWS.md) — 5 non-negotiable domain invariants (deterministic NLI & citation pruning gate, 100% offline execution, experimental parity, sequential single-model RAM safety, and statistical telemetry completeness).
2. [`PRINCIPLES.md`](PRINCIPLES.md) — Escalation triggers and pure, dependency-injected Python defaults.
3. [`HARNESS.md`](HARNESS.md) & [`verify.sh`](verify.sh) — Automated linting, formatting, law invariant tests (`tests/laws/`), and unit tests (`tests/unit/`).
4. [`AGENTS.md`](AGENTS.md) — Agent workflow and governance entrypoint.

---

## Quickstart

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ui]"
```

### 2. Run Verification Suite
```bash
./verify.sh
```

### 3. Launch Lightweight Student PDF Benchmark UI (Streamlit)
```bash
PYTHONPATH=src streamlit run src/vrag/ui.py
```

### 4. Run Deterministic Offline CLI Benchmark Demo
```bash
PYTHONPATH=src python3 -m vrag.cli --demo --output-dir outputs/demo
```

### 5. Benchmark a Local Student/Academic PDF via CLI
```bash
PYTHONPATH=src python3 -m vrag.cli --pdf path/to/student_paper.pdf --output-dir outputs/run1
```

---

## Offline Execution & Zero Internet Requirement

`vrag` and the Streamlit UI operate strictly under **LAW 2 (Offline & Loopback Execution)**:
- **No Internet Required**: Streamlit binds to local loopback (`localhost:8501`).
- **Deterministic Simulation Mode (Fast Offline)**: Runs immediately with no network access, zero downloads, and deterministic offline mock telemetry.
- **Resilient Ollama Auto-Fallback**: If the local Ollama daemon is unreachable or returns `[Errno 61] Connection refused`, the UI and CLI automatically fall back to Deterministic Simulation Mode with an informative notification so benchmark execution never crashes.
- **Live Local LLM Mode**: Uses the local Ollama daemon (`http://localhost:11434`) and local HuggingFace cache (`~/.cache/huggingface`), requiring zero outbound internet once local weights are present.
