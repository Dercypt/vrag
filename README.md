# vrag — Verification-Driven RAG Comparative Benchmarking Harness

> **Thesis**: *A Comparative Study on Reducing Hallucinations in Lightweight Open-Source Large Language Models Using a Verification-Driven RAG Pipeline*

`vrag` is an offline, CLI-driven comparative benchmarking harness designed for macOS (2019 MacBook Pro memory profile). It evaluates **Condition A (Baseline Standard/Naive RAG)** against **Condition B (Proposed Verification-Driven RAG)** across three 4-bit quantized 7B–8B local Ollama models:
- `llama3.1:8b-instruct-q4_K_M`
- `mistral:7b-instruct-v0.3-q4_K_M`
- `qwen2.5:7b-instruct-q4_K_M`

---

## Architecture & Governance

1. [`LAWS.md`](LAWS.md) — 5 non-negotiable domain invariants (deterministic NLI & citation pruning gate, 100% offline execution, experimental parity, sequential single-model RAM safety, and statistical telemetry completeness).
2. [`PRINCIPLES.md`](PRINCIPLES.md) — Escalation triggers and pure, dependency-injected Python defaults.
3. [`HARNESS.md`](HARNESS.md) & [`verify.sh`](verify.sh) — Automated linting, formatting, law invariant tests (`tests/laws/`), and unit tests (`tests/unit/`).
4. [`AGENTS.md`](AGENTS.md) — Agent workflow and governance entrypoint.

---

## Quickstart

### 1. Run Verification Suite
```bash
./verify.sh
```

### 2. Run Deterministic Offline Benchmark Demo
```bash
PYTHONPATH=src python3 -m vrag.cli --demo --output-dir outputs/demo
```

### 3. Benchmark a Local Academic PDF via Ollama
```bash
PYTHONPATH=src python3 -m vrag.cli --pdf path/to/paper.pdf --query "Summarize the methodology and key findings." --output-dir outputs/run1
```
