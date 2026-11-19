# vrag — Verification-Driven RAG Comparative Benchmarking Harness

> **Thesis**: *A Comparative Study on Reducing Hallucinations in Lightweight Open-Source Large Language Models Using a Verification-Driven RAG Pipeline*

`vrag` is a minimal, 100% offline pure-Python benchmarking harness (CLI + native Streamlit UI) built for **standard 8 GB institutional laboratory desktops** (4.5–6.5 GB under 4-bit quantization). It isolates pipeline architecture by running **Condition A (Standard Fixed-Window RAG)** against **Condition B (Semantic Chunking + Bounded Drafting + Sentence-Level NLI Gate)** across four 7B–9B open-weight models using a single standardized summarization prompt:

| Model (Ollama 4-bit `q4_K_M`) | Vendor | Parameter Tier | HalluLens Baseline (Bang et al., 2025) |
| :--- | :--- | :---: | :---: |
| `llama3.1:8b-instruct-q4_K_M` | Meta | 8B | 48.37% |
| `gemma2:9b-instruct-q4_K_M` | Google | 9B | 76.01% |
| `mistral:7b-instruct-v0.3-q4_K_M` | Mistral AI | 7B | 81.19% |
| `qwen2.5:7b-instruct-q4_K_M` | Alibaba | 7B | 85.22% |

---

## Core Comparative Metrics (Chapter 3, pp. 73–75)

| Metric | Unit | Formula / Meaning |
| :--- | :---: | :--- |
| **1. Factual Precision (FActScore)** | `%` | Proportion of surviving atomic claims supported by the uploaded PDF (Min et al., 2023). |
| **2. Entailment Consistency (NLI Score)** | `%` | Verification gate classification accuracy: `(TP + TN) / (TP + TN + FP + FN)` (Honovich et al., 2022). |
| **3. Citation Accuracy (Precision/Recall)** | `%` | F1 of Citation Precision and Citation Recall verifying `[Chunk_ID]`, author surnames, and publication years (Qian et al., 2025). |
| **4. Computational Latency** | `s` | End-to-end time `t_chunk + t_retrieve + t_generate + t_verify` proving local hardware practicality (Zhang et al., 2026). |

---

## Why This Tech Stack? (Oral Defense Reference)

Everything runs in **one minimal Python codebase** with **zero paid APIs** and **zero GPU fine-tuning**:

| Component | Library / Model | Why Chosen (Manuscript Justification) |
| :--- | :--- | :--- |
| **PDF Ingestion** | `pymupdf` (`fitz`) | Fast, 100% offline digital PDF parser preserving page numbers for citation audit trails (Scope: digital prose PDFs). |
| **Sentence Splitting** | `pysbd` | Deterministic rule-based segmentation that handles academic abbreviations (`et al.`, `Fig.`, `Dr.`) without breaking citations (Min et al., 2023; Wang et al., 2024). |
| **Semantic Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Lightweight 22M-param CPU encoder (~90 MB RAM) enabling dynamic cosine-similarity chunking (`threshold=0.75`) without splitting arguments across arbitrary token windows (Wang et al., 2025; Brådland et al., 2025). |
| **Vector Index** | `faiss-cpu` (`IndexFlatIP`) | Exact normalized cosine retrieval on CPU in milliseconds—far lighter than LLM-built entity graphs in GraphRAG (Peng et al., 2025; Chen et al., 2026). |
| **Local LLM Runtime** | Ollama (`q4_K_M`) + `httpx` | Executes frozen black-box 7B–9B models offline over `localhost:11434` (`temperature=0.0, seed=42`) with `keep_alive=0` sequential unloading so only one model occupies 8 GB RAM at a time (Shi et al., 2023; Izacard et al., 2022). |
| **NLI Verification Gate** | `cross-encoder/nli-deberta-v3-small` | Compact CPU NLI model (~180 MB RAM) implementing TRUE/VeriCite sentence-level entailment pruning (`threshold=0.70`) alongside author/year verification (Honovich et al., 2022; Qian et al., 2025). |
| **Statistical Analysis** | `scipy` + `numpy` | Reproducible Shapiro-Wilk normality checks, Paired Student's *t*-test (`ttest_rel`), and Wilcoxon signed-rank fallback at $\alpha = 0.05$ (Chapter 3, p. 75). |
| **RAM Telemetry** | `psutil` | Tracks combined Python + local Ollama RSS (`peak_ram_mb`) to prove 8 GB institutional hardware feasibility. |
| **UI & CLI** | `streamlit` + `rich` | Minimal native Python UI (zero CSS/HTML/JS) and terminal tables displaying the 4 metrics, comparison table, and per-model hallucination reduction. |
| **Verification Harness** | `pytest` + `ruff` (`./verify.sh`) | Enforces all 5 domain laws (`LAWS.md`) and offline unit tests on every change. |

### Quick Defense Q&A

- **Why 7B–9B models instead of 3B–4B or 14B+?** Sub-7B models suffer severe reasoning degradation on dense academic prose (Wang et al., 2025), while 10B–14B+ models require 8–12 GB+ RAM leaving no headroom for retrieval and verification while still showing ~68–78% unguided hallucination rates (Bang et al., 2025).
- **Why flat vector RAG + NLI instead of GraphRAG or Agentic RAG?** Constructing and traversing LLM entity-relationship graphs for every uploaded student PDF multiplies token usage, memory, and latency beyond standard 8 GB school computers (Peng et al., 2025; Chen et al., 2026).
- **Why not paste the entire PDF into a 32K long-context window?** Compact 4K retrieval-augmented prompts match or beat 16K–32K long-context prompts with up to 4× faster summarization by avoiding quadratic self-attention and middle-context attention drift (Xu et al., 2023b; Chuang et al., 2024).
- **Why an external DeBERTa NLI verifier instead of LLM self-checking or ROUGE?** Lightweight 7B–9B models are poorly calibrated and overconfident when wrong (Kadavath et al., 2022; Laban et al., 2023), and word-overlap metrics (ROUGE/BLEU) miss factual contradictions that NLI catches (Honovich et al., 2022).
- **Why a single standardized prompt?** Holding the prompt constant across all documents eliminates prompt-complexity confounding (Mishra & Soy, 2026) and cleanly isolates RAG pipeline architecture as the sole independent variable (Chapter 3, pp. 58–59).

---

## Quickstart

### 1. Setup & Verify
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ui]"
./verify.sh
```

### 2. Launch Minimal Streamlit Dashboard
```bash
streamlit run src/vrag/ui.py
```
Runs directly without `PYTHONPATH=src`. Shows the **4 Core Metrics**, **Comparison Table (Base vs. Verified)**, **Hallucination Reduction by Model (7B–9B)**, **Side-by-Side Hallucination Proof**, and CSV/JSON export.

### 3. CLI Commands
```bash
# Offline deterministic 4-model demo
PYTHONPATH=src python3 -m vrag.cli --demo --output-dir outputs/demo

# Benchmark any uploaded academic PDF
PYTHONPATH=src python3 -m vrag.cli --pdf path/to/paper.pdf --output-dir outputs/run1

# Run Thesis Writing 1 stratified corpus (40% QASPER, 30% SciTLDR, 30% SciFact)
PYTHONPATH=src python3 -m vrag.cli --corpus data/benchmark_corpus --output-dir outputs/corpus

# Check local Ollama daemon & installed 7B-9B models
PYTHONPATH=src python3 -m vrag.cli --doctor
```

---

## Local Ollama Setup (`http://localhost:11434`)

To run live local 7B–9B inference (100% offline after pulling weights):
```bash
brew install --cask ollama
ollama serve
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull gemma2:9b-instruct-q4_K_M
ollama pull mistral:7b-instruct-v0.3-q4_K_M
ollama pull qwen2.5:7b-instruct-q4_K_M
PYTHONPATH=src python3 -m vrag.cli --doctor
```
If Ollama is offline, both the Streamlit UI and CLI automatically fall back to **Offline Simulation Mode** so comparative evaluation never crashes.

---

## Governance Documents

- [`LAWS.md`](LAWS.md) — 5 non-negotiable domain invariants enforced by `tests/laws/`.
- [`PRINCIPLES.md`](PRINCIPLES.md) — Engineering defaults and escalation boundaries.
- [`HARNESS.md`](HARNESS.md) — Verification pipeline (`./verify.sh`) and module architecture map.
- [`AGENTS.md`](AGENTS.md) — Agent operating workflow.
