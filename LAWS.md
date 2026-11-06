# LAWS.md — Non-Negotiable Domain Invariants

> **Project**: *A Comparative Study on Reducing Hallucinations in Lightweight Open-Source Large Language Models Using a Verification-Driven RAG Pipeline* (`vrag`)
>
> **Authority**: These laws are hard architectural constraints. No feature, refactor, or optimization may violate them. Every law is enforced by automated checks in `tests/laws/` and executed via `./verify.sh`.

---

## LAW 1 — Deterministic Verification Gating (Condition B)

In **Condition B (Proposed Verification-Driven RAG)**, no generated atomic claim/sentence may survive into the final output summary if **either** of the following conditions fails:

1. **NLI Entailment Gate**: The local cross-encoder (`cross-encoder/nli-deberta-v3-small`) entailment score `P(Entailment | retrieved_chunk, claim)` is strictly below the configured entailment threshold (`default = 0.70`) or the top predicted label is `Contradiction` or `Neutral`.
2. **Citation, Author & Year Existence Gate**: The claim lacks a valid `[Chunk_ID]` citation tag pointing to an actually retrieved passage, OR contains an extracted parenthetical or narrative citation (e.g., `(Author, Year)`, `Author (Year)`, `Author & Author (Year)`, or `Author et al. (Year)`) whose author surname or publication year does not appear (case-insensitively) within the cited source chunk(s).

**Telemetry Mandate**: Every atomic claim evaluated by the gate—whether retained (`TP`/`FP`) or pruned (`TN`/`FN`)—must be deterministically recorded in the trial's `ConfusionMatrix` (supporting independent ground-truth claim oracles on curated benchmarks and NLI fallback on ad-hoc uploads) and per-claim audit trail.

---

## LAW 2 — Strict Offline & Zero-External-API Execution

All pipeline operations—PDF ingestion (`pymupdf`), sentence segmentation (`pysbd`), semantic embedding (`all-MiniLM-L6-v2`), vector indexing (`faiss-cpu`), LLM generation (local Ollama daemon at `http://localhost:11434`), and NLI verification (`cross-encoder/nli-deberta-v3-small`)—must execute **100% locally**.

- Outbound network requests to paid or external LLM APIs (OpenAI, Anthropic, Gemini, hosted vector DBs, etc.) are strictly forbidden.
- Any HTTP client in `generator.py` must validate that the target host resolves exclusively to loopback (`localhost` / `127.0.0.1`).

---

## LAW 3 — Experimental Parity (Controlled Variable Isolation)

For any comparative evaluation trial $i \in \{1, \dots, N\}$, **Condition A (Baseline Standard RAG)** and **Condition B (Verification-Driven RAG)** must be evaluated under identical experimental conditions:

- Identical source PDF document (verified via SHA-256 digest).
- Identical user query / summarization prompt core.
- Identical local LLM tag (`llama3.1:8b-instruct-q4_K_M`, `mistral:7b-instruct-v0.3-q4_K_M`, or `qwen2.5:7b-instruct-q4_K_M`).
- Identical deterministic decoding hyperparameters (`temperature=0.0`, `seed=42`) and retrieval budget (`top_k`).

---

## LAW 4 — Sequential Single-Model Memory Safety (Hardware Constraint)

Target hardware is a **2019 MacBook Pro** with constrained unified/system RAM.

- At most **one** 7B–8B quantized Ollama LLM may reside in memory at any point in time.
- Multi-model benchmark sweeps must execute strictly sequentially (`llama3.1` $\rightarrow$ `mistral` $\rightarrow$ `qwen2.5`), never concurrently via threads/async workers.
- When transitioning between models, the harness must explicitly issue an unload request (`keep_alive: 0`) to the local Ollama daemon and record peak process RSS memory (`peak_ram_mb`).

---

## LAW 5 — Reproducible Telemetry & Statistical Completeness

Every comparative run in `evaluator.py` must capture and validate complete telemetry aligned with Chapter 3 of the thesis manuscript before exporting results:

1. **Non-Negative Stage Timings**: $t_{\text{chunk}} \ge 0$, $t_{\text{retrieve}} \ge 0$, $t_{\text{generate}} \ge 0$, and $t_{\text{verify}} \ge 0$ (with $t_{\text{verify}} = 0.0$ in Condition A, and $\text{Latency}_{\text{total}} = t_{\text{chunk}} + t_{\text{retrieve}} + t_{\text{generate}} + t_{\text{verify}}$).
2. **Core Quality Metrics & Hallucination Taxonomy**:
   - Atomic FActScore $= \frac{1}{|A_y|}\sum_{a \in A_y}\mathbb{I}[a \text{ is supported by } C] \in [0.0, 1.0]$.
   - NLI Entailment Consistency $= \frac{TP + TN}{TP + TN + FP + FN} \in [0.0, 1.0]$.
   - Citation Precision $= \frac{\text{Citations Entailed by Cited Passage}}{\text{Total Generated Citations}} \in [0.0, 1.0]$ and Citation Recall $= \frac{\text{Claims with Entailed Citations}}{\text{Total Claims Requiring Citation}} \in [0.0, 1.0]$.
   - Separate tracking of **Intrinsic** (contradiction) and **Extrinsic** (unverifiable/fabricated citation) hallucination counts and rates.
3. **Descriptive & Inferential Statistical Significance**: Paired comparisons across $N \ge 2$ trials must compute condition means, standard deviations ($s_d$), percentage error reduction, Shapiro-Wilk normality check ($N \ge 3$), paired Student's $t$-test (`scipy.stats.ttest_rel`), and Wilcoxon signed-rank test (`scipy.stats.wilcoxon` fallback) $p$-values at $\alpha = 0.05$.
