# AGENTS.md — Governance Harness & Agent Operating Manual

Welcome to **`vrag`** — the comparative benchmarking harness for **"A Comparative Study on Reducing Hallucinations in Lightweight Open-Source Large Language Models Using a Verification-Driven RAG Pipeline"**.

## 1. Read Order Before Writing Code

Before modifying any file in this repository, read and adhere to:

1. [`LAWS.md`](file:///Users/zyndrex/vrag-sandbox/LAWS.md) — **Non-Negotiable Domain Invariants** (Verification gating, offline execution, experimental parity, sequential single-model RAM safety, statistical telemetry completeness).
2. [`PRINCIPLES.md`](file:///Users/zyndrex/vrag-sandbox/PRINCIPLES.md) — **Escalation Triggers & Engineering Defaults** (When to stop and ask vs. how to structure pure, testable Python code).
3. [`HARNESS.md`](file:///Users/zyndrex/vrag-sandbox/HARNESS.md) — **Verification Pipeline & Architecture Map** (How `./verify.sh` validates your work).

---

## 2. Mandatory Workflow for Every Change

1. **Check Escalation Triggers**: Verify your planned change does not violate `LAWS.md` or trigger an unapproved escalation in `PRINCIPLES.md`.
2. **Write / Update Tests First**:
   - Domain invariant behavior belongs in `tests/laws/` (marked with `@pytest.mark.law`).
   - Module behavior belongs in `tests/unit/`.
3. **Implement Minimal, Deterministic Code**:
   - Keep `src/vrag/` modules modular and dependency-injected so tests can run offline without loading 8B model weights unless running an integration benchmark.
4. **Run `./verify.sh`**:
   - Never declare a task finished until `./verify.sh` exits with `0`.
