"""Minimal, pure-Python Streamlit UI for 7B-9B Verification-Driven RAG Benchmarking."""

from __future__ import annotations

import csv
import errno
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Allow direct execution (`streamlit run src/vrag/ui.py`) without PYTHONPATH=src.
_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from vrag.cli import format_metric_value  # noqa: E402
from vrag.evaluator import PairedStatisticalResult, run_student_pdf_benchmark  # noqa: E402
from vrag.generator import diagnose_ollama  # noqa: E402
from vrag.models import (  # noqa: E402
    APPROVED_MODELS,
    DEFAULT_STUDENT_BENCHMARK_PROMPTS,
    STANDARDIZED_SUMMARIZATION_PROMPT,
    ClaimVerificationResult,
    NLILabel,
    TrialTelemetry,
)

# --- Constants ---

_FRIENDLY_MODEL_NAMES: dict[str, str] = {
    "llama3.1:8b-instruct-q4_K_M": "Llama-3.1-8B",
    "gemma2:9b-instruct-q4_K_M": "Gemma-2-9B",
    "mistral:7b-instruct-v0.3-q4_K_M": "Mistral-7B",
    "qwen2.5:7b-instruct-q4_K_M": "Qwen-2.5-7B",
}

_SAMPLE_PDF_BYTES: bytes = (
    b"Bang et al. (2025) evaluated 7B to 9B open-source models including Llama-3.1-8B, "
    b"Gemma-2-9B, Mistral-7B, and Qwen-2.5-7B on HalluLens and found baseline hallucination "
    b"rates between 48% and 85%. Kadavath et al. (2022) showed smaller models act overconfident "
    b"when wrong.\f"
    b"Wang et al. (2025) proposed dynamic semantic chunking using cosine similarity thresholds. "
    b"Qian et al. (2025) introduced VeriCite to prune unentailed claims via sentence-level NLI."
)

_CSV_COLUMNS = [
    "trial_id",
    "model_name",
    "mode",
    "factual_precision_pct",
    "entailment_consistency_pct",
    "citation_accuracy_pct",
    "citation_precision_pct",
    "citation_recall_pct",
    "hallucination_error_rate_pct",
    "intrinsic_hallucination_rate_pct",
    "extrinsic_hallucination_rate_pct",
    "computational_latency_sec",
    "tp",
    "tn",
    "fp",
    "fn",
    "peak_ram_mb",
]

_CONN_ERROR_TOKENS = (
    "connection refused",
    "errno 61",
    "errno 111",
    "failed to establish a new connection",
    "all connection attempts failed",
)


# --- Pure-Python View-Model Builders ---


def format_model_label(model_name: str) -> str:
    """Return a concise model display name for tables and selectors."""
    return _FRIENDLY_MODEL_NAMES.get(model_name, model_name.split(":")[0])


def resolve_benchmark_prompts(
    custom_query: str,
    model_count: int,
) -> tuple[tuple[str, str], ...]:
    """Use the single standardized prompt (Ch. 3 pp. 58-59) when model_count >= 2."""
    cleaned = custom_query.strip()
    if model_count >= 2:
        return (
            (("custom", cleaned),)
            if cleaned
            else (("standardized", STANDARDIZED_SUMMARIZATION_PROMPT),)
        )
    # Single-model runs need >= 2 paired observations for paired t-test (LAW 5).
    if cleaned:
        return (
            ("custom_1", cleaned),
            ("custom_2", f"{cleaned} Include exact inline citations."),
        )
    return DEFAULT_STUDENT_BENCHMARK_PROMPTS[:2]


def build_four_metrics_summary_rows(stats: list[PairedStatisticalResult]) -> list[dict[str, str]]:
    """Format the 4 core comparative metrics (Condition A vs Condition B)."""
    return [
        {
            "Metric": res.metric_name,
            "Unit": res.unit,
            "Base RAG (Condition A)": format_metric_value(res.mean_baseline_a, res.unit),
            "Verification-Driven RAG (Condition B)": format_metric_value(
                res.mean_verified_b, res.unit
            ),
            "Delta (B - A)": format_metric_value(res.mean_delta, res.unit, signed=True),
            "SD (s_d)": f"{res.std_delta:.2f}",
            "Error Reduction": f"{res.pct_error_reduction:+.1f}%",
            "Paired t-stat": f"{res.t_statistic:.4f}",
            "p-value": f"{res.t_pvalue:.4e}",
            "Wilcoxon p": f"{res.wilcoxon_pvalue:.4e}"
            if res.wilcoxon_pvalue is not None
            else "N/A",
        }
        for res in stats
    ]


def build_model_hallucination_rows(
    pairs: list[tuple[TrialTelemetry, TrialTelemetry]],
) -> list[dict[str, object]]:
    """Aggregate hallucination reduction per 7B-9B model (Condition A -> Condition B)."""
    grouped: dict[str, list[tuple[TrialTelemetry, TrialTelemetry]]] = {}
    for pair in pairs:
        grouped.setdefault(pair[0].model_name, []).append(pair)

    rows: list[dict[str, object]] = []
    for model_name, model_pairs in grouped.items():
        n = len(model_pairs)
        rows.append(
            {
                "Model": format_model_label(model_name),
                "Base Hallucination Rate": (
                    f"{sum(a.hallucination_error_rate_pct for a, _ in model_pairs) / n:.1f}%"
                ),
                "Verified Hallucination Rate": (
                    f"{sum(b.hallucination_error_rate_pct for _, b in model_pairs) / n:.1f}%"
                ),
                "Intrinsic (Contradictions)": (
                    f"{sum(a.intrinsic_hallucinations for a, _ in model_pairs)} → "
                    f"{sum(b.intrinsic_hallucinations for _, b in model_pairs)}"
                ),
                "Extrinsic (Fabricated)": (
                    f"{sum(a.extrinsic_hallucinations for a, _ in model_pairs)} → "
                    f"{sum(b.extrinsic_hallucinations for _, b in model_pairs)}"
                ),
                "Verified FActScore": (
                    f"{sum(b.factual_precision_pct for _, b in model_pairs) / n:.1f}%"
                ),
                "Avg Latency": (
                    f"{sum(b.computational_latency_sec for _, b in model_pairs) / n:.4f}s"
                ),
            }
        )
    return rows


def build_trial_comparison_rows(
    pairs: list[tuple[TrialTelemetry, TrialTelemetry]],
) -> list[dict[str, object]]:
    """Build per-trial side-by-side comparison rows for the 4 core metrics."""
    return [
        {
            "Trial ID": a.trial_id,
            "Model": format_model_label(a.model_name),
            "Base FActScore (%)": f"{a.factual_precision_pct:.1f}%",
            "Verified FActScore (%)": f"{b.factual_precision_pct:.1f}%",
            "Base NLI Consistency (%)": f"{a.entailment_consistency_pct:.1f}%",
            "Verified NLI Consistency (%)": f"{b.entailment_consistency_pct:.1f}%",
            "Base Citation Acc (%)": f"{a.citation_accuracy_pct:.1f}%",
            "Verified Citation Acc (%)": f"{b.citation_accuracy_pct:.1f}%",
            "Base Latency (s)": f"{a.computational_latency_sec:.4f}s",
            "Verified Latency (s)": f"{b.computational_latency_sec:.4f}s",
            "Verified TP/TN/FP/FN": (
                f"{b.confusion_matrix.tp}/{b.confusion_matrix.tn}/"
                f"{b.confusion_matrix.fp}/{b.confusion_matrix.fn}"
            ),
            "Peak RAM (MB)": f"{b.peak_ram_mb:.1f}",
        }
        for a, b in pairs
    ]


def build_claim_audit_rows(trial_b: TrialTelemetry) -> list[dict[str, object]]:
    """Build sentence-by-sentence verification gate audit rows for Condition B."""
    return [
        {
            "Atomic Sentence": c.sentence,
            "Cited Chunks": ", ".join(c.cited_chunk_ids) or "None",
            "Authors": ", ".join(c.extracted_authors) or "None",
            "Years": ", ".join(c.extracted_years) or "None",
            "NLI Label": c.nli_label.value,
            "NLI Score (%)": f"{c.entailment_score * 100.0:.1f}%",
            "Gate Decision": "SURVIVED (TP)" if c.survived else "PRUNED (TN)",
            "Prune Reason": c.prune_reason or "Verified & Retained",
        }
        for c in trial_b.claim_audit
    ]


def _classify_hallucination(claim: ClaimVerificationResult) -> tuple[str, str] | None:
    """Return (type_label, detail) for an unverified Condition A claim, or None if valid."""
    if (
        claim.citation_exists
        and claim.author_exists
        and claim.year_exists
        and claim.nli_label == NLILabel.ENTAILMENT
        and claim.entailment_score >= 0.70
    ):
        return None
    if claim.nli_label == NLILabel.CONTRADICTION:
        return (
            "Intrinsic (Contradicts PDF)",
            f"NLI contradiction ({claim.entailment_score * 100:.0f}% entailment)",
        )
    if not claim.author_exists or not claim.year_exists:
        return (
            "Extrinsic (Fabricated Citation)",
            "Cited author or year not found in retrieved PDF passage",
        )
    if not claim.citation_exists:
        return (
            "Extrinsic (Missing/Invalid Chunk Tag)",
            f"Cited {', '.join(claim.cited_chunk_ids) or 'no chunk'} (not retrieved)",
        )
    return (
        "Extrinsic (Unsupported Claim)",
        f"NLI {claim.nli_label.value} ({claim.entailment_score * 100:.0f}% entailment)",
    )


def build_base_hallucination_evidence(trial_a: TrialTelemetry) -> list[dict[str, str]]:
    """Extract concrete hallucinated claims from Condition A for transparent proof."""
    evidence: list[dict[str, str]] = []
    for claim in trial_a.claim_audit:
        if classified := _classify_hallucination(claim):
            h_type, detail = classified
            evidence.append({"Type": h_type, "Flagged Claim": claim.sentence, "Detail": detail})
    return evidence


def _trials_to_csv_bytes(trials: list[TrialTelemetry]) -> bytes:
    """Serialize trial telemetry rows to UTF-8 CSV bytes."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(
        {
            "trial_id": t.trial_id,
            "model_name": t.model_name,
            "mode": t.mode.value,
            "factual_precision_pct": f"{t.factual_precision_pct:.2f}",
            "entailment_consistency_pct": f"{t.entailment_consistency_pct:.2f}",
            "citation_accuracy_pct": f"{t.citation_accuracy_pct:.2f}",
            "citation_precision_pct": f"{t.citation_precision_pct:.2f}",
            "citation_recall_pct": f"{t.citation_recall_pct:.2f}",
            "hallucination_error_rate_pct": f"{t.hallucination_error_rate_pct:.2f}",
            "intrinsic_hallucination_rate_pct": f"{t.intrinsic_hallucination_rate_pct:.2f}",
            "extrinsic_hallucination_rate_pct": f"{t.extrinsic_hallucination_rate_pct:.2f}",
            "computational_latency_sec": f"{t.computational_latency_sec:.6f}",
            "tp": t.confusion_matrix.tp,
            "tn": t.confusion_matrix.tn,
            "fp": t.confusion_matrix.fp,
            "fn": t.confusion_matrix.fn,
            "peak_ram_mb": f"{t.peak_ram_mb:.2f}",
        }
        for t in trials
    )
    return buf.getvalue().encode("utf-8")


# --- Resilient Execution ---


def is_connection_refused_error(err: Exception) -> bool:
    """Return True if exception indicates unreachable local Ollama daemon."""
    if isinstance(err, (ConnectionRefusedError, ConnectionResetError)):
        return True
    if isinstance(err, OSError) and err.errno in (errno.ECONNREFUSED, errno.ECONNRESET):
        return True
    if "ConnectError" in type(err).__name__ or "ConnectTimeout" in type(err).__name__:
        return True
    cause = getattr(err, "__cause__", None) or getattr(err, "__context__", None)
    if isinstance(cause, Exception) and is_connection_refused_error(cause):
        return True
    return any(tok in str(err).lower() for tok in _CONN_ERROR_TOKENS)


def execute_benchmark_safely(
    pdf_bytes: bytes,
    pdf_name: str,
    prompts: tuple[tuple[str, str], ...],
    models: tuple[str, ...],
    simulation_mode: bool,
    auto_fallback: bool = True,
) -> tuple[
    list[tuple[TrialTelemetry, TrialTelemetry]] | None,
    list[PairedStatisticalResult] | None,
    str | None,
]:
    """Run benchmark and fall back to Offline Simulation Mode if local Ollama is unreachable."""
    try:
        pairs, stats = run_student_pdf_benchmark(
            pdf_bytes=pdf_bytes,
            pdf_name=pdf_name,
            prompts=prompts,
            models=models,
            simulation_mode=simulation_mode,
        )
        return pairs, stats, None
    except Exception as err:
        if not simulation_mode and auto_fallback and is_connection_refused_error(err):
            try:
                fb_pairs, fb_stats = run_student_pdf_benchmark(
                    pdf_bytes=pdf_bytes,
                    pdf_name=pdf_name,
                    prompts=prompts,
                    models=models,
                    simulation_mode=True,
                )
                return (
                    fb_pairs,
                    fb_stats,
                    f"Ollama unreachable ({err}). Completed in Offline Simulation Mode.",
                )
            except Exception as fallback_err:
                err = fallback_err
        return None, None, f"Benchmark error: {err}"


# --- Minimal Pure-Python Streamlit App ---


def render_streamlit_app() -> None:  # pragma: no cover
    """Render the minimal Streamlit comparative benchmark dashboard."""
    import streamlit as st

    st.set_page_config(page_title="vrag · Hallucination Benchmark", layout="wide")

    with st.sidebar:
        st.title("vrag")
        st.caption("7B–9B Verification-Driven RAG Benchmark")
        uploaded = st.file_uploader("Upload academic PDF", type=["pdf", "txt"])
        selected_models = st.multiselect(
            "7B–9B Models (run sequentially)",
            options=list(APPROVED_MODELS),
            default=list(APPROVED_MODELS),
            format_func=format_model_label,
        )
        health = diagnose_ollama()
        if health.is_online and not health.missing_approved_models:
            st.success(f"Ollama online · {health.base_url}")
        else:
            st.warning(f"Ollama offline or missing models · {health.base_url}")
            with st.expander("Setup commands"):
                for step in health.troubleshooting_steps:
                    st.code(step, language="bash")

        sim_mode = st.checkbox(
            "Offline simulation mode",
            value=not (health.is_online and not health.missing_approved_models),
        )
        prompt_text = st.text_area(
            "Standardized prompt",
            value=STANDARDIZED_SUMMARIZATION_PROMPT,
            height=90,
        )
        run_clicked = st.button("Run benchmark", type="primary", use_container_width=True)

    st.title("Verification-Driven RAG Benchmark")
    st.caption(
        "Condition A (Standard Fixed-Window RAG) vs. Condition B (Semantic Chunking + NLI Gate)"
    )

    if run_clicked:
        pdf_bytes = uploaded.getvalue() if uploaded is not None else _SAMPLE_PDF_BYTES
        pdf_name = uploaded.name if uploaded is not None else "sample_academic_paper.pdf"
        models = tuple(selected_models) if selected_models else tuple(APPROVED_MODELS)
        prompts = resolve_benchmark_prompts(prompt_text, len(models))
        with st.spinner("Running sequential Condition A vs. Condition B evaluation..."):
            pairs, stats, notice = execute_benchmark_safely(
                pdf_bytes=pdf_bytes,
                pdf_name=pdf_name,
                prompts=prompts,
                models=models,
                simulation_mode=sim_mode,
            )
        st.session_state["vrag_results"] = (pairs, stats, pdf_name, notice)

    if "vrag_results" not in st.session_state:
        st.info(
            "Upload any academic PDF in the sidebar (or keep the built-in sample) and click "
            "**Run benchmark** to compare Base RAG against Verification-Driven RAG."
        )
        return

    pairs, stats, pdf_name, notice = st.session_state["vrag_results"]
    if pairs is None or stats is None:
        st.error(notice or "Benchmark execution failed.")
        return
    if notice:
        st.warning(notice)

    st.subheader(f"4 Core Metrics — {pdf_name} ({len(pairs)} paired trials)")
    cols = st.columns(4)
    for col, res in zip(cols, stats):
        is_latency = res.unit == "s"
        col.metric(
            label=res.metric_name,
            value=format_metric_value(res.mean_verified_b, res.unit),
            delta=f"{format_metric_value(res.mean_delta, res.unit, signed=True)} vs Base",
            delta_color="inverse" if is_latency else "normal",
        )

    st.subheader("Comparison Table: Base RAG vs. Verification-Driven RAG")
    st.dataframe(build_four_metrics_summary_rows(stats), use_container_width=True, hide_index=True)

    st.subheader("Hallucination Reduction by Model (7B–9B)")
    st.dataframe(build_model_hallucination_rows(pairs), use_container_width=True, hide_index=True)

    st.subheader("Inspect Hallucination Proof")
    trial_labels = [f"{format_model_label(a.model_name)} · {a.trial_id}" for a, _ in pairs]
    idx = st.selectbox(
        "Select trial", options=range(len(pairs)), format_func=trial_labels.__getitem__
    )
    sel_a, sel_b = pairs[idx]
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Condition A — Base RAG (Unverified)**")
        st.write(sel_a.output_summary or "No output.")
        st.caption(
            f"FActScore {sel_a.factual_precision_pct:.1f}% · "
            f"Hallucinations {sel_a.hallucination_error_rate_pct:.1f}% · "
            f"Citations {sel_a.citation_accuracy_pct:.1f}% · "
            f"Latency {sel_a.computational_latency_sec:.4f}s"
        )
    with col_b:
        st.markdown("**Condition B — Verification-Driven RAG (Pruned)**")
        st.write(sel_b.output_summary or "All unsupported claims pruned.")
        st.caption(
            f"FActScore {sel_b.factual_precision_pct:.1f}% · "
            f"Hallucinations {sel_b.hallucination_error_rate_pct:.1f}% · "
            f"Citations {sel_b.citation_accuracy_pct:.1f}% · "
            f"Latency {sel_b.computational_latency_sec:.4f}s"
        )

    evidence = build_base_hallucination_evidence(sel_a)
    with st.expander(f"Flagged Hallucinations in Base RAG ({len(evidence)})", expanded=True):
        if evidence:
            st.dataframe(evidence, use_container_width=True, hide_index=True)
        else:
            st.caption("No hallucinations detected in Condition A for this trial.")

    with st.expander("Sentence-by-Sentence Verification Log (Condition B)", expanded=False):
        st.dataframe(build_claim_audit_rows(sel_b), use_container_width=True, hide_index=True)

    with st.expander("Per-Trial Execution Telemetry", expanded=False):
        st.dataframe(build_trial_comparison_rows(pairs), use_container_width=True, hide_index=True)

    all_trials = [t for pair in pairs for t in pair]
    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "Download telemetry CSV",
        data=_trials_to_csv_bytes(all_trials),
        file_name="vrag_telemetry.csv",
        mime="text/csv",
    )
    dl2.download_button(
        "Download summary JSON",
        data=json.dumps(
            {
                "trials": [asdict(t) for t in all_trials],
                "statistical_tests": [asdict(s) for s in stats],
            },
            indent=2,
        ).encode("utf-8"),
        file_name="vrag_summary.json",
        mime="application/json",
    )


if __name__ == "__main__":  # pragma: no cover
    render_streamlit_app()
