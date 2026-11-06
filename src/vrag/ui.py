"""Lightweight Streamlit UI for Student PDF Benchmarking across the 4 Core Thesis Metrics."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict

from vrag.cli import format_metric_value
from vrag.evaluator import (
    PairedStatisticalResult,
    run_student_pdf_benchmark,
)
from vrag.generator import check_ollama_status
from vrag.models import (
    APPROVED_MODELS,
    DEFAULT_STUDENT_BENCHMARK_PROMPTS,
    TrialTelemetry,
)

METRIC_DESCRIPTIONS: dict[str, str] = {
    "Factual Precision (FActScore)": (
        "Percentage of generated claims actually supported by the uploaded PDF (%)."
    ),
    "Entailment Consistency (NLI Score)": (
        "Accuracy of classifying sentences as supported vs. hallucinated (TP vs. FP) (%)."
    ),
    "Citation Accuracy (Precision/Recall)": (
        "Whether real authors and chunk/page numbers are cited without inventing fake ones (%)."
    ),
    "Computational Latency": (
        "End-to-end execution time (in seconds) on local 2019 MacBook Pro hardware (s)."
    ),
}


def build_four_metrics_summary_rows(
    stats: list[PairedStatisticalResult],
) -> list[dict[str, str]]:
    """Format the 4 core comparative metrics into display rows with % and s units."""
    rows: list[dict[str, str]] = []
    for res in stats:
        rows.append(
            {
                "Metric": res.metric_name,
                "Description": METRIC_DESCRIPTIONS.get(res.metric_name, ""),
                "Unit": res.unit,
                "Base RAG (Condition A)": format_metric_value(res.mean_baseline_a, res.unit),
                "Verification-Driven RAG (Condition B)": format_metric_value(
                    res.mean_verified_b, res.unit
                ),
                "Delta (B - A)": format_metric_value(res.mean_delta, res.unit, signed=True),
                "Paired t-stat": f"{res.t_statistic:.4f}",
                "p-value": f"{res.t_pvalue:.4e}",
            }
        )
    return rows


def build_trial_comparison_rows(
    pairs: list[tuple[TrialTelemetry, TrialTelemetry]],
) -> list[dict[str, object]]:
    """Build per-trial side-by-side comparison rows for the 4 core metrics."""
    rows: list[dict[str, object]] = []
    for trial_a, trial_b in pairs:
        rows.append(
            {
                "Trial ID": trial_a.trial_id,
                "Model": trial_a.model_name,
                "Prompt": trial_a.prompt,
                "Base FActScore (%)": f"{trial_a.factual_precision_pct:.1f}%",
                "Verified FActScore (%)": f"{trial_b.factual_precision_pct:.1f}%",
                "Base NLI Consistency (%)": f"{trial_a.entailment_consistency_pct:.1f}%",
                "Verified NLI Consistency (%)": f"{trial_b.entailment_consistency_pct:.1f}%",
                "Base Citation Acc (%)": f"{trial_a.citation_accuracy_pct:.1f}%",
                "Verified Citation Acc (%)": f"{trial_b.citation_accuracy_pct:.1f}%",
                "Base Latency (s)": f"{trial_a.computational_latency_sec:.4f}s",
                "Verified Latency (s)": f"{trial_b.computational_latency_sec:.4f}s",
                "Verified TP/TN/FP/FN": (
                    f"{trial_b.confusion_matrix.tp}/{trial_b.confusion_matrix.tn}/"
                    f"{trial_b.confusion_matrix.fp}/{trial_b.confusion_matrix.fn}"
                ),
                "Peak RAM (MB)": f"{trial_b.peak_ram_mb:.1f}",
            }
        )
    return rows


def build_claim_audit_rows(trial_b: TrialTelemetry) -> list[dict[str, object]]:
    """Build sentence-by-sentence verification gate audit rows for Condition B."""
    rows: list[dict[str, object]] = []
    for claim in trial_b.claim_audit:
        rows.append(
            {
                "Atomic Sentence": claim.sentence,
                "Cited Chunks": ", ".join(claim.cited_chunk_ids) or "None",
                "Extracted Authors": ", ".join(claim.extracted_authors) or "None",
                "NLI Label": claim.nli_label.value,
                "NLI Score (%)": f"{claim.entailment_score * 100.0:.1f}%",
                "Gate Decision": "SURVIVED (TP)" if claim.survived else "PRUNED (TN)",
                "Prune Reason": claim.prune_reason or "Verified & Retained",
            }
        )
    return rows


def _trials_to_csv_bytes(trials: list[TrialTelemetry]) -> bytes:
    buf = io.StringIO()
    fieldnames = [
        "trial_id",
        "model_name",
        "mode",
        "factual_precision_pct",
        "entailment_consistency_pct",
        "citation_accuracy_pct",
        "computational_latency_sec",
        "tp",
        "tn",
        "fp",
        "fn",
        "peak_ram_mb",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for t in trials:
        writer.writerow(
            {
                "trial_id": t.trial_id,
                "model_name": t.model_name,
                "mode": t.mode.value,
                "factual_precision_pct": f"{t.factual_precision_pct:.2f}",
                "entailment_consistency_pct": f"{t.entailment_consistency_pct:.2f}",
                "citation_accuracy_pct": f"{t.citation_accuracy_pct:.2f}",
                "computational_latency_sec": f"{t.computational_latency_sec:.6f}",
                "tp": t.confusion_matrix.tp,
                "tn": t.confusion_matrix.tn,
                "fp": t.confusion_matrix.fp,
                "fn": t.confusion_matrix.fn,
                "peak_ram_mb": f"{t.peak_ram_mb:.2f}",
            }
        )
    return buf.getvalue().encode("utf-8")


def is_connection_refused_error(err: Exception) -> bool:
    """Return True if exception indicates unreachable local Ollama daemon or connection refusal."""
    import errno

    if isinstance(err, (ConnectionRefusedError, ConnectionResetError)):
        return True
    if isinstance(err, OSError) and err.errno in (errno.ECONNREFUSED, errno.ECONNRESET):
        return True

    err_type_name = type(err).__name__
    if "ConnectError" in err_type_name or "ConnectTimeout" in err_type_name:
        return True

    cause = getattr(err, "__cause__", None) or getattr(err, "__context__", None)
    if cause is not None and isinstance(cause, Exception) and is_connection_refused_error(cause):
        return True

    msg = str(err).lower()
    return (
        "connection refused" in msg
        or "errno 61" in msg
        or "errno 111" in msg
        or "failed to establish a new connection" in msg
        or "all connection attempts failed" in msg
    )


def execute_benchmark_safely(
    pdf_bytes: bytes,
    pdf_name: str,
    prompts: tuple[tuple[str, str], ...],
    models: tuple[str, ...],
    simulation_mode: bool,
) -> tuple[
    list[tuple[TrialTelemetry, TrialTelemetry]] | None,
    list[PairedStatisticalResult] | None,
    str | None,
]:
    """Execute student PDF benchmark with robust exception catching and diagnostic message."""
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
        return (
            None,
            None,
            (
                f"Benchmark execution encountered an error: {err}\n\n"
                "Tip: If Ollama is unreachable or model weights encounter environment mismatches, "
                "check 'Use Deterministic Simulation Mode (Fast Offline)' in the sidebar."
            ),
        )


def render_streamlit_app() -> None:  # pragma: no cover
    """Render the minimal Streamlit comparative benchmark UI."""
    import streamlit as st

    st.set_page_config(
        page_title="vrag — Verification-Driven RAG Benchmark",
        layout="wide",
    )
    st.title("Verification-Driven RAG vs. Base RAG — Comparative Benchmark")
    st.caption(
        "Thesis Harness: Evaluating Factual Precision (FActScore %), Entailment Consistency "
        "(NLI Score %), Citation Accuracy (%), and Computational Latency (s)."
    )

    with st.sidebar:
        st.header("1. Student PDF Input")
        uploaded_file = st.file_uploader(
            "Upload Student / Academic PDF",
            type=["pdf", "txt"],
            help="Upload a student paper PDF to run the 4-metric comparative benchmark.",
        )

        st.header("2. Benchmark Configuration")
        selected_models = st.multiselect(
            "Local Ollama Models (Sequential Execution)",
            options=list(APPROVED_MODELS),
            default=[APPROVED_MODELS[0]],
        )
        ollama_online = check_ollama_status()
        sim_mode = st.checkbox(
            "Use Deterministic Simulation Mode (Fast Offline)",
            value=not ollama_online,
            help="When enabled, simulates LLM responses grounded on your uploaded PDF chunks.",
        )
        custom_query = st.text_input(
            "Optional Custom Prompt (leave blank to run 4-prompt benchmark suite)",
            value="",
        )
        run_btn = st.button("Run Comparative Benchmark", type="primary")

    if not run_btn:
        st.info(
            "Upload a student PDF in the sidebar (or use the built-in sample paper) and click "
            "**Run Comparative Benchmark** to compare Base RAG vs. Verification-Driven RAG."
        )
        return

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        pdf_name = uploaded_file.name
    else:
        pdf_name = "sample_student_thesis.pdf"
        pdf_bytes = (
            b"Vaswani et al. (2017) introduced the Transformer architecture based on "
            b"self-attention mechanisms.\f"
            b"Lewis et al. (2020) proposed Retrieval-Augmented Generation to ground LLMs on "
            b"retrieved passages and reduce factual hallucinations."
        )

    models_tuple = tuple(selected_models) if selected_models else (APPROVED_MODELS[0],)
    if custom_query.strip():
        prompts = (
            ("custom_1", custom_query.strip()),
            ("custom_2", f"{custom_query.strip()} (Provide detailed methodology citations.)"),
        )
    else:
        prompts = DEFAULT_STUDENT_BENCHMARK_PROMPTS

    with st.spinner("Running sequential Condition A vs. Condition B benchmark..."):
        pairs, stats, error_msg = execute_benchmark_safely(
            pdf_bytes=pdf_bytes,
            pdf_name=pdf_name,
            prompts=prompts,
            models=models_tuple,
            simulation_mode=sim_mode,
        )

    if error_msg or pairs is None or stats is None:
        st.error(error_msg or "Benchmark execution failed.")
        return

    st.subheader("1. Core 4-Metric Comparative Results (Base RAG vs. Verification-Driven RAG)")
    cols = st.columns(4)
    for col, res in zip(cols, stats):
        col.metric(
            label=res.metric_name,
            value=format_metric_value(res.mean_verified_b, res.unit),
            delta=format_metric_value(res.mean_delta, res.unit, signed=True),
        )

    st.table(build_four_metrics_summary_rows(stats))

    st.subheader("2. Side-by-Side Output & Verification Gate Audit (First Trial)")
    first_a, first_b = pairs[0]
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Condition A: Base RAG (Unverified Output)")
        st.warning(first_a.output_summary or "(Empty output)")
        st.caption(
            f"FActScore: {first_a.factual_precision_pct:.1f}% | "
            f"NLI Consistency: {first_a.entailment_consistency_pct:.1f}% | "
            f"Citation Accuracy: {first_a.citation_accuracy_pct:.1f}% | "
            f"Latency: {first_a.computational_latency_sec:.4f}s"
        )
    with col_b:
        st.markdown("#### Condition B: Verification-Driven RAG (Pruned Output)")
        st.success(first_b.output_summary or "(All sentences pruned by verification gate)")
        st.caption(
            f"FActScore: {first_b.factual_precision_pct:.1f}% | "
            f"NLI Consistency: {first_b.entailment_consistency_pct:.1f}% | "
            f"Citation Accuracy: {first_b.citation_accuracy_pct:.1f}% | "
            f"Latency: {first_b.computational_latency_sec:.4f}s"
        )

    st.markdown("##### Atomic Claim Verification & Pruning Log (Condition B)")
    st.dataframe(build_claim_audit_rows(first_b), use_container_width=True)

    st.subheader("3. Per-Trial Breakdown & Export")
    st.dataframe(build_trial_comparison_rows(pairs), use_container_width=True)

    all_trials = [t for pair in pairs for t in pair]
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "Download Telemetry CSV",
            data=_trials_to_csv_bytes(all_trials),
            file_name="vrag_telemetry.csv",
            mime="text/csv",
        )
    with dl_col2:
        json_payload = json.dumps(
            {
                "trials": [asdict(t) for t in all_trials],
                "statistical_tests": [asdict(s) for s in stats],
            },
            indent=2,
        ).encode("utf-8")
        st.download_button(
            "Download Statistical Summary JSON",
            data=json_payload,
            file_name="vrag_summary.json",
            mime="application/json",
        )


if __name__ == "__main__":  # pragma: no cover
    render_streamlit_app()
