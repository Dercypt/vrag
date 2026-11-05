"""Unit tests for the Streamlit UI view-model builders and 4-metric formatting."""

from __future__ import annotations

from vrag.evaluator import run_student_pdf_benchmark
from vrag.ui import (
    build_claim_audit_rows,
    build_four_metrics_summary_rows,
    build_trial_comparison_rows,
)


def test_ui_view_models_format_four_metrics_and_claim_audit() -> None:
    pdf_bytes = (
        b"Vaswani et al. (2017) proposed the Transformer architecture using self-attention.\f"
        b"Lewis et al. (2020) introduced Retrieval-Augmented Generation for factual grounding."
    )
    pairs, stats = run_student_pdf_benchmark(
        pdf_bytes=pdf_bytes,
        pdf_name="student_paper.pdf",
        models=("llama3.1:8b-instruct-q4_K_M",),
        simulation_mode=True,
    )

    metric_rows = build_four_metrics_summary_rows(stats)
    assert len(metric_rows) == 4
    assert [r["Metric"] for r in metric_rows] == [
        "Factual Precision (FActScore)",
        "Entailment Consistency (NLI Score)",
        "Citation Accuracy (Precision/Recall)",
        "Computational Latency",
    ]
    assert metric_rows[0]["Unit"] == "%"
    assert "%" in metric_rows[0]["Base RAG (Condition A)"]
    assert "%" in metric_rows[0]["Verification-Driven RAG (Condition B)"]
    assert metric_rows[3]["Unit"] == "s"
    assert "s" in metric_rows[3]["Base RAG (Condition A)"]

    trial_rows = build_trial_comparison_rows(pairs)
    assert len(trial_rows) == 4
    assert "Base FActScore (%)" in trial_rows[0]
    assert "Verified FActScore (%)" in trial_rows[0]
    assert "Verified Latency (s)" in trial_rows[0]

    _trial_a, trial_b = pairs[0]
    audit_rows = build_claim_audit_rows(trial_b)
    assert len(audit_rows) >= 2
    statuses = [r["Gate Decision"] for r in audit_rows]
    assert "SURVIVED (TP)" in statuses
    assert "PRUNED (TN)" in statuses
