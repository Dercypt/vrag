"""Unit tests for the Streamlit UI view-model builders and 4-metric formatting."""

from __future__ import annotations

import pytest

from vrag.evaluator import run_student_pdf_benchmark
from vrag.ui import (
    build_claim_audit_rows,
    build_four_metrics_summary_rows,
    build_trial_comparison_rows,
    execute_benchmark_safely,
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


def test_execute_benchmark_safely_success_and_error_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_bytes = b"Sample text for safe execution."

    # 1. Successful simulation execution
    pairs, stats, err = execute_benchmark_safely(
        pdf_bytes=pdf_bytes,
        pdf_name="sample.pdf",
        prompts=(
            ("p1", "What is the core methodology?"),
            ("p2", "What are the main findings?"),
        ),
        models=("llama3.1:8b-instruct-q4_K_M",),
        simulation_mode=True,
    )
    assert err is None
    assert pairs is not None and len(pairs) == 2
    assert stats is not None and len(stats) == 4

    # 2. Simulated failure (e.g. tokenizer/transformers mismatch or unreachable Ollama)
    def fail_benchmark(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ImportError(
            "tokenizers>=0.22.0,<=0.23.0 is required for a normal functioning of this module"
        )

    import vrag.ui

    monkeypatch.setattr(vrag.ui, "run_student_pdf_benchmark", fail_benchmark)
    failed_pairs, failed_stats, error_msg = execute_benchmark_safely(
        pdf_bytes=pdf_bytes,
        pdf_name="sample.pdf",
        prompts=(("p1", "prompt"),),
        models=("llama3.1:8b-instruct-q4_K_M",),
        simulation_mode=False,
    )
    assert failed_pairs is None
    assert failed_stats is None
    assert error_msg is not None
    assert "Benchmark execution encountered an error" in error_msg
    assert "Use Deterministic Simulation Mode (Fast Offline)" in error_msg
