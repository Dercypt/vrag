"""Unit tests for the Streamlit UI view-model builders and 4-metric formatting."""

from __future__ import annotations

import pytest

from vrag.evaluator import run_student_pdf_benchmark
from vrag.ui import (
    build_claim_audit_rows,
    build_four_metrics_summary_rows,
    build_trial_comparison_rows,
    execute_benchmark_safely,
    is_connection_refused_error,
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


def test_is_connection_refused_error() -> None:
    import errno

    assert is_connection_refused_error(ConnectionRefusedError("Connection refused"))
    assert is_connection_refused_error(ConnectionResetError("Connection reset by peer"))
    assert is_connection_refused_error(OSError(errno.ECONNREFUSED, "Connection refused"))
    assert is_connection_refused_error(RuntimeError("[Errno 61] Connection refused"))
    assert is_connection_refused_error(RuntimeError("Failed to establish a new connection"))

    class FakeConnectError(Exception):
        pass

    assert is_connection_refused_error(FakeConnectError("Cannot reach localhost:11434"))

    # Nested exceptions
    try:
        try:
            raise ConnectionRefusedError("nested connection refused")
        except ConnectionRefusedError as orig:
            raise RuntimeError("Wrapper error") from orig
    except RuntimeError as wrapped:
        assert is_connection_refused_error(wrapped)

    # Negative cases
    assert not is_connection_refused_error(ValueError("Invalid token size"))
    assert not is_connection_refused_error(ImportError("Missing module"))
    assert not is_connection_refused_error(KeyError("missing_key"))


def test_execute_benchmark_safely_falls_back_on_connection_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrag.ui

    pdf_bytes = b"Sample text for connection refused fallback test."

    original_run = vrag.ui.run_student_pdf_benchmark
    attempt_count = 0

    def mock_run_benchmark(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal attempt_count
        attempt_count += 1
        sim_mode = kwargs.get("simulation_mode", False)
        if not sim_mode:
            raise ConnectionRefusedError("[Errno 61] Connection refused")
        return original_run(*args, **kwargs)

    monkeypatch.setattr(vrag.ui, "run_student_pdf_benchmark", mock_run_benchmark)

    # 1. Default auto_fallback=True: Falls back cleanly to simulation mode
    pairs, stats, notice = execute_benchmark_safely(
        pdf_bytes=pdf_bytes,
        pdf_name="sample.pdf",
        prompts=(
            ("p1", "What is the core methodology?"),
            ("p2", "What are the main findings?"),
        ),
        models=("llama3.1:8b-instruct-q4_K_M",),
        simulation_mode=False,
    )
    assert pairs is not None and len(pairs) == 2
    assert stats is not None and len(stats) == 4
    assert notice is not None
    assert "Notice: Local Ollama daemon is unreachable" in notice
    assert "Deterministic Simulation Mode (Fast Offline)" in notice
    assert attempt_count == 2  # 1st live attempt failed, 2nd simulation attempt succeeded

    # 2. Disabled auto_fallback=False: Yields diagnostic error tip without retrying
    fail_pairs, fail_stats, error_msg = execute_benchmark_safely(
        pdf_bytes=pdf_bytes,
        pdf_name="sample.pdf",
        prompts=(
            ("p1", "What is the core methodology?"),
            ("p2", "What are the main findings?"),
        ),
        models=("llama3.1:8b-instruct-q4_K_M",),
        simulation_mode=False,
        auto_fallback=False,
    )
    assert fail_pairs is None
    assert fail_stats is None
    assert error_msg is not None
    assert "Benchmark execution encountered an error: [Errno 61] Connection refused" in error_msg
    assert "Tip: If Ollama is unreachable" in error_msg

    # 3. If fallback also encounters fatal error, yields diagnostic error
    def mock_all_fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        if not kwargs.get("simulation_mode", False):
            raise ConnectionRefusedError("[Errno 61] Connection refused")
        raise RuntimeError("Fallback environment failure")

    monkeypatch.setattr(vrag.ui, "run_student_pdf_benchmark", mock_all_fail)
    all_fail_pairs, all_fail_stats, all_fail_err = execute_benchmark_safely(
        pdf_bytes=pdf_bytes,
        pdf_name="sample.pdf",
        prompts=(
            ("p1", "What is the core methodology?"),
            ("p2", "What are the main findings?"),
        ),
        models=("llama3.1:8b-instruct-q4_K_M",),
        simulation_mode=False,
        auto_fallback=True,
    )
    assert all_fail_pairs is None
    assert all_fail_stats is None
    assert all_fail_err is not None
    assert "Fallback environment failure" in all_fail_err
