"""Unit tests for the minimal pure-Python Streamlit UI view-models and AppTest execution."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vrag.evaluator import run_student_pdf_benchmark
from vrag.models import APPROVED_MODELS, STANDARDIZED_SUMMARIZATION_PROMPT
from vrag.ui import (
    build_base_hallucination_evidence,
    build_claim_audit_rows,
    build_four_metrics_summary_rows,
    build_model_hallucination_rows,
    build_trial_comparison_rows,
    execute_benchmark_safely,
    format_model_label,
    is_connection_refused_error,
    resolve_benchmark_prompts,
)


def test_format_model_label_covers_all_four_manuscript_models() -> None:
    labels = [format_model_label(m) for m in APPROVED_MODELS]
    assert labels == ["Llama-3.1-8B", "Gemma-2-9B", "Mistral-7B", "Qwen-2.5-7B"]


def test_resolve_benchmark_prompts_uses_uniform_prompt_for_multi_model_runs() -> None:
    # Chapter 3 (pp. 58-59): single uniform prompt when running >= 2 models
    prompts = resolve_benchmark_prompts("", model_count=4)
    assert prompts == (("standardized", STANDARDIZED_SUMMARIZATION_PROMPT),)

    custom = resolve_benchmark_prompts("Summarize key findings.", model_count=4)
    assert custom == (("custom", "Summarize key findings."),)

    # Single-model selection falls back to >= 2 prompts so paired t-test satisfies n >= 2
    single_model_prompts = resolve_benchmark_prompts("", model_count=1)
    assert len(single_model_prompts) >= 2


def test_ui_view_models_format_four_metrics_models_and_claim_audit() -> None:
    pdf_bytes = (
        b"Bang et al. (2025) evaluated 7B to 9B open-source models on HalluLens and found "
        b"baseline hallucination rates between 48% and 85%. "
        b"Kadavath et al. (2022) showed smaller models act overconfident when wrong.\f"
        b"Wang et al. (2025) proposed dynamic semantic chunking using cosine similarity. "
        b"Qian et al. (2025) introduced VeriCite to prune unentailed claims via sentence-level NLI."
    )
    pairs, stats = run_student_pdf_benchmark(
        pdf_bytes=pdf_bytes,
        pdf_name="thesis_sample.pdf",
        prompts=(("standardized", STANDARDIZED_SUMMARIZATION_PROMPT),),
        models=APPROVED_MODELS,
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

    model_rows = build_model_hallucination_rows(pairs)
    assert len(model_rows) == 4
    for row in model_rows:
        assert row["Base Hallucination Rate"] not in ("0.0%", "100.0%")
        assert row["Verified Hallucination Rate"] not in ("0.0%", "100.0%")
        assert "→" in str(row["Intrinsic (Contradictions)"])
        assert "→" in str(row["Extrinsic (Fabricated)"])

    trial_rows = build_trial_comparison_rows(pairs)
    assert len(trial_rows) == 4
    assert "Base FActScore (%)" in trial_rows[0]
    assert "Verified FActScore (%)" in trial_rows[0]
    assert "Verified Latency (s)" in trial_rows[0]

    trial_a, trial_b = pairs[0]
    evidence = build_base_hallucination_evidence(trial_a)
    assert len(evidence) >= 2
    assert any("Intrinsic" in item["Type"] for item in evidence)
    assert any("Extrinsic" in item["Type"] for item in evidence)

    audit_rows = build_claim_audit_rows(trial_b)
    assert len(audit_rows) >= 2
    statuses = [r["Gate Decision"] for r in audit_rows]
    assert "SURVIVED (TP)" in statuses
    assert "PRUNED (TN)" in statuses


def test_is_connection_refused_error() -> None:
    assert is_connection_refused_error(ConnectionRefusedError("Connection refused"))
    assert is_connection_refused_error(ConnectionResetError("Connection reset by peer"))
    assert is_connection_refused_error(OSError(errno.ECONNREFUSED, "Connection refused"))
    assert is_connection_refused_error(RuntimeError("[Errno 61] Connection refused"))
    assert is_connection_refused_error(RuntimeError("Failed to establish a new connection"))

    class FakeConnectError(Exception):
        pass

    assert is_connection_refused_error(FakeConnectError("Cannot reach localhost:11434"))
    assert not is_connection_refused_error(ValueError("Invalid token size"))
    assert not is_connection_refused_error(ImportError("Missing module"))


def test_execute_benchmark_safely_and_connection_refused_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrag.ui

    pdf_bytes = b"Sample text for safe execution."
    prompts = (("p1", "Methodology?"), ("p2", "Findings?"))

    pairs, stats, err = execute_benchmark_safely(
        pdf_bytes=pdf_bytes,
        pdf_name="sample.pdf",
        prompts=prompts,
        models=("llama3.1:8b-instruct-q4_K_M",),
        simulation_mode=True,
    )
    assert err is None
    assert pairs is not None and len(pairs) == 2
    assert stats is not None and len(stats) == 4

    original_run = vrag.ui.run_student_pdf_benchmark

    def mock_refused(*args, **kwargs):  # type: ignore[no-untyped-def]
        if not kwargs.get("simulation_mode", False):
            raise ConnectionRefusedError("[Errno 61] Connection refused")
        return original_run(*args, **kwargs)

    monkeypatch.setattr(vrag.ui, "run_student_pdf_benchmark", mock_refused)
    fb_pairs, fb_stats, notice = execute_benchmark_safely(
        pdf_bytes=pdf_bytes,
        pdf_name="sample.pdf",
        prompts=prompts,
        models=("llama3.1:8b-instruct-q4_K_M",),
        simulation_mode=False,
    )
    assert fb_pairs is not None and fb_stats is not None
    assert notice is not None and "Offline Simulation Mode" in notice

    fail_pairs, fail_stats, fail_msg = execute_benchmark_safely(
        pdf_bytes=pdf_bytes,
        pdf_name="sample.pdf",
        prompts=prompts,
        models=("llama3.1:8b-instruct-q4_K_M",),
        simulation_mode=False,
        auto_fallback=False,
    )
    assert fail_pairs is None and fail_stats is None
    assert fail_msg is not None and "Connection refused" in fail_msg


def test_ui_streamlit_apptest_without_pythonpath_and_run_flow() -> None:
    pytest.importorskip("streamlit.testing.v1")

    repo_root = Path(__file__).resolve().parents[2]
    ui_path = repo_root / "src" / "vrag" / "ui.py"
    src_dir = str((repo_root / "src").resolve())

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    code = (
        "import runpy, sys; "
        "from streamlit.testing.v1 import AppTest; "
        f"sys.path[:] = [p for p in sys.path if p != {src_dir!r}]; "
        f"runpy.run_path({str(ui_path)!r}, run_name='__vrag_rerun_1__'); "
        f"runpy.run_path({str(ui_path)!r}, run_name='__vrag_rerun_2__'); "
        f"assert sys.path.count({src_dir!r}) == 1, sys.path; "
        f"sys.path[:] = [p for p in sys.path if p != {src_dir!r}]; "
        f"at = AppTest.from_file({str(ui_path)!r}, default_timeout=30).run(); "
        "assert not at.exception, list(at.exception); "
        "at.sidebar.button[0].click().run(); "
        "assert not at.exception, list(at.exception); "
        "assert len(at.metric) == 4, len(at.metric)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        proc.returncode == 0
    ), f"Streamlit AppTest failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
