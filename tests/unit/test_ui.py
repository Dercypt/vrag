"""Unit tests for the Streamlit UI view-model builders and 4-metric formatting."""

from __future__ import annotations

import pytest

from vrag.evaluator import run_student_pdf_benchmark
from vrag.generator import OllamaHealthReport
from vrag.ui import (
    _DARK_TOKENS,
    _LIGHT_TOKENS,
    _esc,
    _section,
    build_base_hallucination_evidence,
    build_claim_audit_rows,
    build_four_metrics_summary_rows,
    build_metric_cards_html,
    build_model_hallucination_rows,
    build_ollama_troubleshooting_html,
    build_output_panel_html,
    build_table_html,
    build_theme_css,
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


def test_theme_css_generation_and_tokens() -> None:
    light_css = build_theme_css(dark=False)
    dark_css = build_theme_css(dark=True)

    assert "color-scheme: light;" in light_css
    assert "color-scheme: dark;" in dark_css
    assert f"--bg: {_LIGHT_TOKENS['bg']};" in light_css
    assert f"--bg: {_DARK_TOKENS['bg']};" in dark_css
    assert "font-family: 'Inter'" in light_css
    assert ".vr-card" in dark_css
    assert ".vr-grid" in light_css

    # Section generator with and without subtitle
    sec_with_sub = _section("Results", "Detailed analysis")
    assert '<div class="vr-section">Results</div>' in sec_with_sub
    assert '<div class="vr-sub">Detailed analysis</div>' in sec_with_sub

    sec_no_sub = _section("Settings")
    assert '<div class="vr-section">Settings</div>' in sec_no_sub
    assert "vr-sub" not in sec_no_sub


def test_build_metric_cards_html_rendering() -> None:
    from vrag.evaluator import PairedStatisticalResult

    stats = [
        PairedStatisticalResult(
            metric_name="Factual Precision (FActScore)",
            mean_baseline_a=60.0,
            mean_verified_b=85.0,
            mean_delta=25.0,
            t_statistic=4.2,
            t_pvalue=0.001,
            unit="%",
        ),
        PairedStatisticalResult(
            metric_name="Computational Latency",
            mean_baseline_a=2.5,
            mean_verified_b=1.8,
            mean_delta=-0.7,
            t_statistic=-3.1,
            t_pvalue=0.01,
            unit="s",
        ),
        PairedStatisticalResult(
            metric_name="Citation Accuracy (Precision/Recall)",
            mean_baseline_a=70.0,
            mean_verified_b=70.0,
            mean_delta=0.0,
            t_statistic=0.0,
            t_pvalue=1.0,
            unit="%",
        ),
    ]

    html_out = build_metric_cards_html(stats)
    assert '<div class="vr-grid">' in html_out
    assert "Factual Precision (FActScore)" in html_out
    assert "85.00%" in html_out
    assert "vr-chip good" in html_out
    assert "▲ +25.00%" in html_out
    assert "Base 60.00%" in html_out

    # Latency: lower is better (-0.7 delta is good and arrow is ▼)
    assert "Computational Latency" in html_out
    assert "1.8000s" in html_out
    assert "▼ -0.7000s" in html_out

    # Zero delta: neutral tone with dash
    assert "vr-chip neutral" in html_out
    assert "– +0.00%" in html_out


def test_build_table_html_and_badges() -> None:
    rows = [
        {
            "Sentence #": "1",
            "Atomic Sentence": "Transformers use self-attention.",
            "Gate Decision": "SURVIVED (TP)",
        },
        {
            "Sentence #": "2",
            "Atomic Sentence": "Einstein invented transformers.",
            "Gate Decision": "PRUNED (TN)",
        },
    ]
    html_table = build_table_html(rows)
    assert '<table class="vr-table">' in html_table
    assert "<th>Atomic Sentence</th>" in html_table
    assert '<span class="vr-badge good">SURVIVED (TP)</span>' in html_table
    assert '<span class="vr-badge bad">PRUNED (TN)</span>' in html_table
    assert '<td class="wide">Transformers use self-attention.</td>' in html_table


def test_build_output_panel_html() -> None:
    from vrag.models import ConfusionMatrix, PipelineMode, StageLatencies, TrialTelemetry

    trial = TrialTelemetry(
        trial_id="test_trial",
        pdf_sha256="abc12345",
        prompt="Explain transformers",
        model_name="llama3.1:8b-instruct-q4_K_M",
        mode=PipelineMode.VERIFIED_B,
        latencies=StageLatencies(
            t_chunk=0.1,
            t_retrieve=0.2,
            t_generate=0.7,
            t_verify=0.2345,
        ),
        atomic_factscore=1.0,
        entailment_consistency=1.0,
        citation_precision=1.0,
        citation_recall=1.0,
        confusion_matrix=ConfusionMatrix(tp=2, fp=0, tn=0, fn=0),
        peak_ram_mb=500.0,
        output_summary="Grounded thesis summary.",
    )

    panel_verified = build_output_panel_html(trial, verified=True)
    assert 'class="vr-panel verified"' in panel_verified
    assert "Verification-driven RAG" in panel_verified
    assert "Condition B · unsupported claims pruned" in panel_verified
    assert "Grounded thesis summary." in panel_verified
    assert "FActScore <b>100.0%</b>" in panel_verified
    assert "Latency <b>1.2345s</b>" in panel_verified

    panel_base = build_output_panel_html(trial, verified=False)
    assert 'class="vr-panel base"' in panel_base
    assert "Base RAG" in panel_base
    assert "Condition A · unverified output" in panel_base


def test_ui_edge_cases_and_html_escaping() -> None:
    from vrag.models import ConfusionMatrix, PipelineMode, StageLatencies, TrialTelemetry

    # 1. Empty table returns fallback note
    assert build_table_html([]) == '<div class="vr-note">No rows to display.</div>'

    # 2. Empty output panel falls back to descriptive empty state
    empty_trial = TrialTelemetry(
        trial_id="empty_trial",
        pdf_sha256="abc12345",
        prompt="Explain transformers",
        model_name="llama3.1:8b-instruct-q4_K_M",
        mode=PipelineMode.VERIFIED_B,
        latencies=StageLatencies(t_chunk=0.0, t_retrieve=0.0, t_generate=0.0, t_verify=0.0),
        atomic_factscore=0.0,
        entailment_consistency=0.0,
        citation_precision=0.0,
        citation_recall=0.0,
        confusion_matrix=ConfusionMatrix(tp=0, fp=0, tn=2, fn=0),
        peak_ram_mb=400.0,
        output_summary="",
    )

    empty_verified = build_output_panel_html(empty_trial, verified=True)
    assert 'class="body empty"' in empty_verified
    assert "All sentences were pruned by the verification gate." in empty_verified

    empty_base = build_output_panel_html(empty_trial, verified=False)
    assert 'class="body empty"' in empty_base
    assert "No output." in empty_base

    # 3. HTML escaping prevents XSS / markup corruption
    assert (
        _esc("<script>alert('xss')</script>")
        == "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"
    )
    xss_row = [
        {
            "Claim": "<script>alert('xss')</script>",
            "Gate Decision": "PRUNED (TN)",
        }
    ]
    rendered_xss = build_table_html(xss_row)
    assert "<script>" not in rendered_xss
    assert "&lt;script&gt;" in rendered_xss


def test_build_ollama_troubleshooting_html_rendering() -> None:
    # 1. Offline report renders setup guide and commands
    offline_report = OllamaHealthReport(
        is_online=False,
        base_url="http://localhost:11434",
        installed_models=(),
        missing_approved_models=("llama3.1:8b-instruct-q4_K_M",),
        message="Local Ollama daemon is unreachable at http://localhost:11434.",
        troubleshooting_steps=(
            "Install Ollama: brew install --cask ollama",
            "Start the daemon in a terminal: ollama serve",
        ),
    )
    html_offline = build_ollama_troubleshooting_html(offline_report)
    assert "How to Fix: Ollama Offline" in html_offline
    assert "brew install --cask ollama" in html_offline
    assert "ollama serve" in html_offline

    # 2. Online with missing models
    partial_report = OllamaHealthReport(
        is_online=True,
        base_url="http://localhost:11434",
        installed_models=("mistral:7b-instruct-v0.3-q4_K_M",),
        missing_approved_models=("llama3.1:8b-instruct-q4_K_M",),
        message="Ollama is online, but 1 model is missing.",
        troubleshooting_steps=("ollama pull llama3.1:8b-instruct-q4_K_M",),
    )
    html_partial = build_ollama_troubleshooting_html(partial_report)
    assert "Missing Approved Models" in html_partial
    assert "ollama pull llama3.1:8b-instruct-q4_K_M" in html_partial

    # 3. Online and fully ready
    ready_report = OllamaHealthReport(
        is_online=True,
        base_url="http://localhost:11434",
        installed_models=("llama3.1:8b-instruct-q4_K_M",),
        missing_approved_models=(),
        message="Ollama is ready.",
        troubleshooting_steps=(),
    )
    html_ready = build_ollama_troubleshooting_html(ready_report)
    assert "Daemon Ready" in html_ready


def test_build_model_hallucination_rows_and_base_evidence() -> None:
    from vrag.models import APPROVED_MODELS

    pdf_bytes = (
        b"Bang et al. (2025) evaluated 7B to 8B open-source models on HalluLens and found "
        b"baseline hallucination rates between 48% and 85%. "
        b"Kadavath et al. (2022) showed smaller models act overconfident when wrong.\f"
        b"Wang et al. (2025) proposed dynamic semantic chunking using cosine similarity. "
        b"Qian et al. (2025) introduced VeriCite to prune unentailed claims via sentence-level NLI."
    )
    pairs, _stats = run_student_pdf_benchmark(
        pdf_bytes=pdf_bytes,
        pdf_name="thesis_sample.pdf",
        models=APPROVED_MODELS,
        simulation_mode=True,
    )

    model_rows = build_model_hallucination_rows(pairs)
    assert len(model_rows) == 3
    for row in model_rows:
        assert "Model" in row
        assert "Base Hallucination Rate" in row
        assert "Verified Hallucination Rate" in row
        assert "Intrinsic (Contradictions)" in row
        assert "Extrinsic (Fabricated)" in row
        assert "Verified FActScore" in row
        assert "Avg Latency" in row

    # Condition A of first trial must yield concrete hallucination evidence
    trial_a, _trial_b = pairs[0]
    evidence = build_base_hallucination_evidence(trial_a)
    assert len(evidence) >= 1
    types = {item["Type"] for item in evidence}
    assert any("Intrinsic" in t or "Extrinsic" in t for t in types)
    assert all("Flagged Claim" in item and "Detail" in item for item in evidence)


def test_ui_script_executes_without_pythonpath() -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    ui_path = repo_root / "src" / "vrag" / "ui.py"
    src_dir = str((repo_root / "src").resolve())

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    code = (
        "import runpy, sys; "
        f"sys.path[:] = [p for p in sys.path if p != {src_dir!r}]; "
        f"runpy.run_path({str(ui_path)!r}, run_name='__vrag_import_check__')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"ui.py failed without PYTHONPATH:\n{proc.stderr}"


def test_ui_streamlit_apptest_without_pythonpath_and_rerun_idempotency() -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path

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
        f"at = AppTest.from_file({str(ui_path)!r}).run(); "
        "assert not at.exception, list(at.exception)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        "Streamlit AppTest failed without PYTHONPATH:\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
