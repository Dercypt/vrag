"""Unit tests for the CLI comparative benchmark runner and Rich table rendering."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from vrag.cli import (
    format_metric_value,
    main,
    render_comparison_table,
    run_doctor_check,
    run_synthetic_demo_benchmark,
)
from vrag.generator import OllamaHealthReport


def test_format_metric_value_formats_percentage_and_seconds() -> None:
    assert format_metric_value(92.5, "%") == "92.50%"
    assert format_metric_value(1.23456, "s") == "1.2346s"
    assert format_metric_value(15.0, "%", signed=True) == "+15.00%"
    assert format_metric_value(-0.05, "s", signed=True) == "-0.0500s"


def test_run_synthetic_demo_benchmark_executes_all_models_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    stats = run_synthetic_demo_benchmark(output_dir=tmp_path)
    assert len(stats) == 4

    table = render_comparison_table(stats)
    assert table.row_count == 4
    console = Console(record=True, width=120)
    console.print(table)
    rendered_text = console.export_text()
    assert "Factual Precision (FActScore)" in rendered_text
    assert "Entailment Consistency (NLI Score)" in rendered_text
    assert "Citation Accuracy (Precision/Recall)" in rendered_text
    assert "Computational Latency" in rendered_text
    assert "%" in rendered_text

    assert (tmp_path / "telemetry.csv").exists()
    assert (tmp_path / "summary.json").exists()


def test_cli_main_demo_flag_exits_zero(tmp_path: Path) -> None:
    exit_code = main(["--demo", "--output-dir", str(tmp_path)])
    assert exit_code == 0
    assert (tmp_path / "telemetry.csv").exists()


def test_cli_main_pdf_input_runs_simulation_when_ollama_offline(tmp_path: Path) -> None:
    sample_pdf = tmp_path / "test_doc.pdf"
    sample_pdf.write_bytes(
        b"Vaswani et al. (2017) introduced transformers.\f"
        b"Lewis et al. (2020) proposed retrieval-augmented generation."
    )
    exit_code = main(["--pdf", str(sample_pdf), "--output-dir", str(tmp_path)])
    assert exit_code == 0
    assert (tmp_path / "telemetry.csv").exists()
    assert (tmp_path / "summary.json").exists()


def test_run_doctor_check_offline_returns_exit_code_1() -> None:
    offline_report = OllamaHealthReport(
        is_online=False,
        base_url="http://localhost:11434",
        installed_models=(),
        missing_approved_models=("llama3.1:8b-instruct-q4_K_M",),
        message="Local Ollama daemon is unreachable.",
        troubleshooting_steps=("Install Ollama: brew install --cask ollama",),
    )
    code = run_doctor_check(report_override=offline_report)
    assert code == 1


def test_run_doctor_check_online_ready_returns_0() -> None:
    online_report = OllamaHealthReport(
        is_online=True,
        base_url="http://localhost:11434",
        installed_models=("llama3.1:8b-instruct-q4_K_M",),
        missing_approved_models=(),
        message="Ready",
        troubleshooting_steps=(),
    )
    code = run_doctor_check(report_override=online_report)
    assert code == 0


def test_cli_main_doctor_flag() -> None:
    # Running --doctor against local environment
    exit_code = main(["--doctor"])
    assert exit_code in (0, 1)
