"""Unit tests for the CLI comparative benchmark runner and Rich table rendering."""

from __future__ import annotations

from pathlib import Path

from vrag.cli import main, render_comparison_table, run_synthetic_demo_benchmark


def test_run_synthetic_demo_benchmark_executes_all_models_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    stats = run_synthetic_demo_benchmark(output_dir=tmp_path)
    assert len(stats) == 5

    table = render_comparison_table(stats)
    assert table.row_count == 5

    assert (tmp_path / "telemetry.csv").exists()
    assert (tmp_path / "summary.json").exists()


def test_cli_main_demo_flag_exits_zero(tmp_path: Path) -> None:
    exit_code = main(["--demo", "--output-dir", str(tmp_path)])
    assert exit_code == 0
    assert (tmp_path / "telemetry.csv").exists()
