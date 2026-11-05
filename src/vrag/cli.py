"""CLI entrypoint for comparative RAG benchmarking."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from vrag.evaluator import PairedStatisticalResult

console = Console()


def render_comparison_table(results: list[PairedStatisticalResult]) -> Table:
    """Build a Rich comparative table for Condition A vs Condition B."""
    table = Table(title="Condition A (Baseline RAG) vs. Condition B (Verification-Driven RAG)")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Condition A (Mean)", justify="right")
    table.add_column("Condition B (Mean)", justify="right")
    table.add_column("Delta (B - A)", justify="right", style="bold green")
    table.add_column("Paired t-stat", justify="right")
    table.add_column("p-value", justify="right", style="bold yellow")

    for res in results:
        table.add_row(
            res.metric_name,
            f"{res.mean_baseline_a:.4f}",
            f"{res.mean_verified_b:.4f}",
            f"{res.mean_delta:+.4f}",
            f"{res.t_statistic:.4f}",
            f"{res.t_pvalue:.4e}",
        )
    return table
