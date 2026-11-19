"""CLI entrypoint for comparative RAG benchmarking."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from vrag.corpus import run_curated_corpus_benchmark
from vrag.evaluator import (
    PairedStatisticalResult,
    export_summary_json,
    export_telemetry_csv,
    run_student_pdf_benchmark,
)
from vrag.generator import OllamaHealthReport, diagnose_ollama
from vrag.models import APPROVED_MODELS, STANDARDIZED_SUMMARIZATION_PROMPT, TrialTelemetry

console = Console()

# --- Formatting & Rich Table Renderers ---


def format_metric_value(value: float, unit: str, signed: bool = False) -> str:
    """Format a metric value with its unit ('%' or 's')."""
    sign = "+" if signed else ""
    return f"{value:{sign}.4f}s" if unit == "s" else f"{value:{sign}.2f}%"


def render_comparison_table(results: list[PairedStatisticalResult]) -> Table:
    """Build a Rich comparative table for Condition A vs Condition B across the 4 core metrics."""
    table = Table(title="Condition A (Base RAG) vs. Condition B (Verification-Driven RAG)")
    for label, style, justify in (
        ("Metric", "bold cyan", "left"),
        ("Base (Mean)", None, "right"),
        ("Verified (Mean)", None, "right"),
        ("Delta", "bold green", "right"),
        ("SD", None, "right"),
        ("Err Red", None, "right"),
        ("t-stat", None, "right"),
        ("p-value", "bold yellow", "right"),
    ):
        table.add_column(label, style=style, justify=justify, no_wrap=(label == "Metric"))  # type: ignore[arg-type]

    for res in results:
        table.add_row(
            res.metric_name,
            format_metric_value(res.mean_baseline_a, res.unit),
            format_metric_value(res.mean_verified_b, res.unit),
            format_metric_value(res.mean_delta, res.unit, signed=True),
            f"{res.std_delta:.2f}",
            f"{res.pct_error_reduction:+.1f}%",
            f"{res.t_statistic:.4f}",
            f"{res.t_pvalue:.4e}",
        )
    return table


def render_trial_telemetry_table(trials: list[TrialTelemetry]) -> Table:
    """Build a per-trial Rich table showing the 4 core metrics, confusion matrix, and RAM usage."""
    table = Table(title="Per-Trial Execution Telemetry")
    for label, style, justify in (
        ("Trial ID", "cyan", "left"),
        ("Model", "magenta", "left"),
        ("Mode", None, "left"),
        ("FActScore (%)", None, "right"),
        ("NLI Consistency (%)", None, "right"),
        ("Citation Acc (%)", None, "right"),
        ("Latency (s)", None, "right"),
        ("TP/TN/FP/FN", None, "center"),
        ("RAM (MB)", None, "right"),
    ):
        table.add_column(label, style=style, justify=justify)  # type: ignore[arg-type]

    for t in trials:
        cm = t.confusion_matrix
        table.add_row(
            t.trial_id,
            t.model_name,
            t.mode.value,
            f"{t.factual_precision_pct:.1f}%",
            f"{t.entailment_consistency_pct:.1f}%",
            f"{t.citation_accuracy_pct:.1f}%",
            f"{t.computational_latency_sec:.4f}s",
            f"{cm.tp}/{cm.tn}/{cm.fp}/{cm.fn}",
            f"{t.peak_ram_mb:.1f}",
        )
    return table


def render_model_comparison_table(
    pairs: list[tuple[TrialTelemetry, TrialTelemetry]],
) -> Table:
    """Build a per-model Rich table showing visible hallucination reduction (Base -> Verified)."""
    table = Table(title="Hallucination Reduction by Model (7B-9B Tier)")
    for label, style, justify in (
        ("Model", "bold magenta", "left"),
        ("Base Halluc (%)", None, "right"),
        ("Verified Halluc (%)", "bold green", "right"),
        ("Intrinsic (A->B)", None, "center"),
        ("Extrinsic (A->B)", None, "center"),
        ("Verified FActScore", None, "right"),
    ):
        table.add_column(label, style=style, justify=justify)  # type: ignore[arg-type]

    grouped: dict[str, list[tuple[TrialTelemetry, TrialTelemetry]]] = {}
    for pair in pairs:
        grouped.setdefault(pair[0].model_name, []).append(pair)

    for model_name, model_pairs in grouped.items():
        n = len(model_pairs)
        base_h = sum(a.hallucination_error_rate_pct for a, _ in model_pairs) / n
        ver_h = sum(b.hallucination_error_rate_pct for _, b in model_pairs) / n
        intr_a = sum(a.intrinsic_hallucinations for a, _ in model_pairs)
        intr_b = sum(b.intrinsic_hallucinations for _, b in model_pairs)
        extr_a = sum(a.extrinsic_hallucinations for a, _ in model_pairs)
        extr_b = sum(b.extrinsic_hallucinations for _, b in model_pairs)
        ver_fs = sum(b.factual_precision_pct for _, b in model_pairs) / n
        table.add_row(
            model_name,
            f"{base_h:.1f}%",
            f"{ver_h:.1f}%",
            f"{intr_a} -> {intr_b}",
            f"{extr_a} -> {extr_b}",
            f"{ver_fs:.1f}%",
        )
    return table


def _export_and_print_results(
    pairs: list[tuple[TrialTelemetry, TrialTelemetry]],
    stats: list[PairedStatisticalResult],
    output_dir: Path | str,
) -> None:
    """Persist CSV/JSON artifacts and render Rich comparative summary tables."""
    all_trials = [t for pair in pairs for t in pair]
    out_path = Path(output_dir)
    export_telemetry_csv(all_trials, out_path / "telemetry.csv")
    export_summary_json(all_trials, stats, out_path / "summary.json")
    console.print(render_trial_telemetry_table(all_trials))
    console.print(render_model_comparison_table(pairs))
    console.print(render_comparison_table(stats))


# --- Benchmark & Diagnostics Runners ---

_DEMO_PDF_BYTES = (
    b"Bang et al. (2025) evaluated 7B to 9B open-source models including Llama-3.1-8B, "
    b"Gemma-2-9B, Mistral-7B, and Qwen-2.5-7B on HalluLens, finding 48% to 85% baseline error "
    b"rates. Kadavath et al. (2022) showed smaller models act overconfident when wrong.\f"
    b"Wang et al. (2025) proposed dynamic semantic chunking using cosine similarity thresholds. "
    b"Qian et al. (2025) introduced VeriCite to prune unentailed claims via sentence-level NLI."
)


def run_synthetic_demo_benchmark(output_dir: Path | str) -> list[PairedStatisticalResult]:
    """Run an offline deterministic 4-model benchmark sweep and export CSV/JSON artifacts."""
    pairs, stats = run_student_pdf_benchmark(
        pdf_bytes=_DEMO_PDF_BYTES,
        pdf_name="thesis_demo.pdf",
        models=APPROVED_MODELS,
        top_k=2,
        simulation_mode=True,
    )
    _export_and_print_results(pairs, stats, output_dir)
    return stats


def run_doctor_check(report_override: OllamaHealthReport | None = None) -> int:
    """Run health check against local Ollama service and print rich diagnostics."""
    report = report_override if report_override is not None else diagnose_ollama()
    status_style = "bold green" if report.is_online else "bold red"
    status_label = "ONLINE" if report.is_online else "OFFLINE"

    console.print("\n[bold]vrag Environment Doctor[/bold]")
    console.print(
        f"Ollama Daemon: [{status_style}]{status_label}[/{status_style}] ({report.base_url})"
    )
    console.print(f"Status Note:   {report.message}\n")

    if report.is_online:
        console.print(f"[bold cyan]Installed Models ({len(report.installed_models)}):[/bold cyan]")
        for m in report.installed_models:
            console.print(f"  • {m}")
        if report.missing_approved_models:
            console.print(
                f"\n[bold yellow]Missing Approved Models "
                f"({len(report.missing_approved_models)}):[/bold yellow]"
            )
            for m in report.missing_approved_models:
                console.print(f"  • {m} (Run: ollama pull {m})")
        else:
            console.print(
                "\n[bold green]✓ All approved thesis models are present and ready.[/bold green]"
            )
    else:
        console.print("[bold yellow]Troubleshooting & Remediation Steps:[/bold yellow]")
        for idx, step in enumerate(report.troubleshooting_steps, 1):
            console.print(f"  {idx}. {step}")

    return 0 if (report.is_online and not report.missing_approved_models) else 1


def _resolve_simulation_mode(force_simulation: bool, target_label: str) -> bool:
    """Return True if offline simulation should be used, printing a notice when falling back."""
    if force_simulation:
        return True
    health = diagnose_ollama()
    if health.is_online and not health.missing_approved_models:
        return False
    console.print(
        f"[yellow]Notice: {health.message} "
        f"Running {target_label} in Deterministic Simulation Mode (Fast Offline).[/yellow]"
    )
    return True


# --- CLI Entrypoint ---


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for vrag."""
    parser = argparse.ArgumentParser(
        prog="vrag",
        description="Comparative Benchmarking Harness: Baseline vs. Verification-Driven RAG",
    )
    parser.add_argument("--pdf", type=str, help="Path to academic PDF file to benchmark.")
    parser.add_argument(
        "--corpus",
        type=str,
        help="Path to curated benchmark corpus directory (e.g. data/benchmark_corpus).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=STANDARDIZED_SUMMARIZATION_PROMPT,
        help="Prompt query to run across Condition A and Condition B.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to write telemetry.csv and summary.json.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run deterministic offline comparative benchmark across all 4 models.",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Force deterministic offline simulation mode.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Diagnose local Ollama daemon connectivity and approved models.",
    )
    args = parser.parse_args(argv)

    if args.doctor:
        return run_doctor_check()

    if args.corpus:
        sim_mode = _resolve_simulation_mode(args.simulation, "curated corpus")
        pairs, stats = run_curated_corpus_benchmark(
            corpus_dir=Path(args.corpus),
            models=APPROVED_MODELS,
            top_k=3,
            simulation_mode=sim_mode,
        )
        _export_and_print_results(pairs, stats, args.output_dir)
        return 0

    if args.demo or not args.pdf:
        run_synthetic_demo_benchmark(output_dir=args.output_dir)
        return 0

    pdf_path = Path(args.pdf)
    sim_mode = _resolve_simulation_mode(args.simulation, "benchmark")
    pairs, stats = run_student_pdf_benchmark(
        pdf_bytes=pdf_path.read_bytes(),
        pdf_name=pdf_path.name,
        models=APPROVED_MODELS,
        top_k=3,
        simulation_mode=sim_mode,
    )
    _export_and_print_results(pairs, stats, args.output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
