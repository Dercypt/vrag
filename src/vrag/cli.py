"""CLI entrypoint for comparative RAG benchmarking."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from vrag.evaluator import (
    ComparativeEvaluator,
    PairedStatisticalResult,
    compute_suite_statistics,
    export_summary_json,
    export_telemetry_csv,
)
from vrag.generator import OllamaGenerator
from vrag.models import APPROVED_MODELS, NLILabel, NLIScore, TrialTelemetry

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


def render_trial_telemetry_table(trials: list[TrialTelemetry]) -> Table:
    """Build a per-trial Rich table showing timings, confusion matrix, and RAM usage."""
    table = Table(title="Per-Trial Execution Telemetry")
    table.add_column("Trial ID", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Mode")
    table.add_column("FActScore", justify="right")
    table.add_column("TP/TN/FP/FN", justify="center")
    table.add_column("t_total (s)", justify="right")
    table.add_column("RAM (MB)", justify="right")

    for t in trials:
        cm = t.confusion_matrix
        table.add_row(
            t.trial_id,
            t.model_name,
            t.mode.value,
            f"{t.atomic_factscore:.3f}",
            f"{cm.tp}/{cm.tn}/{cm.fp}/{cm.fn}",
            f"{t.latencies.total_seconds:.4f}",
            f"{t.peak_ram_mb:.1f}",
        )
    return table


def run_synthetic_demo_benchmark(output_dir: Path | str) -> list[PairedStatisticalResult]:
    """Run an offline deterministic multi-model benchmark sweep and export CSV/JSON artifacts."""
    out_path = Path(output_dir)
    sample_docs = [
        (
            "doc1",
            (
                b"Vaswani et al. (2017) introduced the Transformer architecture using "
                b"self-attention mechanisms. Self-attention computes token representations.\f"
                b"Lewis et al. (2020) introduced Retrieval-Augmented Generation for NLP tasks."
            ),
            "Summarize the Transformer and RAG architectures.",
        ),
        (
            "doc2",
            (
                b"He et al. (2021) presented DeBERTa using disentangled attention for NLI.\f"
                b"Min et al. (2023) introduced FActScore for atomic factual precision evaluation."
            ),
            "How are NLI and FActScore used in factual verification?",
        ),
    ]

    def demo_embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.2] if ("Transformer" in t or "DeBERTa" in t) else [0.2, 1.0] for t in texts]

    def demo_post(_url: str, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("keep_alive") == 0:
            return {"done": True}
        prompt = str(payload.get("prompt", ""))
        if "VERIFIABLE SUMMARY:" in prompt:
            return {
                "response": (
                    "Vaswani et al. (2017) proposed the Transformer architecture [C1]. "
                    "According to (FakeAuthor, 2024), recurrent gates are mandatory [C1]."
                )
            }
        return {
            "response": (
                "Vaswani et al. (2017) proposed the Transformer architecture. "
                "Recurrent gates completely replace attention in all modern LLMs."
            )
        }

    def demo_nli(_premise: str, hypothesis: str) -> NLIScore:
        if "recurrent gates" in hypothesis.lower():
            return NLIScore(
                label=NLILabel.CONTRADICTION,
                entailment_prob=0.02,
                contradiction_prob=0.95,
            )
        return NLIScore(label=NLILabel.ENTAILMENT, entailment_prob=0.96)

    evaluator = ComparativeEvaluator(
        generator=OllamaGenerator(http_post_fn=demo_post),
        embed_fn=demo_embed,
        nli_scorer=demo_nli,
    )
    pairs = evaluator.run_sequential_model_sweep(
        documents=sample_docs,
        models=APPROVED_MODELS,
        top_k=2,
    )
    all_trials = [t for pair in pairs for t in pair]
    stats = compute_suite_statistics(pairs)

    export_telemetry_csv(all_trials, out_path / "telemetry.csv")
    export_summary_json(all_trials, stats, out_path / "summary.json")

    console.print(render_trial_telemetry_table(all_trials))
    console.print(render_comparison_table(stats))
    return stats


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for vrag."""
    parser = argparse.ArgumentParser(
        prog="vrag",
        description="Comparative Benchmarking Harness: Baseline vs. Verification-Driven RAG",
    )
    parser.add_argument("--pdf", type=str, help="Path to academic PDF file to benchmark.")
    parser.add_argument(
        "--query",
        type=str,
        default="Summarize the key contributions and methodology of this paper.",
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
        help="Run deterministic offline comparative benchmark across all 3 models.",
    )
    args = parser.parse_args(argv)

    if args.demo or not args.pdf:
        run_synthetic_demo_benchmark(output_dir=args.output_dir)
        return 0

    pdf_path = Path(args.pdf)
    pdf_bytes = pdf_path.read_bytes()
    evaluator = ComparativeEvaluator()
    pairs = evaluator.run_sequential_model_sweep(
        documents=[(pdf_path.stem, pdf_bytes, args.query)],
        models=APPROVED_MODELS,
        top_k=3,
    )
    all_trials = [t for pair in pairs for t in pair]
    stats = compute_suite_statistics(pairs)

    out_dir = Path(args.output_dir)
    export_telemetry_csv(all_trials, out_dir / "telemetry.csv")
    export_summary_json(all_trials, stats, out_dir / "summary.json")

    console.print(render_trial_telemetry_table(all_trials))
    console.print(render_comparison_table(stats))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
