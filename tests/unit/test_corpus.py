"""Unit tests for the Thesis Writing 1 curated academic PDF benchmark corpus and oracle."""

from __future__ import annotations

from pathlib import Path

from vrag.cli import main
from vrag.corpus import (
    DEFAULT_CORPUS_DIR,
    build_ground_truth_oracle,
    ensure_corpus_pdfs,
    load_benchmark_corpus,
    run_curated_corpus_benchmark,
)
from vrag.models import GroundTruthSpec


def test_build_ground_truth_oracle_classifies_supported_vs_traps() -> None:
    spec = GroundTruthSpec(
        supported_claims=(
            "Quitaleg and Ramos (2025) evaluated 120 college modules and found a 62% error rate.",
            "Semantic cosine chunking preserves complete academic arguments across sentences.",
        ),
        intrinsic_traps=(
            "Quitaleg and Ramos (2025) found a 0% baseline error rate across 120 modules.",
        ),
        extrinsic_traps=(
            "Paningbatan (2024) proved that cloud GPUs eliminate all hallucinations.",
        ),
        valid_citations=(("Quitaleg", "2025"), ("Ramos", "2025")),
    )
    oracle = build_ground_truth_oracle(spec)

    assert oracle("Quitaleg and Ramos (2025) tested 120 college modules, finding a 62% error rate.")
    assert not oracle("Quitaleg and Ramos (2025) found a 0% baseline error rate.")
    assert not oracle("Paningbatan (2024) proved that cloud GPUs eliminate all hallucinations.")
    assert not oracle("Quitaleg and Ramos (2019) evaluated 120 college modules.")


def test_load_default_benchmark_corpus_and_run_in_simulation_mode(tmp_path: Path) -> None:
    ensure_corpus_pdfs(DEFAULT_CORPUS_DIR)
    items = load_benchmark_corpus(DEFAULT_CORPUS_DIR)
    assert len(items) >= 6
    for item in items:
        assert item.pdf_bytes.startswith(b"%PDF")
        assert len(item.ground_truth.supported_claims) >= 2

    pairs, stats = run_curated_corpus_benchmark(
        corpus_dir=DEFAULT_CORPUS_DIR,
        models=("llama3.1:8b-instruct-q4_K_M",),
        simulation_mode=True,
    )
    assert len(pairs) == len(items)
    assert len(stats) == 4
    assert stats[0].mean_verified_b >= 90.0
    assert stats[0].mean_verified_b > stats[0].mean_baseline_a

    exit_code = main(
        [
            "--corpus",
            str(DEFAULT_CORPUS_DIR),
            "--simulation",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "telemetry.csv").exists()
    assert (tmp_path / "summary.json").exists()
