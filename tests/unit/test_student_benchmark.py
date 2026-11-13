"""Unit tests for the multi-prompt student PDF benchmark suite and Ollama preflight check."""

from __future__ import annotations

import pytest

from vrag.evaluator import run_student_pdf_benchmark
from vrag.generator import check_ollama_status
from vrag.models import DEFAULT_STUDENT_BENCHMARK_PROMPTS


def test_default_student_benchmark_prompts_has_four_standardized_queries() -> None:
    assert len(DEFAULT_STUDENT_BENCHMARK_PROMPTS) == 4
    keys = [k for k, _ in DEFAULT_STUDENT_BENCHMARK_PROMPTS]
    assert keys == ["contributions", "methodology", "findings", "limitations"]


def test_check_ollama_status_rejects_non_local_and_handles_unreachable_local() -> None:
    with pytest.raises(ValueError, match="LAW 2 violation"):
        check_ollama_status("https://api.openai.com")

    # Using an unused loopback port should cleanly return False without raising
    assert check_ollama_status("http://127.0.0.1:59999", timeout=0.1) is False


def test_run_student_pdf_benchmark_in_simulation_mode_computes_four_metrics() -> None:
    student_pdf_bytes = (
        b"Smith et al. (2023) investigated lightweight open-source LLMs for academic RAG. "
        b"They demonstrated that fixed-window chunking splits semantic context.\f"
        b"Lee & Patel (2024) proposed a DeBERTa NLI verification gate that prunes "
        b"unentailed claims and fabricated citations on consumer hardware."
    )

    pairs, stats = run_student_pdf_benchmark(
        pdf_bytes=student_pdf_bytes,
        pdf_name="student_thesis.pdf",
        models=("llama3.1:8b-instruct-q4_K_M",),
        simulation_mode=True,
    )

    # 4 standardized benchmark prompts x 1 model = 4 paired trials
    assert len(pairs) == 4
    assert len(stats) == 4

    # Condition B must achieve higher Factual Precision, Entailment Consistency, & Citation Accuracy
    factscore_stat = stats[0]
    entailment_stat = stats[1]
    citation_stat = stats[2]
    latency_stat = stats[3]

    assert factscore_stat.metric_name == "Factual Precision (FActScore)"
    assert factscore_stat.mean_verified_b > factscore_stat.mean_baseline_a
    assert factscore_stat.unit == "%"

    assert entailment_stat.metric_name == "Entailment Consistency (NLI Score)"
    assert entailment_stat.mean_verified_b > entailment_stat.mean_baseline_a
    assert entailment_stat.unit == "%"

    assert citation_stat.metric_name == "Citation Accuracy (Precision/Recall)"
    assert citation_stat.mean_verified_b > citation_stat.mean_baseline_a
    assert citation_stat.unit == "%"

    assert latency_stat.metric_name == "Computational Latency"
    assert latency_stat.mean_verified_b >= 0.0
    assert latency_stat.unit == "s"


def test_simulation_avoids_outliers_and_all_models_hallucinate() -> None:
    import math

    from vrag.models import APPROVED_MODELS

    pdf_bytes = (
        b"Bang et al. (2025) evaluated 7B to 8B open-source models on HalluLens and found "
        b"baseline hallucination rates between 48% and 85%. "
        b"Kadavath et al. (2022) showed smaller models are poorly calibrated and overconfident.\f"
        b"Wang et al. (2025) proposed dynamic semantic chunking using cosine similarity thresholds "
        b"to preserve complete academic arguments across sentences. "
        b"Qian et al. (2025) introduced VeriCite to prune unentailed claims via sentence-level NLI."
    )

    pairs, stats = run_student_pdf_benchmark(
        pdf_bytes=pdf_bytes,
        pdf_name="ccs9_manuscript.pdf",
        models=APPROVED_MODELS,
        simulation_mode=True,
    )

    # 4 prompts x 3 models = 12 paired trials
    assert len(pairs) == 12
    assert len(stats) == 4

    # Quality metrics must not collapse to 0% or 100% outliers
    for stat in stats[:3]:
        assert (
            20.0 <= stat.mean_baseline_a <= 80.0
        ), f"{stat.metric_name} baseline outlier: {stat.mean_baseline_a}"
        assert (
            70.0 <= stat.mean_verified_b <= 98.0
        ), f"{stat.metric_name} verified outlier: {stat.mean_verified_b}"
        assert stat.std_delta > 0.0, f"{stat.metric_name} has zero variance"
        assert math.isfinite(stat.t_statistic), f"{stat.metric_name} t_stat is not finite"

    # Every approved model must exhibit hallucinations in Condition A that Condition B reduces
    for model in APPROVED_MODELS:
        model_pairs = [p for p in pairs if p[0].model_name == model]
        assert len(model_pairs) == 4
        total_halluc_a = sum(
            p[0].intrinsic_hallucinations + p[0].extrinsic_hallucinations for p in model_pairs
        )
        total_halluc_b = sum(
            p[1].intrinsic_hallucinations + p[1].extrinsic_hallucinations for p in model_pairs
        )
        assert total_halluc_a >= 4, f"{model} showed too few baseline hallucinations"
        assert total_halluc_b < total_halluc_a, f"{model} did not reduce hallucinations"
