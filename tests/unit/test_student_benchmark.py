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
