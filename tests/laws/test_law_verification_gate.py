"""Domain Law Invariant Tests enforcing LAWS.md."""

from __future__ import annotations

import pytest

from vrag.evaluator import run_paired_significance_test, verify_experimental_parity
from vrag.generator import OllamaGenerator, validate_offline_ollama_url
from vrag.models import (
    Chunk,
    ConfusionMatrix,
    NLILabel,
    NLIScore,
    PipelineMode,
    StageLatencies,
    TrialTelemetry,
)
from vrag.verifier import verify_and_prune_summary


@pytest.mark.law
def test_law1_prunes_unentailed_and_fabricated_citations_and_logs_confusion_matrix() -> None:
    """LAW 1: Condition B must deterministically prune unentailed or fabricated-citation claims."""
    retrieved_chunks = [
        Chunk(
            chunk_id="C1",
            text=(
                "Vaswani et al. (2017) introduced the Transformer architecture "
                "based solely on attention mechanisms."
            ),
            page_number=1,
        ),
        Chunk(
            chunk_id="C2",
            text=(
                "Lewis et al. (2020) proposed Retrieval-Augmented Generation (RAG) "
                "to ground sequence-to-sequence models on non-parametric memory."
            ),
            page_number=2,
        ),
    ]

    # Draft summary with 4 atomic sentences:
    # 1. Valid entailed claim + valid [C1] + valid author (Vaswani) -> SURVIVES (TP)
    # 2. Fabricated author (FakeAuthor, 2024) not in [C1] -> PRUNED (TN)
    # 3. Unentailed / Contradicted claim against [C2] -> PRUNED (TN)
    # 4. Missing / non-existent chunk citation [C99] -> PRUNED (TN)
    draft_summary = (
        "Vaswani et al. (2017) proposed the Transformer using attention mechanisms [C1]. "
        "According to (FakeAuthor, 2024), Transformers require recurrent layers [C1]. "
        "Lewis et al. (2020) proved that retrieval degrades factual accuracy [C2]. "
        "Quantum annealers replace vector databases in standard RAG [C99]."
    )

    def deterministic_nli_scorer(premise: str, hypothesis: str) -> NLIScore:
        if "degrades factual accuracy" in hypothesis:
            return NLIScore(
                label=NLILabel.CONTRADICTION,
                entailment_prob=0.05,
                contradiction_prob=0.90,
                neutral_prob=0.05,
            )
        return NLIScore(
            label=NLILabel.ENTAILMENT,
            entailment_prob=0.94,
            contradiction_prob=0.02,
            neutral_prob=0.04,
        )

    report = verify_and_prune_summary(
        generated_summary=draft_summary,
        retrieved_chunks=retrieved_chunks,
        entailment_threshold=0.70,
        nli_scorer=deterministic_nli_scorer,
    )

    # Only sentence 1 is allowed to survive in the final Condition B summary
    assert (
        report.verified_summary
        == "Vaswani et al. (2017) proposed the Transformer using attention mechanisms [C1]."
    )
    assert "FakeAuthor" not in report.verified_summary
    assert "degrades factual accuracy" not in report.verified_summary
    assert "Quantum annealers" not in report.verified_summary

    # Confusion matrix must log 1 TP (retained valid claim) and 3 TN (pruned invalid claims), 0 FP
    assert report.confusion_matrix == ConfusionMatrix(tp=1, tn=3, fp=0, fn=0)
    assert report.confusion_matrix.entailment_consistency == 1.0
    assert report.atomic_factscore == 1.0

    # Verify individual audit reasons
    reasons = [r.prune_reason for r in report.claim_results]
    assert reasons == [
        None,
        "fabricated_author_citation",
        "unentailed_claim:contradiction",
        "missing_or_invalid_chunk_citation",
    ]


@pytest.mark.law
def test_law2_strict_offline_loopback_enforcement() -> None:
    """LAW 2: External or paid LLM API endpoints must be rejected immediately."""
    assert validate_offline_ollama_url("http://localhost:11434") == "http://localhost:11434"
    assert validate_offline_ollama_url("http://127.0.0.1:11434/") == "http://127.0.0.1:11434"

    with pytest.raises(ValueError, match="LAW 2 violation"):
        validate_offline_ollama_url("https://api.openai.com/v1")

    with pytest.raises(ValueError, match="LAW 2 violation"):
        OllamaGenerator(base_url="https://remote-llm.example.com")


@pytest.mark.law
def test_law3_experimental_parity_enforcement() -> None:
    """LAW 3: Condition A and Condition B must share identical PDF SHA-256, prompt, and model."""
    base_latencies = StageLatencies(t_chunk=0.1, t_retrieve=0.1, t_generate=1.2, t_verify=0.0)
    ver_latencies = StageLatencies(t_chunk=0.2, t_retrieve=0.1, t_generate=1.3, t_verify=0.4)

    trial_a = TrialTelemetry(
        trial_id="t1",
        pdf_sha256="abc123sha",
        prompt="Summarize the methodology.",
        model_name="llama3.1:8b-instruct-q4_K_M",
        mode=PipelineMode.BASELINE_A,
        latencies=base_latencies,
        atomic_factscore=0.65,
        entailment_consistency=0.65,
        citation_precision=0.0,
        citation_recall=0.0,
        confusion_matrix=ConfusionMatrix(),
        peak_ram_mb=4200.0,
        output_summary="Unverified baseline text.",
    )
    trial_b = TrialTelemetry(
        trial_id="t1",
        pdf_sha256="abc123sha",
        prompt="Summarize the methodology.",
        model_name="llama3.1:8b-instruct-q4_K_M",
        mode=PipelineMode.VERIFIED_B,
        latencies=ver_latencies,
        atomic_factscore=1.0,
        entailment_consistency=1.0,
        citation_precision=1.0,
        citation_recall=0.5,
        confusion_matrix=ConfusionMatrix(tp=2, tn=1, fp=0, fn=0),
        peak_ram_mb=4350.0,
        output_summary="Verified summary [C1].",
    )

    verify_experimental_parity(trial_a, trial_b)

    mismatched_pdf_b = TrialTelemetry(**{**trial_b.__dict__, "pdf_sha256": "different_sha"})
    with pytest.raises(ValueError, match="LAW 3 violation: PDF SHA-256 mismatch"):
        verify_experimental_parity(trial_a, mismatched_pdf_b)


@pytest.mark.law
def test_law4_sequential_single_model_unload() -> None:
    """LAW 4: Transitioning between 7B-8B models must unload the previous model first."""
    gen = OllamaGenerator()
    gen.ensure_single_model_loaded("llama3.1:8b-instruct-q4_K_M")
    assert gen.active_model == "llama3.1:8b-instruct-q4_K_M"
    assert gen.unload_history == []

    # Switching to Mistral must automatically unload Llama 3.1
    gen.ensure_single_model_loaded("mistral:7b-instruct-v0.3-q4_K_M")
    assert gen.active_model == "mistral:7b-instruct-v0.3-q4_K_M"
    assert gen.unload_history == ["llama3.1:8b-instruct-q4_K_M"]


@pytest.mark.law
def test_law5_telemetry_and_paired_ttest_completeness() -> None:
    """LAW 5: Latencies must be non-negative and paired t-tests require valid paired arrays."""
    with pytest.raises(ValueError, match="LAW 5 violation"):
        StageLatencies(t_chunk=-0.01, t_retrieve=0.1, t_generate=1.0, t_verify=0.0)

    stat_res = run_paired_significance_test(
        metric_name="Atomic FActScore",
        baseline_scores=[0.60, 0.62, 0.58, 0.64, 0.61],
        verified_scores=[0.92, 0.95, 0.90, 0.96, 0.94],
    )
    assert stat_res.mean_delta > 0.30
    assert 0.0 <= stat_res.t_pvalue < 0.001
