"""Unit tests for verifier citation regex extraction, NLI pruning, and baseline scoring."""

from __future__ import annotations

import pytest

from vrag.models import Chunk, NLILabel, NLIScore
from vrag.verifier import (
    evaluate_unverified_summary,
    extract_author_surnames,
    extract_author_year_citations,
    extract_chunk_ids,
    split_atomic_claims,
    verify_and_prune_summary,
)


def test_extract_chunk_ids_and_author_surnames_handles_compound_citations() -> None:
    sentence = (
        "Prior studies (Smith & Jones, 2021) and Vaswani et al. (2017) "
        "confirm attention scaling [C1, C2]."
    )
    assert extract_chunk_ids(sentence) == ("C1", "C2")
    assert extract_author_surnames(sentence) == ("Smith", "Jones", "Vaswani")
    assert extract_author_year_citations(sentence) == (
        ("Smith", "2021"),
        ("Jones", "2021"),
        ("Vaswani", "2017"),
    )


def test_extract_narrative_and_semicolon_citations_and_atomic_decomposition() -> None:
    narrative = (
        "Quitaleg and Ramos (2025) evaluated 120 modules [C1]; "
        "Paningbatan (2024) claimed cloud GPUs eliminate errors [C1]."
    )
    claims = split_atomic_claims(narrative)
    assert len(claims) == 2
    assert extract_author_year_citations(claims[0]) == (("Quitaleg", "2025"), ("Ramos", "2025"))
    assert extract_author_year_citations(claims[1]) == (("Paningbatan", "2024"),)

    multi_paren = "Recent surveys (Chuang et al., 2024; Tam et al., 2023) confirm drift [C1]."
    assert extract_author_year_citations(multi_paren) == (("Chuang", "2024"), ("Tam", "2023"))


def test_evaluate_unverified_summary_records_false_positives_for_condition_a() -> None:
    chunks = [
        Chunk(
            chunk_id="C1",
            text="Vaswani et al. (2017) introduced the Transformer architecture.",
            page_number=1,
        )
    ]
    # Condition A output contains 1 entailed sentence and 1 hallucinated sentence (both unpruned)
    baseline_output = (
        "Vaswani et al. (2017) introduced the Transformer architecture. "
        "Transformers completely eliminate matrix multiplication on GPUs."
    )

    def fake_nli(premise: str, hypothesis: str) -> NLIScore:
        if "eliminate matrix multiplication" in hypothesis:
            return NLIScore(
                label=NLILabel.CONTRADICTION, entailment_prob=0.04, contradiction_prob=0.92
            )
        return NLIScore(label=NLILabel.ENTAILMENT, entailment_prob=0.95)

    report = evaluate_unverified_summary(
        unverified_summary=baseline_output,
        retrieved_chunks=chunks,
        entailment_threshold=0.70,
        nli_scorer=fake_nli,
    )

    # In Condition A, nothing is pruned: verified_summary equals unverified_summary
    assert report.verified_summary == baseline_output
    # 1 TP (entailed) and 1 FP (hallucination that survived into baseline output)
    assert report.confusion_matrix.tp == 1
    assert report.confusion_matrix.fp == 1
    assert report.confusion_matrix.tn == 0
    assert report.atomic_factscore == 0.5
    assert report.confusion_matrix.entailment_consistency == 0.5


def test_verify_and_prune_summary_handles_empty_input() -> None:
    report = verify_and_prune_summary(
        generated_summary="   ",
        retrieved_chunks=[Chunk(chunk_id="C1", text="Context text.", page_number=1)],
    )
    assert report.verified_summary == ""
    assert report.claim_results == ()
    assert report.confusion_matrix.total == 0


def test_deberta_nli_scorer_import_error_provides_actionable_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    from vrag.verifier import default_deberta_nli_scorer

    orig_import = builtins.__import__

    def mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if "sentence_transformers" in name:
            raise ImportError(
                "tokenizers>=0.22.0,<=0.23.0 is required for a normal functioning of this module, "
                "but found tokenizers==0.21.1."
            )
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(
        ImportError,
        match="Ensure tokenizers and transformers versions are compatible",
    ):
        default_deberta_nli_scorer("premise", "hypothesis")
