"""Deterministic NLI & citation existence verification gate enforcing LAW 1."""

from __future__ import annotations

import re
from collections.abc import Callable

from vrag.chunker import split_sentences
from vrag.models import (
    Chunk,
    ClaimVerificationResult,
    ConfusionMatrix,
    NLILabel,
    NLIScore,
    VerificationReport,
)

NLIScorerFn = Callable[[str, str], NLIScore]

# Matches inline chunk citations like [C1], [Chunk_2], [C1, C2]
CHUNK_CITATION_PATTERN = re.compile(r"\[(C(?:hunk_?)?\d+(?:\s*,\s*C(?:hunk_?)?\d+)*)\]")

# Matches parenthetical (Author, 2023), (Smith & Jones, 2021), (Author et al., 2021)
PAREN_AUTHOR_YEAR_PATTERN = re.compile(
    r"\(([A-Z][A-Za-z\-]+)(?:\s+et\s+al\.)?"
    r"(?:\s*(?:&|and)\s*([A-Z][A-Za-z\-]+))?,\s*\d{4}[a-z]?\)"
)
NARRATIVE_AUTHOR_YEAR_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z\-]+)(?:\s+et\s+al\.)\s*\(\d{4}[a-z]?\)"
)


def extract_chunk_ids(sentence: str) -> tuple[str, ...]:
    """Extract cited chunk IDs (e.g., ('C1', 'C2')) from a generated sentence."""
    found: list[str] = []
    for match in CHUNK_CITATION_PATTERN.finditer(sentence):
        raw_group = match.group(1)
        for part in raw_group.split(","):
            cid = part.strip()
            if cid and cid not in found:
                found.append(cid)
    return tuple(found)


def extract_author_surnames(sentence: str) -> tuple[str, ...]:
    """Extract cited author surnames from (Author, Year) or Author et al. (Year) patterns."""
    authors: list[str] = []
    for match in PAREN_AUTHOR_YEAR_PATTERN.finditer(sentence):
        for group_idx in (1, 2):
            surname = (match.group(group_idx) or "").strip()
            if surname and surname not in authors:
                authors.append(surname)
    for match in NARRATIVE_AUTHOR_YEAR_PATTERN.finditer(sentence):
        surname = match.group(1).strip()
        if surname and surname not in authors:
            authors.append(surname)
    return tuple(authors)


def default_deberta_nli_scorer(premise: str, hypothesis: str) -> NLIScore:
    """Score (premise, hypothesis) with local cross-encoder/nli-deberta-v3-small."""
    import math

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as err:
        raise ImportError(
            f"Failed to import cross-encoder / transformers: {err}. "
            "Ensure tokenizers and transformers versions are compatible "
            "(e.g., pip install 'tokenizers>=0.22.0,<=0.23.0' 'huggingface-hub>=0.34.0,<1.0' "
            "or pass an injectable nli_scorer / use Simulation Mode)."
        ) from err

    model = CrossEncoder("cross-encoder/nli-deberta-v3-small", device="cpu")
    logits = model.predict([(premise, hypothesis)])[0]
    # cross-encoder/nli-deberta-v3-small label mapping: 0=contradiction, 1=entailment, 2=neutral
    max_logit = float(max(logits))
    exps = [math.exp(float(x) - max_logit) for x in logits]
    total = sum(exps)
    probs = [e / total for e in exps]

    contradiction_prob, entailment_prob, neutral_prob = probs[0], probs[1], probs[2]
    if entailment_prob >= contradiction_prob and entailment_prob >= neutral_prob:
        label = NLILabel.ENTAILMENT
    elif contradiction_prob >= neutral_prob:
        label = NLILabel.CONTRADICTION
    else:
        label = NLILabel.NEUTRAL

    return NLIScore(
        label=label,
        entailment_prob=entailment_prob,
        neutral_prob=neutral_prob,
        contradiction_prob=contradiction_prob,
    )


def evaluate_unverified_summary(
    unverified_summary: str,
    retrieved_chunks: list[Chunk],
    entailment_threshold: float = 0.70,
    nli_scorer: NLIScorerFn | None = None,
) -> VerificationReport:
    """Evaluate Condition A (unpruned baseline) to measure surviving hallucinations (FP)."""
    scorer = nli_scorer or default_deberta_nli_scorer
    chunk_map: dict[str, Chunk] = {c.chunk_id: c for c in retrieved_chunks}
    sentences = split_sentences(unverified_summary)

    if not sentences:
        return VerificationReport(
            verified_summary="",
            claim_results=(),
            confusion_matrix=ConfusionMatrix(),
            atomic_factscore=0.0,
            citation_precision=0.0,
            citation_recall=0.0,
        )

    claim_results: list[ClaimVerificationResult] = []
    tp = fp = 0
    valid_citations_count = 0
    total_citations_count = 0
    cited_valid_chunk_ids: set[str] = set()
    full_premise = " ".join(c.text for c in retrieved_chunks)

    for sentence in sentences:
        cited_ids = extract_chunk_ids(sentence)
        authors = extract_author_surnames(sentence)
        valid_ids = [cid for cid in cited_ids if cid in chunk_map]
        citation_exists = len(cited_ids) > 0 and len(valid_ids) == len(cited_ids)

        if cited_ids:
            total_citations_count += len(cited_ids)
            valid_citations_count += len(valid_ids)
            cited_valid_chunk_ids.update(valid_ids)

        premise = " ".join(chunk_map[cid].text for cid in valid_ids) if valid_ids else full_premise
        author_exists = True
        if authors:
            premise_lower = premise.lower()
            author_exists = all(a.lower() in premise_lower for a in authors)

        clean_hypothesis = CHUNK_CITATION_PATTERN.sub("", sentence).strip()
        if premise.strip() and clean_hypothesis:
            nli_result = scorer(premise, clean_hypothesis)
        else:
            nli_result = NLIScore(label=NLILabel.NEUTRAL, entailment_prob=0.0, neutral_prob=1.0)

        is_entailed = (
            nli_result.label == NLILabel.ENTAILMENT
            and nli_result.entailment_prob >= entailment_threshold
            and author_exists
        )

        # In Condition A, all sentences survive unpruned
        if is_entailed:
            tp += 1
        else:
            fp += 1

        claim_results.append(
            ClaimVerificationResult(
                sentence=sentence,
                cited_chunk_ids=cited_ids,
                extracted_authors=authors,
                nli_label=nli_result.label,
                entailment_score=nli_result.entailment_prob,
                citation_exists=citation_exists,
                author_exists=author_exists,
                survived=True,
                prune_reason=None,
            )
        )

    matrix = ConfusionMatrix(tp=tp, tn=0, fp=fp, fn=0)
    factscore = tp / len(sentences)
    citation_precision = (
        valid_citations_count / total_citations_count if total_citations_count > 0 else 0.0
    )
    citation_recall = (
        len(cited_valid_chunk_ids) / len(retrieved_chunks) if retrieved_chunks else 0.0
    )

    return VerificationReport(
        verified_summary=unverified_summary.strip(),
        claim_results=tuple(claim_results),
        confusion_matrix=matrix,
        atomic_factscore=factscore,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
    )


def verify_and_prune_summary(
    generated_summary: str,
    retrieved_chunks: list[Chunk],
    entailment_threshold: float = 0.70,
    nli_scorer: NLIScorerFn | None = None,
) -> VerificationReport:
    """Enforce LAW 1: Verify every atomic sentence and deterministically prune violations."""
    scorer = nli_scorer or default_deberta_nli_scorer
    chunk_map: dict[str, Chunk] = {c.chunk_id: c for c in retrieved_chunks}
    sentences = split_sentences(generated_summary)

    if not sentences:
        return VerificationReport(
            verified_summary="",
            claim_results=(),
            confusion_matrix=ConfusionMatrix(),
            atomic_factscore=0.0,
            citation_precision=0.0,
            citation_recall=0.0,
        )

    claim_results: list[ClaimVerificationResult] = []
    surviving_sentences: list[str] = []
    tp = tn = fp = fn = 0
    valid_citations_count = 0
    total_citations_count = 0
    cited_valid_chunk_ids: set[str] = set()

    for sentence in sentences:
        cited_ids = extract_chunk_ids(sentence)
        authors = extract_author_surnames(sentence)

        total_citations_count += max(len(cited_ids), 1)
        valid_ids = [cid for cid in cited_ids if cid in chunk_map]
        citation_exists = len(cited_ids) > 0 and len(valid_ids) == len(cited_ids)

        if citation_exists:
            combined_premise = " ".join(chunk_map[cid].text for cid in valid_ids)
        else:
            combined_premise = " ".join(c.text for c in retrieved_chunks)

        author_exists = True
        if authors:
            premise_lower = combined_premise.lower()
            author_exists = all(author.lower() in premise_lower for author in authors)

        clean_hypothesis = CHUNK_CITATION_PATTERN.sub("", sentence).strip()
        if combined_premise.strip() and clean_hypothesis:
            nli_result = scorer(combined_premise, clean_hypothesis)
        else:
            nli_result = NLIScore(label=NLILabel.NEUTRAL, entailment_prob=0.0, neutral_prob=1.0)

        is_entailed = (
            nli_result.label == NLILabel.ENTAILMENT
            and nli_result.entailment_prob >= entailment_threshold
        )
        ground_truth_valid = citation_exists and author_exists and is_entailed

        prune_reason: str | None = None
        if not citation_exists:
            prune_reason = "missing_or_invalid_chunk_citation"
        elif not author_exists:
            prune_reason = "fabricated_author_citation"
        elif not is_entailed:
            prune_reason = f"unentailed_claim:{nli_result.label.value}"

        survived = prune_reason is None

        if survived and ground_truth_valid:
            tp += 1
            valid_citations_count += len(valid_ids)
            cited_valid_chunk_ids.update(valid_ids)
            surviving_sentences.append(sentence)
        elif not survived and not ground_truth_valid:
            tn += 1
        elif survived and not ground_truth_valid:
            fp += 1
            surviving_sentences.append(sentence)
        else:
            fn += 1

        claim_results.append(
            ClaimVerificationResult(
                sentence=sentence,
                cited_chunk_ids=cited_ids,
                extracted_authors=authors,
                nli_label=nli_result.label,
                entailment_score=nli_result.entailment_prob,
                citation_exists=citation_exists,
                author_exists=author_exists,
                survived=survived,
                prune_reason=prune_reason,
            )
        )

    matrix = ConfusionMatrix(tp=tp, tn=tn, fp=fp, fn=fn)
    factscore = (tp / len(surviving_sentences)) if surviving_sentences else 1.0
    citation_precision = (
        valid_citations_count / total_citations_count if total_citations_count > 0 else 0.0
    )
    citation_recall = (
        len(cited_valid_chunk_ids) / len(retrieved_chunks) if retrieved_chunks else 0.0
    )

    return VerificationReport(
        verified_summary=" ".join(surviving_sentences),
        claim_results=tuple(claim_results),
        confusion_matrix=matrix,
        atomic_factscore=factscore,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
    )
