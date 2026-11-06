"""Deterministic NLI, citation, author & publication-year verification gate enforcing LAW 1."""

from __future__ import annotations

import functools
import re
from collections.abc import Callable
from typing import Any

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
GroundTruthFn = Callable[[str], bool]

# Matches inline chunk citations like [C1], [Chunk_2], [C1, C2]
CHUNK_CITATION_PATTERN = re.compile(r"\[(C(?:hunk_?)?\d+(?:\s*,\s*C(?:hunk_?)?\d+)*)\]")

_NAME_TOKEN = r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ\-]+"
_YEAR_TOKEN = r"(?:19|20)\d{2}[a-z]?"

_NON_AUTHOR_WORDS: frozenset[str] = frozenset(
    {
        "In",
        "On",
        "At",
        "By",
        "For",
        "From",
        "During",
        "Since",
        "After",
        "Before",
        "Between",
        "Within",
        "Under",
        "Over",
        "Table",
        "Figure",
        "Appendix",
        "Chapter",
        "Section",
        "Equation",
        "Step",
        "Phase",
        "Part",
        "Objective",
        "Condition",
        "Mode",
        "Model",
        "Page",
        "Chunk",
        "Goal",
        "Type",
        "Stage",
    }
)

_PAREN_GROUP_PATTERN = re.compile(r"\(([^()]+)\)")
_PAREN_SEGMENT_PATTERN = re.compile(
    rf"^\s*({_NAME_TOKEN})(?:\s+et\s+al\.)?"
    rf"(?:\s*(?:&|and)\s*({_NAME_TOKEN}))?,\s*({_YEAR_TOKEN})\s*$"
)
_NARRATIVE_CITATION_PATTERN = re.compile(
    rf"\b({_NAME_TOKEN})(?:\s+et\s+al\.|\s+(?:&|and)\s+({_NAME_TOKEN}))?\s*\(({_YEAR_TOKEN})\)"
)

# Splits coordinated multi-claim compound sentences while keeping ordinary lists intact
_COMPOUND_CLAUSE_SPLIT_PATTERN = re.compile(
    rf";\s+|,\s+(?:and|whereas|while|but)\s+"
    rf"(?={_NAME_TOKEN}\s+(?:\({_YEAR_TOKEN}|et\s+al\.|(?:and|&)\s+{_NAME_TOKEN}))"
)


@functools.lru_cache(maxsize=1)
def _get_cached_deberta_cross_encoder(cls: Any) -> Any:
    """Cache cross-encoder/nli-deberta-v3-small singleton on CPU per PRINCIPLES.md."""
    return cls("cross-encoder/nli-deberta-v3-small", device="cpu")


def split_atomic_claims(text: str) -> list[str]:
    """Decompose text into sentences and split coordinated multi-claim propositions."""
    sentences = split_sentences(text)
    atomic_claims: list[str] = []

    for sentence in sentences:
        parts = [p.strip() for p in _COMPOUND_CLAUSE_SPLIT_PATTERN.split(sentence) if p.strip()]
        if len(parts) <= 1:
            atomic_claims.append(sentence)
            continue

        trailing_citations = extract_chunk_ids(parts[-1])
        tag_suffix = f" [{', '.join(trailing_citations)}]" if trailing_citations else ""

        for idx, part in enumerate(parts):
            clause = part
            if idx < len(parts) - 1 and tag_suffix and not extract_chunk_ids(clause):
                clause = clause.rstrip(".!?") + tag_suffix + "."
            elif not clause.endswith((".", "!", "?")):
                clause = clause + "."
            atomic_claims.append(clause)

    return atomic_claims


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


def extract_author_year_citations(sentence: str) -> tuple[tuple[str, str], ...]:
    """Extract ordered (AuthorSurname, Year) pairs from parenthetical & narrative citations."""
    positioned_pairs: list[tuple[int, str, str]] = []

    for group_match in _PAREN_GROUP_PATTERN.finditer(sentence):
        base_pos = group_match.start()
        offset = 0
        for segment in group_match.group(1).split(";"):
            seg_match = _PAREN_SEGMENT_PATTERN.match(segment)
            if seg_match:
                year = seg_match.group(3).strip()
                for g_idx in (1, 2):
                    surname = (seg_match.group(g_idx) or "").strip()
                    if surname and surname not in _NON_AUTHOR_WORDS:
                        positioned_pairs.append((base_pos + offset + g_idx, surname, year))
            offset += len(segment) + 1

    for nar_match in _NARRATIVE_CITATION_PATTERN.finditer(sentence):
        pos = nar_match.start()
        year = nar_match.group(3).strip()
        for g_idx in (1, 2):
            surname = (nar_match.group(g_idx) or "").strip()
            if surname and surname not in _NON_AUTHOR_WORDS:
                positioned_pairs.append((pos + g_idx, surname, year))

    positioned_pairs.sort(key=lambda item: item[0])
    unique_pairs: list[tuple[str, str]] = []
    for _, surname, year in positioned_pairs:
        pair = (surname, year)
        if pair not in unique_pairs:
            unique_pairs.append(pair)
    return tuple(unique_pairs)


def extract_author_surnames(sentence: str) -> tuple[str, ...]:
    """Extract unique cited author surnames in order of appearance."""
    surnames: list[str] = []
    for surname, _year in extract_author_year_citations(sentence):
        if surname not in surnames:
            surnames.append(surname)
    return tuple(surnames)


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

    model = _get_cached_deberta_cross_encoder(CrossEncoder)
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
    ground_truth_fn: GroundTruthFn | None = None,
) -> VerificationReport:
    """Evaluate Condition A (unpruned baseline) to measure surviving hallucinations (FP)."""
    scorer = nli_scorer or default_deberta_nli_scorer
    chunk_map: dict[str, Chunk] = {c.chunk_id: c for c in retrieved_chunks}
    claims = split_atomic_claims(unverified_summary)

    if not claims:
        return VerificationReport(
            verified_summary="",
            claim_results=(),
            confusion_matrix=ConfusionMatrix(),
            atomic_factscore=0.0,
            citation_precision=0.0,
            citation_recall=0.0,
            intrinsic_hallucinations=0,
            extrinsic_hallucinations=0,
        )

    claim_results: list[ClaimVerificationResult] = []
    tp = fp = 0
    intrinsic_count = 0
    extrinsic_count = 0
    entailed_citations_count = 0
    total_citations_count = 0
    claims_with_entailed_citation = 0
    full_premise = " ".join(c.text for c in retrieved_chunks)

    for sentence in claims:
        cited_ids = extract_chunk_ids(sentence)
        author_years = extract_author_year_citations(sentence)
        authors = tuple(dict.fromkeys(a for a, _ in author_years))
        years = tuple(dict.fromkeys(y for _, y in author_years))
        valid_ids = [cid for cid in cited_ids if cid in chunk_map]
        citation_exists = len(cited_ids) > 0 and len(valid_ids) == len(cited_ids)

        total_citations_count += len(cited_ids)
        premise = " ".join(chunk_map[cid].text for cid in valid_ids) if valid_ids else full_premise
        premise_lower = premise.lower()
        author_exists = all(a.lower() in premise_lower for a in authors) if authors else True
        year_exists = all(y.lower() in premise_lower for y in years) if years else True

        clean_hypothesis = CHUNK_CITATION_PATTERN.sub("", sentence).strip()
        if premise.strip() and clean_hypothesis:
            nli_result = scorer(premise, clean_hypothesis)
        else:
            nli_result = NLIScore(label=NLILabel.NEUTRAL, entailment_prob=0.0, neutral_prob=1.0)

        nli_supported = (
            nli_result.label == NLILabel.ENTAILMENT
            and nli_result.entailment_prob >= entailment_threshold
            and author_exists
            and year_exists
        )
        is_supported = (
            ground_truth_fn(clean_hypothesis) if ground_truth_fn is not None else nli_supported
        )

        if is_supported:
            tp += 1
            if valid_ids:
                entailed_citations_count += len(valid_ids)
                claims_with_entailed_citation += 1
        else:
            fp += 1
            if nli_result.label == NLILabel.CONTRADICTION:
                intrinsic_count += 1
            else:
                extrinsic_count += 1

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
                extracted_years=years,
                year_exists=year_exists,
            )
        )

    matrix = ConfusionMatrix(tp=tp, tn=0, fp=fp, fn=0)
    factscore = tp / len(claims)
    citation_precision = (
        entailed_citations_count / total_citations_count if total_citations_count > 0 else 0.0
    )
    citation_recall = claims_with_entailed_citation / len(claims)

    return VerificationReport(
        verified_summary=unverified_summary.strip(),
        claim_results=tuple(claim_results),
        confusion_matrix=matrix,
        atomic_factscore=factscore,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        intrinsic_hallucinations=intrinsic_count,
        extrinsic_hallucinations=extrinsic_count,
    )


def verify_and_prune_summary(
    generated_summary: str,
    retrieved_chunks: list[Chunk],
    entailment_threshold: float = 0.70,
    nli_scorer: NLIScorerFn | None = None,
    ground_truth_fn: GroundTruthFn | None = None,
) -> VerificationReport:
    """Enforce LAW 1: Verify every atomic claim and deterministically prune violations."""
    scorer = nli_scorer or default_deberta_nli_scorer
    chunk_map: dict[str, Chunk] = {c.chunk_id: c for c in retrieved_chunks}
    claims = split_atomic_claims(generated_summary)

    if not claims:
        return VerificationReport(
            verified_summary="",
            claim_results=(),
            confusion_matrix=ConfusionMatrix(),
            atomic_factscore=0.0,
            citation_precision=0.0,
            citation_recall=0.0,
            intrinsic_hallucinations=0,
            extrinsic_hallucinations=0,
        )

    claim_results: list[ClaimVerificationResult] = []
    surviving_sentences: list[str] = []
    tp = tn = fp = fn = 0
    intrinsic_count = 0
    extrinsic_count = 0
    entailed_citations_count = 0
    total_surviving_citations = 0
    claims_with_entailed_citation = 0

    for sentence in claims:
        cited_ids = extract_chunk_ids(sentence)
        author_years = extract_author_year_citations(sentence)
        authors = tuple(dict.fromkeys(a for a, _ in author_years))
        years = tuple(dict.fromkeys(y for _, y in author_years))

        valid_ids = [cid for cid in cited_ids if cid in chunk_map]
        citation_exists = len(cited_ids) > 0 and len(valid_ids) == len(cited_ids)

        if citation_exists:
            combined_premise = " ".join(chunk_map[cid].text for cid in valid_ids)
        else:
            combined_premise = " ".join(c.text for c in retrieved_chunks)

        premise_lower = combined_premise.lower()
        author_exists = all(a.lower() in premise_lower for a in authors) if authors else True
        year_exists = all(y.lower() in premise_lower for y in years) if years else True

        clean_hypothesis = CHUNK_CITATION_PATTERN.sub("", sentence).strip()
        if combined_premise.strip() and clean_hypothesis:
            nli_result = scorer(combined_premise, clean_hypothesis)
        else:
            nli_result = NLIScore(label=NLILabel.NEUTRAL, entailment_prob=0.0, neutral_prob=1.0)

        is_entailed = (
            nli_result.label == NLILabel.ENTAILMENT
            and nli_result.entailment_prob >= entailment_threshold
        )

        prune_reason: str | None = None
        if not citation_exists:
            prune_reason = "missing_or_invalid_chunk_citation"
        elif not author_exists:
            prune_reason = "fabricated_author_citation"
        elif not year_exists:
            prune_reason = "fabricated_year_citation"
        elif not is_entailed:
            prune_reason = f"unentailed_claim:{nli_result.label.value}"

        survived = prune_reason is None
        ground_truth_valid = (
            ground_truth_fn(clean_hypothesis) if ground_truth_fn is not None else survived
        )

        if survived and ground_truth_valid:
            tp += 1
            surviving_sentences.append(sentence)
            total_surviving_citations += len(cited_ids)
            entailed_citations_count += len(valid_ids)
            if valid_ids:
                claims_with_entailed_citation += 1
        elif not survived and not ground_truth_valid:
            tn += 1
        elif survived and not ground_truth_valid:
            fp += 1
            surviving_sentences.append(sentence)
            total_surviving_citations += len(cited_ids)
            if nli_result.label == NLILabel.CONTRADICTION:
                intrinsic_count += 1
            else:
                extrinsic_count += 1
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
                extracted_years=years,
                year_exists=year_exists,
            )
        )

    matrix = ConfusionMatrix(tp=tp, tn=tn, fp=fp, fn=fn)
    factscore = (tp / len(surviving_sentences)) if surviving_sentences else 0.0
    citation_precision = (
        entailed_citations_count / total_surviving_citations
        if total_surviving_citations > 0
        else 0.0
    )
    claims_requiring_citation = tp + fp + fn
    citation_recall = (
        claims_with_entailed_citation / claims_requiring_citation
        if claims_requiring_citation > 0
        else 0.0
    )

    return VerificationReport(
        verified_summary=" ".join(surviving_sentences),
        claim_results=tuple(claim_results),
        confusion_matrix=matrix,
        atomic_factscore=factscore,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        intrinsic_hallucinations=intrinsic_count,
        extrinsic_hallucinations=extrinsic_count,
    )
