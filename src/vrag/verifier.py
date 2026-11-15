"""Deterministic NLI, citation, author & publication-year verification gate enforcing LAW 1."""

from __future__ import annotations

import functools
import math
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

# --- Citation & Claim Extraction ---

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

# Splits coordinated multi-claim compound sentences while keeping ordinary lists intact.
_COMPOUND_CLAUSE_SPLIT_PATTERN = re.compile(
    rf";\s+|,\s+(?:and|whereas|while|but)\s+"
    rf"(?={_NAME_TOKEN}\s+(?:\({_YEAR_TOKEN}|et\s+al\.|(?:and|&)\s+{_NAME_TOKEN}))"
)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "by",
        "with",
        "that",
        "this",
        "is",
        "are",
        "was",
        "were",
        "via",
        "into",
        "from",
        "across",
        "et",
        "al",
        "their",
        "how",
        "it",
        "as",
        "at",
        "be",
    }
)
_WORD_RE = re.compile(r"[a-zà-öø-ÿ0-9]+")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def content_tokens(text: str) -> set[str]:
    """Extract lowercase content words excluding short tokens and common stopwords."""
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1}


def numeric_tokens(text: str) -> set[str]:
    """Extract numeric literals so subtle figure distortions fail oracle matching."""
    return set(_NUMBER_RE.findall(text))


def split_atomic_claims(text: str) -> list[str]:
    """Decompose text into sentences and split coordinated multi-claim propositions."""
    atomic_claims: list[str] = []
    for sentence in split_sentences(text):
        parts = [p.strip() for p in _COMPOUND_CLAUSE_SPLIT_PATTERN.split(sentence) if p.strip()]
        if len(parts) <= 1:
            atomic_claims.append(sentence)
            continue

        # Propagate trailing chunk tags (e.g. [C1]) to earlier split clauses lacking their own tag.
        trailing = extract_chunk_ids(parts[-1])
        tag_suffix = f" [{', '.join(trailing)}]" if trailing else ""
        for idx, clause in enumerate(parts):
            if idx < len(parts) - 1 and tag_suffix and not extract_chunk_ids(clause):
                clause = clause.rstrip(".!?") + tag_suffix + "."
            elif not clause.endswith((".", "!", "?")):
                clause += "."
            atomic_claims.append(clause)
    return atomic_claims


def extract_chunk_ids(sentence: str) -> tuple[str, ...]:
    """Extract unique cited chunk IDs (e.g. ('C1', 'C2')) in order of appearance."""
    found: dict[str, None] = {}
    for match in CHUNK_CITATION_PATTERN.finditer(sentence):
        for part in match.group(1).split(","):
            if cid := part.strip():
                found.setdefault(cid, None)
    return tuple(found)


def _append_match_authors(
    target: list[tuple[int, str, str]],
    match: re.Match[str],
    base_pos: int,
) -> None:
    year = match.group(3).strip()
    for g_idx in (1, 2):
        surname = (match.group(g_idx) or "").strip()
        if surname and surname not in _NON_AUTHOR_WORDS:
            target.append((base_pos + g_idx, surname, year))


def extract_author_year_citations(sentence: str) -> tuple[tuple[str, str], ...]:
    """Extract ordered (AuthorSurname, Year) pairs from parenthetical & narrative citations."""
    positioned: list[tuple[int, str, str]] = []

    for group_match in _PAREN_GROUP_PATTERN.finditer(sentence):
        offset = group_match.start()
        for segment in group_match.group(1).split(";"):
            if seg_match := _PAREN_SEGMENT_PATTERN.match(segment):
                _append_match_authors(positioned, seg_match, offset)
            offset += len(segment) + 1

    for nar_match in _NARRATIVE_CITATION_PATTERN.finditer(sentence):
        _append_match_authors(positioned, nar_match, nar_match.start())

    positioned.sort(key=lambda item: item[0])
    return tuple(dict.fromkeys((surname, year) for _, surname, year in positioned))


def extract_author_surnames(sentence: str) -> tuple[str, ...]:
    """Extract unique cited author surnames in order of appearance."""
    return tuple(dict.fromkeys(surname for surname, _ in extract_author_year_citations(sentence)))


# --- Local DeBERTa NLI Scorer ---


@functools.lru_cache(maxsize=1)
def _get_cached_deberta_cross_encoder(cls: Any) -> Any:
    """Cache cross-encoder/nli-deberta-v3-small singleton on CPU per PRINCIPLES.md."""
    return cls("cross-encoder/nli-deberta-v3-small", device="cpu")


def default_deberta_nli_scorer(premise: str, hypothesis: str) -> NLIScore:
    """Score (premise, hypothesis) with local cross-encoder/nli-deberta-v3-small."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as err:
        raise ImportError(
            f"Failed to import cross-encoder / transformers: {err}. "
            "Ensure tokenizers and transformers versions are compatible "
            "(e.g., pip install 'tokenizers>=0.22.0,<=0.23.0' 'huggingface-hub>=0.34.0,<1.0' "
            "or pass an injectable nli_scorer / use Simulation Mode)."
        ) from err

    logits = _get_cached_deberta_cross_encoder(CrossEncoder).predict([(premise, hypothesis)])[0]
    # cross-encoder/nli-deberta-v3-small label order: 0=contradiction, 1=entailment, 2=neutral
    max_logit = float(max(logits))
    exps = [math.exp(float(x) - max_logit) for x in logits]
    total = sum(exps)
    contra_p, entail_p, neutral_p = exps[0] / total, exps[1] / total, exps[2] / total

    if entail_p >= contra_p and entail_p >= neutral_p:
        label = NLILabel.ENTAILMENT
    elif contra_p >= neutral_p:
        label = NLILabel.CONTRADICTION
    else:
        label = NLILabel.NEUTRAL

    return NLIScore(
        label=label,
        entailment_prob=entail_p,
        neutral_prob=neutral_p,
        contradiction_prob=contra_p,
    )


# --- Deterministic Verification Gate (LAW 1) ---


def _prune_reason_for_claim(
    *,
    citation_exists: bool,
    author_exists: bool,
    year_exists: bool,
    is_entailed: bool,
    nli_label: NLILabel,
) -> str | None:
    if not citation_exists:
        return "missing_or_invalid_chunk_citation"
    if not author_exists:
        return "fabricated_author_citation"
    if not year_exists:
        return "fabricated_year_citation"
    if not is_entailed:
        return f"unentailed_claim:{nli_label.value}"
    return None


def _verify_claims(
    summary: str,
    retrieved_chunks: list[Chunk],
    *,
    entailment_threshold: float,
    nli_scorer: NLIScorerFn | None,
    ground_truth_fn: GroundTruthFn | None,
    prune: bool,
) -> VerificationReport:
    """Shared verification pass for Condition A (unpruned) and Condition B (gated pruning)."""
    claims = split_atomic_claims(summary)
    if not claims:
        return VerificationReport(
            verified_summary="",
            claim_results=(),
            confusion_matrix=ConfusionMatrix(),
            atomic_factscore=0.0,
            citation_precision=0.0,
            citation_recall=0.0,
        )

    scorer = nli_scorer or default_deberta_nli_scorer
    chunk_map = {c.chunk_id: c for c in retrieved_chunks}
    full_premise = " ".join(c.text for c in retrieved_chunks)

    claim_results: list[ClaimVerificationResult] = []
    surviving: list[str] = []
    tp = tn = fp = fn = 0
    intrinsic = extrinsic = entailed_cites = total_cites = claims_with_entailed_cite = 0

    for sentence in claims:
        cited_ids = extract_chunk_ids(sentence)
        author_years = extract_author_year_citations(sentence)
        authors = tuple(dict.fromkeys(a for a, _ in author_years))
        years = tuple(dict.fromkeys(y for _, y in author_years))
        valid_ids = [cid for cid in cited_ids if cid in chunk_map]
        citation_exists = bool(cited_ids) and len(valid_ids) == len(cited_ids)

        use_valid = citation_exists if prune else bool(valid_ids)
        premise = " ".join(chunk_map[cid].text for cid in valid_ids) if use_valid else full_premise
        premise_lower = premise.lower()
        author_exists = all(a.lower() in premise_lower for a in authors)
        year_exists = all(y.lower() in premise_lower for y in years)

        hypothesis = CHUNK_CITATION_PATTERN.sub("", sentence).strip()
        nli_result = (
            scorer(premise, hypothesis)
            if premise.strip() and hypothesis
            else NLIScore(label=NLILabel.NEUTRAL, entailment_prob=0.0, neutral_prob=1.0)
        )
        is_entailed = (
            nli_result.label == NLILabel.ENTAILMENT
            and nli_result.entailment_prob >= entailment_threshold
        )

        if prune:
            prune_reason = _prune_reason_for_claim(
                citation_exists=citation_exists,
                author_exists=author_exists,
                year_exists=year_exists,
                is_entailed=is_entailed,
                nli_label=nli_result.label,
            )
            survived = prune_reason is None
            gt_valid = ground_truth_fn(hypothesis) if ground_truth_fn is not None else survived
            if survived:
                surviving.append(sentence)
                total_cites += len(cited_ids)
                if gt_valid:
                    tp += 1
                    entailed_cites += len(valid_ids)
                    if valid_ids:
                        claims_with_entailed_cite += 1
                else:
                    fp += 1
                    if nli_result.label == NLILabel.CONTRADICTION:
                        intrinsic += 1
                    else:
                        extrinsic += 1
            elif gt_valid:
                fn += 1
            else:
                tn += 1
        else:
            prune_reason = None
            survived = True
            total_cites += len(cited_ids)
            nli_supported = is_entailed and author_exists and year_exists
            is_supported = (
                ground_truth_fn(hypothesis) if ground_truth_fn is not None else nli_supported
            )
            if is_supported:
                tp += 1
                if valid_ids:
                    entailed_cites += len(valid_ids)
                    claims_with_entailed_cite += 1
            else:
                fp += 1
                if nli_result.label == NLILabel.CONTRADICTION:
                    intrinsic += 1
                else:
                    extrinsic += 1

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

    survivor_count = len(surviving) if prune else len(claims)
    claims_req = (tp + fp + fn) if prune else len(claims)
    return VerificationReport(
        verified_summary=" ".join(surviving) if prune else summary.strip(),
        claim_results=tuple(claim_results),
        confusion_matrix=ConfusionMatrix(tp=tp, tn=tn, fp=fp, fn=fn),
        atomic_factscore=(tp / survivor_count) if survivor_count > 0 else 0.0,
        citation_precision=(entailed_cites / total_cites) if total_cites > 0 else 0.0,
        citation_recall=(claims_with_entailed_cite / claims_req) if claims_req > 0 else 0.0,
        intrinsic_hallucinations=intrinsic,
        extrinsic_hallucinations=extrinsic,
    )


def evaluate_unverified_summary(
    unverified_summary: str,
    retrieved_chunks: list[Chunk],
    entailment_threshold: float = 0.70,
    nli_scorer: NLIScorerFn | None = None,
    ground_truth_fn: GroundTruthFn | None = None,
) -> VerificationReport:
    """Evaluate Condition A (unpruned baseline) to measure surviving hallucinations (FP)."""
    return _verify_claims(
        unverified_summary,
        retrieved_chunks,
        entailment_threshold=entailment_threshold,
        nli_scorer=nli_scorer,
        ground_truth_fn=ground_truth_fn,
        prune=False,
    )


def verify_and_prune_summary(
    generated_summary: str,
    retrieved_chunks: list[Chunk],
    entailment_threshold: float = 0.70,
    nli_scorer: NLIScorerFn | None = None,
    ground_truth_fn: GroundTruthFn | None = None,
) -> VerificationReport:
    """Enforce LAW 1: Verify every atomic claim and deterministically prune violations."""
    return _verify_claims(
        generated_summary,
        retrieved_chunks,
        entailment_threshold=entailment_threshold,
        nli_scorer=nli_scorer,
        ground_truth_fn=ground_truth_fn,
        prune=True,
    )
