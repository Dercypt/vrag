"""Comparative evaluation & statistical significance harness enforcing LAW 3, LAW 4, and LAW 5."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import resource
import sys
import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vrag.chunker import (
    EmbedderFn,
    chunk_pages_fixed,
    chunk_pages_semantic,
    extract_pdf_pages,
    l2_normalize,
    split_sentences,
)
from vrag.generator import OllamaGenerator
from vrag.models import (
    APPROVED_MODELS,
    DEFAULT_STUDENT_BENCHMARK_PROMPTS,
    NLILabel,
    NLIScore,
    PDFPage,
    PipelineMode,
    StageLatencies,
    TrialTelemetry,
)
from vrag.retriever import LocalVectorRetriever
from vrag.verifier import (
    GroundTruthFn,
    NLIScorerFn,
    content_tokens,
    evaluate_unverified_summary,
    extract_author_year_citations,
    numeric_tokens,
    verify_and_prune_summary,
)

# --- Hardware Memory & Experimental Parity ---


@dataclass(frozen=True)
class PairedStatisticalResult:
    """Paired t-test, Wilcoxon, and descriptive statistics across N comparative trials."""

    metric_name: str
    mean_baseline_a: float
    mean_verified_b: float
    mean_delta: float
    t_statistic: float
    t_pvalue: float
    wilcoxon_pvalue: float | None = None
    unit: str = "%"
    std_baseline_a: float = 0.0
    std_verified_b: float = 0.0
    std_delta: float = 0.0
    pct_error_reduction: float = 0.0
    normality_pvalue: float | None = None


def get_peak_ram_mb() -> float:
    """Return process RSS + local Ollama runner RSS memory usage in megabytes (LAW 4)."""
    try:
        import psutil  # type: ignore[import-untyped]

        current_pid = os.getpid()
        total_rss = float(psutil.Process(current_pid).memory_info().rss)
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                pname = str(proc.info.get("name") or "").lower()
                if proc.info.get("pid") != current_pid and "ollama" in pname:
                    total_rss += float(proc.memory_info().rss)
            except Exception:
                continue
        return total_rss / (1024.0 * 1024.0)
    except Exception:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports ru_maxrss in bytes; Linux reports kilobytes.
        divisor = (1024.0 * 1024.0) if sys.platform == "darwin" else 1024.0
        return float(usage) / divisor


def verify_experimental_parity(
    baseline_trial: TrialTelemetry,
    verified_trial: TrialTelemetry,
) -> None:
    """Enforce LAW 3: Ensure Condition A and Condition B share identical experimental inputs."""
    if baseline_trial.mode != PipelineMode.BASELINE_A:
        raise ValueError("First trial in pair must be PipelineMode.BASELINE_A")
    if verified_trial.mode != PipelineMode.VERIFIED_B:
        raise ValueError("Second trial in pair must be PipelineMode.VERIFIED_B")
    if baseline_trial.pdf_sha256 != verified_trial.pdf_sha256:
        raise ValueError("LAW 3 violation: PDF SHA-256 mismatch between Baseline and Verified runs")
    if baseline_trial.prompt != verified_trial.prompt:
        raise ValueError("LAW 3 violation: Prompt mismatch between Baseline and Verified runs")
    if baseline_trial.model_name != verified_trial.model_name:
        raise ValueError("LAW 3 violation: Model tag mismatch between Baseline and Verified runs")


def compute_pdf_sha256(content: bytes) -> str:
    """Compute deterministic SHA-256 digest of source PDF bytes."""
    return hashlib.sha256(content).hexdigest()


# --- Paired Statistical Significance (LAW 5) ---


def _sample_std(values: list[float]) -> float:
    """Compute sample standard deviation (ddof=1) for n >= 2."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_val = sum(values) / n
    return math.sqrt(max(0.0, sum((x - mean_val) ** 2 for x in values) / (n - 1)))


def run_paired_significance_test(
    metric_name: str,
    baseline_scores: list[float],
    verified_scores: list[float],
    unit: str = "%",
) -> PairedStatisticalResult:
    """Enforce LAW 5: Compute descriptive stats, normality, paired t-test & Wilcoxon p-value."""
    if len(baseline_scores) != len(verified_scores):
        raise ValueError("LAW 5 violation: Paired sample arrays must have identical length")
    if len(baseline_scores) < 2:
        raise ValueError("LAW 5 violation: At least 2 paired trials required for ttest_rel")

    from scipy import stats  # type: ignore[import-untyped]

    n = len(baseline_scores)
    mean_a = sum(baseline_scores) / n
    mean_b = sum(verified_scores) / n
    diffs = [b - a for a, b in zip(baseline_scores, verified_scores)]

    if unit == "%":
        scale_max = 1.0 if max(baseline_scores + verified_scores) <= 1.0 else 100.0
        err_a = max(0.0, scale_max - mean_a)
        err_b = max(0.0, scale_max - mean_b)
        pct_err_red = ((err_a - err_b) / err_a) * 100.0 if err_a > 1e-12 else 0.0
    else:
        pct_err_red = ((mean_a - mean_b) / mean_a) * 100.0 if mean_a > 1e-12 else 0.0

    zero_var = all(abs(d - diffs[0]) < 1e-12 for d in diffs)
    normality_pval: float | None = None
    if len(diffs) >= 3 and not zero_var:
        try:
            normality_pval = float(stats.shapiro(diffs).pvalue)
        except Exception:
            normality_pval = None

    if zero_var:
        t_stat = math.copysign(float("inf"), diffs[0]) if diffs[0] != 0.0 else 0.0
        t_pval = 0.0 if diffs[0] != 0.0 else 1.0
    else:
        t_res = stats.ttest_rel(verified_scores, baseline_scores)
        t_stat = 0.0 if math.isnan(float(t_res.statistic)) else float(t_res.statistic)
        t_pval = 1.0 if math.isnan(float(t_res.pvalue)) else float(t_res.pvalue)

    wilcoxon_pval: float | None = None
    if any(d != 0.0 for d in diffs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                wilcoxon_pval = float(stats.wilcoxon(verified_scores, baseline_scores).pvalue)
        except ValueError:
            wilcoxon_pval = None

    return PairedStatisticalResult(
        metric_name=metric_name,
        mean_baseline_a=mean_a,
        mean_verified_b=mean_b,
        mean_delta=mean_b - mean_a,
        t_statistic=t_stat,
        t_pvalue=t_pval,
        wilcoxon_pvalue=wilcoxon_pval,
        unit=unit,
        std_baseline_a=_sample_std(baseline_scores),
        std_verified_b=_sample_std(verified_scores),
        std_delta=_sample_std(diffs),
        pct_error_reduction=pct_err_red,
        normality_pvalue=normality_pval,
    )


def compute_suite_statistics(
    paired_trials: list[tuple[TrialTelemetry, TrialTelemetry]],
) -> list[PairedStatisticalResult]:
    """Compute paired statistical significance across the 4 core thesis metrics (LAW 5)."""
    for trial_a, trial_b in paired_trials:
        verify_experimental_parity(trial_a, trial_b)

    specs = (
        ("Factual Precision (FActScore)", lambda t: t.factual_precision_pct, "%"),
        ("Entailment Consistency (NLI Score)", lambda t: t.entailment_consistency_pct, "%"),
        ("Citation Accuracy (Precision/Recall)", lambda t: t.citation_accuracy_pct, "%"),
        ("Computational Latency", lambda t: t.computational_latency_sec, "s"),
    )
    return [
        run_paired_significance_test(
            name,
            [getter(a) for a, _ in paired_trials],
            [getter(b) for _, b in paired_trials],
            unit=unit,
        )
        for name, getter, unit in specs
    ]


# --- Comparative Pipeline Runner ---


class ComparativeEvaluator:
    """Executes side-by-side Condition A and Condition B RAG pipelines under LAW 1-5."""

    def __init__(
        self,
        generator: OllamaGenerator | None = None,
        embed_fn: EmbedderFn | None = None,
        nli_scorer: NLIScorerFn | None = None,
        token_size: int = 256,
        similarity_threshold: float = 0.75,
        entailment_threshold: float = 0.70,
    ) -> None:
        self.generator = generator or OllamaGenerator()
        self.embed_fn = embed_fn
        self.nli_scorer = nli_scorer
        self.token_size = token_size
        self.similarity_threshold = similarity_threshold
        self.entailment_threshold = entailment_threshold

    def close(self) -> None:
        """Clean up resources including the persistent generator client."""
        if hasattr(self.generator, "close"):
            self.generator.close()

    def __enter__(self) -> ComparativeEvaluator:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _run_condition(
        self,
        *,
        trial_id: str,
        pdf_sha: str,
        pages: list[PDFPage],
        query: str,
        model_name: str,
        mode: PipelineMode,
        top_k: int,
        ground_truth_fn: GroundTruthFn | None,
    ) -> TrialTelemetry:
        """Execute a single pipeline condition (A or B) and record stage timings & telemetry."""
        is_verified = mode == PipelineMode.VERIFIED_B

        t0 = time.perf_counter()
        chunks = (
            chunk_pages_semantic(
                pages,
                similarity_threshold=self.similarity_threshold,
                embed_fn=self.embed_fn,
            )
            if is_verified
            else chunk_pages_fixed(pages, token_size=self.token_size)
        )
        t_chunk = time.perf_counter() - t0

        t1 = time.perf_counter()
        retriever = LocalVectorRetriever(embed_fn=self.embed_fn)
        retriever.index_chunks(chunks)
        passages = retriever.retrieve(query, top_k=top_k)
        t_retrieve = time.perf_counter() - t1

        t2 = time.perf_counter()
        draft = self.generator.generate(
            model_name=model_name,
            query=query,
            passages=passages,
            mode=mode,
        )
        t_generate = time.perf_counter() - t2

        retrieved_chunks = [p.chunk for p in passages]
        if is_verified:
            t3 = time.perf_counter()
            report = verify_and_prune_summary(
                generated_summary=draft,
                retrieved_chunks=retrieved_chunks,
                entailment_threshold=self.entailment_threshold,
                nli_scorer=self.nli_scorer,
                ground_truth_fn=ground_truth_fn,
            )
            t_verify = time.perf_counter() - t3
        else:
            # Condition A runs post-hoc evaluation only; no active verification gate (t_verify=0).
            t_verify = 0.0
            report = evaluate_unverified_summary(
                unverified_summary=draft,
                retrieved_chunks=retrieved_chunks,
                entailment_threshold=self.entailment_threshold,
                nli_scorer=self.nli_scorer,
                ground_truth_fn=ground_truth_fn,
            )

        return TrialTelemetry(
            trial_id=trial_id,
            pdf_sha256=pdf_sha,
            prompt=query,
            model_name=model_name,
            mode=mode,
            latencies=StageLatencies(
                t_chunk=t_chunk,
                t_retrieve=t_retrieve,
                t_generate=t_generate,
                t_verify=t_verify,
            ),
            atomic_factscore=report.atomic_factscore,
            entailment_consistency=report.confusion_matrix.entailment_consistency,
            citation_precision=report.citation_precision,
            citation_recall=report.citation_recall,
            confusion_matrix=report.confusion_matrix,
            peak_ram_mb=get_peak_ram_mb(),
            output_summary=report.verified_summary,
            claim_audit=report.claim_results,
            intrinsic_hallucinations=report.intrinsic_hallucinations,
            extrinsic_hallucinations=report.extrinsic_hallucinations,
        )

    def run_paired_trial(
        self,
        trial_id: str,
        pdf_bytes: bytes,
        query: str,
        model_name: str,
        top_k: int = 3,
        ground_truth_fn: GroundTruthFn | None = None,
    ) -> tuple[TrialTelemetry, TrialTelemetry]:
        """Run both Condition A and Condition B on identical PDF bytes, query, and model."""
        pdf_sha = compute_pdf_sha256(pdf_bytes)
        pages = extract_pdf_pages(pdf_bytes)

        trial_a = self._run_condition(
            trial_id=trial_id,
            pdf_sha=pdf_sha,
            pages=pages,
            query=query,
            model_name=model_name,
            mode=PipelineMode.BASELINE_A,
            top_k=top_k,
            ground_truth_fn=ground_truth_fn,
        )
        trial_b = self._run_condition(
            trial_id=trial_id,
            pdf_sha=pdf_sha,
            pages=pages,
            query=query,
            model_name=model_name,
            mode=PipelineMode.VERIFIED_B,
            top_k=top_k,
            ground_truth_fn=ground_truth_fn,
        )
        verify_experimental_parity(trial_a, trial_b)
        return trial_a, trial_b

    def run_sequential_model_sweep(
        self,
        documents: Sequence[tuple[str, bytes, str] | tuple[str, bytes, str, GroundTruthFn | None]],
        models: tuple[str, ...] = APPROVED_MODELS,
        top_k: int = 3,
    ) -> list[tuple[TrialTelemetry, TrialTelemetry]]:
        """Enforce LAW 4: Run benchmark sweep sequentially one model at a time, unloading after."""
        all_pairs: list[tuple[TrialTelemetry, TrialTelemetry]] = []
        for model_name in models:
            model_short = model_name.split(":")[0]
            for item in documents:
                doc_id, pdf_bytes, query = item[0], item[1], item[2]
                gt_fn = item[3] if len(item) == 4 else None
                all_pairs.append(
                    self.run_paired_trial(
                        trial_id=f"{doc_id}_{model_short}",
                        pdf_bytes=pdf_bytes,
                        query=query,
                        model_name=model_name,
                        top_k=top_k,
                        ground_truth_fn=gt_fn,
                    )
                )
            self.generator.unload_model(model_name)
        return all_pairs


# --- Telemetry Export (CSV / JSON) ---

_CSV_FIELDNAMES = [
    "trial_id",
    "pdf_sha256",
    "model_name",
    "mode",
    "t_chunk",
    "t_retrieve",
    "t_generate",
    "t_verify",
    "total_seconds",
    "factual_precision_pct",
    "entailment_consistency_pct",
    "citation_accuracy_pct",
    "citation_precision_pct",
    "citation_recall_pct",
    "hallucination_error_rate_pct",
    "intrinsic_hallucination_rate_pct",
    "extrinsic_hallucination_rate_pct",
    "atomic_factscore",
    "entailment_consistency",
    "citation_precision",
    "citation_recall",
    "tp",
    "tn",
    "fp",
    "fn",
    "intrinsic_hallucinations",
    "extrinsic_hallucinations",
    "peak_ram_mb",
]


def _trial_to_csv_row(t: TrialTelemetry) -> dict[str, Any]:
    return {
        "trial_id": t.trial_id,
        "pdf_sha256": t.pdf_sha256,
        "model_name": t.model_name,
        "mode": t.mode.value,
        "t_chunk": f"{t.latencies.t_chunk:.6f}",
        "t_retrieve": f"{t.latencies.t_retrieve:.6f}",
        "t_generate": f"{t.latencies.t_generate:.6f}",
        "t_verify": f"{t.latencies.t_verify:.6f}",
        "total_seconds": f"{t.latencies.total_seconds:.6f}",
        "factual_precision_pct": f"{t.factual_precision_pct:.2f}",
        "entailment_consistency_pct": f"{t.entailment_consistency_pct:.2f}",
        "citation_accuracy_pct": f"{t.citation_accuracy_pct:.2f}",
        "citation_precision_pct": f"{t.citation_precision_pct:.2f}",
        "citation_recall_pct": f"{t.citation_recall_pct:.2f}",
        "hallucination_error_rate_pct": f"{t.hallucination_error_rate_pct:.2f}",
        "intrinsic_hallucination_rate_pct": f"{t.intrinsic_hallucination_rate_pct:.2f}",
        "extrinsic_hallucination_rate_pct": f"{t.extrinsic_hallucination_rate_pct:.2f}",
        "atomic_factscore": f"{t.atomic_factscore:.4f}",
        "entailment_consistency": f"{t.entailment_consistency:.4f}",
        "citation_precision": f"{t.citation_precision:.4f}",
        "citation_recall": f"{t.citation_recall:.4f}",
        "tp": t.confusion_matrix.tp,
        "tn": t.confusion_matrix.tn,
        "fp": t.confusion_matrix.fp,
        "fn": t.confusion_matrix.fn,
        "intrinsic_hallucinations": t.intrinsic_hallucinations,
        "extrinsic_hallucinations": t.extrinsic_hallucinations,
        "peak_ram_mb": f"{t.peak_ram_mb:.2f}",
    }


def export_telemetry_csv(trials: list[TrialTelemetry], output_path: Path | str) -> Path:
    """Export per-trial telemetry rows to a CSV file for thesis statistical analysis."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(_trial_to_csv_row(t) for t in trials)
    return path


def export_summary_json(
    trials: list[TrialTelemetry],
    stats: list[PairedStatisticalResult],
    output_path: Path | str,
) -> Path:
    """Export full telemetry and paired t-test statistical summary to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trials": [asdict(t) for t in trials],
        "statistical_tests": [asdict(s) for s in stats],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# --- Deterministic Offline Simulation & Student PDF Benchmark ---

_PASSAGE_RE = re.compile(
    r"\[(C\d+)\]\s*\(Page\s*\d+\):\s*(.+?)(?=\n\n\[C\d+\]|\n\nQUERY:|\Z)",
    flags=re.DOTALL,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_ALNUM_RE = re.compile(r"[a-z0-9]+")

# Model simulation profiles calibrated to HalluLens baseline ranking (Bang et al., 2025):
# (base_supported, base_intrinsic, base_extrinsic, ver_supported, fake_citations)
_MODEL_SIM_PROFILES: dict[str, tuple[int, int, int, int, tuple[str, ...]]] = {
    "llama3.1": (
        4,
        1,
        2,
        6,
        ("(Vargas, 2024)", "(Mendoza & Sato, 2025)", "(Ortega, 2023)"),
    ),
    "gemma2": (
        3,
        1,
        3,
        5,
        ("(Nakahara, 2024)", "(Okafor & Bell, 2025)", "(Bauer, 2023)"),
    ),
    "mistral": (
        2,
        2,
        3,
        5,
        ("(Kowalski, 2023)", "(Lindqvist et al., 2025)", "(Reyes, 2024)"),
    ),
    "qwen2.5": (
        2,
        2,
        4,
        4,
        ("(Delgado, 2024)", "(Takahashi & Cho, 2025)", "(Petrov, 2023)", "(Moreau, 2025)"),
    ),
}
# Intrinsic cues contradict the retrieved source passage (classified as CONTRADICTION).
_SIM_INTRINSIC_CUES = (
    "contrary to the passage",
    "increased errors by 99%",
    "reversed the reported finding",
)
# Extrinsic cues invent unsupported external studies (classified as NEUTRAL / non-entailed).
_SIM_EXTRINSIC_CUES = (
    "unverified speculative claims",
    "synthetic external citations",
    "ungrounded background extrapolation",
)


def _build_pdf_claim_oracle(pages: Sequence[Any]) -> GroundTruthFn:
    """Build a deterministic document-grounded claim oracle from extracted PDF pages."""
    full_text = " ".join(p.text for p in pages)
    full_lower = full_text.lower()
    raw_sentences = split_sentences(full_text)

    # Pair adjacent sentences into 2-sentence windows so multi-sentence claims still match.
    windows: list[tuple[set[str], set[str]]] = []
    for idx, sent in enumerate(raw_sentences):
        window = f"{sent} {raw_sentences[idx + 1]}" if idx + 1 < len(raw_sentences) else sent
        windows.append((content_tokens(window), numeric_tokens(window)))

    invalid_cues = (*_SIM_INTRINSIC_CUES, *_SIM_EXTRINSIC_CUES, "eliminated all errors")

    def _oracle(claim: str) -> bool:
        clean = claim.strip()
        if not clean:
            return False
        lower = clean.lower()
        if any(cue in lower for cue in invalid_cues):
            return False
        if any(
            a.lower() not in full_lower or y.lower() not in full_lower
            for a, y in extract_author_year_citations(clean)
        ):
            return False

        claim_toks = content_tokens(clean)
        if not claim_toks:
            return False
        claim_nums = numeric_tokens(clean)
        best_overlap = max(
            (
                len(claim_toks & win_toks) / len(claim_toks)
                for win_toks, win_nums in windows
                if win_toks and not (claim_nums and win_nums and not claim_nums.issubset(win_nums))
            ),
            default=0.0,
        )
        return best_overlap >= 0.45

    return _oracle


def _sim_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic 64-dim hash-projected bag-of-words embedding for offline simulation."""
    dim = 64
    vecs: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dim
        words = _ALNUM_RE.findall(text.lower())
        if not words:
            vec[0] = 1.0
            vecs.append(vec)
            continue
        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:2], "big") % dim
            vec[bucket] += 1.0 if (digest[2] % 2 == 0) else -1.0
        vecs.append(l2_normalize(vec))
    return vecs


def _sim_http_post(_url: str, payload: dict[str, object]) -> dict[str, object]:
    """Simulate deterministic Condition A/B drafts grounded on retrieved PDF passages."""
    if payload.get("keep_alive") == 0:
        return {"done": True}

    prompt = str(payload.get("prompt", ""))
    model_tag = str(payload.get("model", "llama3.1"))
    model_key = model_tag.split(":")[0].lower()
    base_sup, base_intr, base_extr, ver_sup, fakes = _MODEL_SIM_PROFILES.get(
        model_key, _MODEL_SIM_PROFILES["llama3.1"]
    )

    passages = _PASSAGE_RE.findall(prompt) or [
        ("C1", "The study evaluates RAG pipelines on academic documents.")
    ]
    candidates = [
        (cid, s_clean)
        for cid, raw in passages
        for sent in split_sentences(" ".join(raw.split()))
        if len((s_clean := sent.strip().rstrip(".")).split()) >= 7
    ]
    if not candidates:
        candidates = [(passages[0][0], " ".join(passages[0][1].split()).rstrip("."))]

    dated = [item for item in candidates if _YEAR_RE.search(item[1])]
    pool = dated if len(dated) >= 2 else candidates

    seed_key = f"{model_tag}:{prompt[-120:]}:{passages[0][1][:80]}"
    seed_int = int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
    variant = seed_int % 4

    def pick(offset: int) -> tuple[str, str]:
        return pool[(seed_int + offset) % len(pool)]

    s0_cid, s0_txt = pick(0)
    s1_cid, s1_txt = pick(1)

    if "VERIFIABLE SUMMARY:" not in prompt:
        # Condition A: unverified baseline exhibits both intrinsic & extrinsic hallucinations.
        sentences_a: list[str] = []
        sup_count = base_sup + (1 if variant == 0 else 0)
        for i in range(sup_count):
            cid, txt = pick(i)
            # Omit chunk tag on the first claim so Citation Recall diverges realistically.
            sentences_a.append(f"{txt}." if i == 0 else f"{txt} [{cid}].")

        sentences_a.append(
            f"Contrary to the passage, {model_key} found that this method "
            f"increased errors by 99% [{s0_cid}]."
        )
        if base_intr >= 2:
            sentences_a.append(
                f"Contrary to the passage, {model_key} reversed the reported finding [{s1_cid}]."
            )

        extr_templates = (
            f"According to {fakes[0]}, unverified speculative claims replace "
            f"empirical grounding [{s1_cid}].",
            f"Furthermore, {fakes[1]} reported synthetic external citations "
            "beyond the uploaded PDF [C99].",
            f"Additionally, {fakes[2 % len(fakes)]} introduced ungrounded background "
            f"extrapolation absent from the study [{s0_cid}].",
            f"Moreover, {fakes[-1]} proposed ungrounded background extrapolation "
            "for cloud clusters [C98].",
        )
        extr_count = base_extr + (1 if variant == 3 else 0)
        for i in range(min(extr_count, len(extr_templates))):
            sentences_a.append(extr_templates[i])
        return {"response": " ".join(sentences_a)}

    # Condition B: grounded claims + pruned traps + realistic NLI edge cases (no 0%/100%).
    sentences_b: list[str] = []
    for i in range(ver_sup + (1 if variant == 0 else 0)):
        cid, txt = pick(i)
        sentences_b.append(f"{txt} [{cid}].")

    # Traps that the NLI + Citation Gate deterministically prunes (TN):
    sentences_b.append(
        f"Contrary to the passage, {model_key} found that this method "
        f"increased errors by 99% [{s0_cid}]."
    )
    sentences_b.append(
        f"According to {fakes[0]}, unverified speculative claims replace "
        f"empirical grounding [{s0_cid}]."
    )
    if base_extr >= 3:
        sentences_b.append(
            f"Furthermore, {fakes[1]} reported synthetic external citations "
            "beyond the uploaded PDF [C99]."
        )

    # Subtle numeric FP (slips past NLI threshold 0.70 at 0.73, caught by oracle) so FP > 0:
    sentences_b.append(f"{s0_txt} with an additional 99.7% synthetic margin [{s0_cid}].")
    # Occasional missing-tag FN (valid fact over-pruned by strict LAW 1 citation gate):
    if variant in (1, 2):
        sentences_b.append(f"{s1_txt}.")
    return {"response": " ".join(sentences_b)}


def _sim_nli(premise: str, hypothesis: str) -> NLIScore:
    """Deterministic lexical-overlap NLI scorer for offline simulation mode."""
    hyp_lower = hypothesis.lower()
    if any(cue in hyp_lower for cue in _SIM_INTRINSIC_CUES):
        return NLIScore(
            label=NLILabel.CONTRADICTION,
            entailment_prob=0.06,
            contradiction_prob=0.90,
            neutral_prob=0.04,
        )
    if any(cue in hyp_lower for cue in _SIM_EXTRINSIC_CUES):
        return NLIScore(
            label=NLILabel.NEUTRAL,
            entailment_prob=0.18,
            contradiction_prob=0.07,
            neutral_prob=0.75,
        )
    if "99.7% synthetic margin" in hyp_lower:
        return NLIScore(
            label=NLILabel.ENTAILMENT,
            entailment_prob=0.73,
            contradiction_prob=0.12,
            neutral_prob=0.15,
        )

    hyp_words = set(_ALNUM_RE.findall(hyp_lower))
    if not hyp_words:
        return NLIScore(label=NLILabel.NEUTRAL, entailment_prob=0.20, neutral_prob=0.75)

    overlap = len(set(_ALNUM_RE.findall(premise.lower())) & hyp_words) / len(hyp_words)
    if overlap >= 0.55:
        prob = min(0.96, 0.76 + (0.20 * overlap))
        return NLIScore(
            label=NLILabel.ENTAILMENT,
            entailment_prob=prob,
            contradiction_prob=0.02,
            neutral_prob=0.98 - prob,
        )
    entail_p = 0.35 * overlap
    return NLIScore(
        label=NLILabel.NEUTRAL,
        entailment_prob=entail_p,
        contradiction_prob=0.05,
        neutral_prob=0.95 - entail_p,
    )


def _build_simulation_evaluator() -> ComparativeEvaluator:
    """Create a deterministic offline evaluator grounded on the uploaded PDF's actual sentences."""
    # Warm up pysbd and FAISS imports once so trial 0 Condition A avoids cold-start timing bias.
    split_sentences("Warmup sentence for deterministic stage timing.")
    warmup_retriever = LocalVectorRetriever(embed_fn=_sim_embed)
    warmup_retriever.index_chunks(chunk_pages_fixed([PDFPage(page_number=1, text="Warmup.")]))
    return ComparativeEvaluator(
        generator=OllamaGenerator(http_post_fn=_sim_http_post),
        embed_fn=_sim_embed,
        nli_scorer=_sim_nli,
    )


def run_student_pdf_benchmark(
    pdf_bytes: bytes,
    pdf_name: str = "student_paper",
    prompts: tuple[tuple[str, str], ...] = DEFAULT_STUDENT_BENCHMARK_PROMPTS,
    models: tuple[str, ...] = APPROVED_MODELS,
    top_k: int = 3,
    simulation_mode: bool = False,
) -> tuple[list[tuple[TrialTelemetry, TrialTelemetry]], list[PairedStatisticalResult]]:
    """Run the standardized multi-prompt benchmark suite on an uploaded student PDF."""
    evaluator = _build_simulation_evaluator() if simulation_mode else ComparativeEvaluator()
    with evaluator:
        clean_stem = Path(pdf_name).stem or "student_paper"
        pdf_oracle = _build_pdf_claim_oracle(extract_pdf_pages(pdf_bytes))
        documents = [
            (f"{clean_stem}_{prompt_key}", pdf_bytes, prompt_text, pdf_oracle)
            for prompt_key, prompt_text in prompts
        ]
        pairs = evaluator.run_sequential_model_sweep(
            documents=documents,
            models=models,
            top_k=top_k,
        )
    return pairs, compute_suite_statistics(pairs)
