"""Comparative evaluation & statistical significance harness enforcing LAW 3, LAW 4, and LAW 5."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import resource
import sys
import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from vrag.chunker import (
    EmbedderFn,
    chunk_pages_fixed,
    chunk_pages_semantic,
    extract_pdf_pages,
    split_sentences,
)
from vrag.generator import OllamaGenerator
from vrag.models import (
    APPROVED_MODELS,
    DEFAULT_STUDENT_BENCHMARK_PROMPTS,
    NLILabel,
    NLIScore,
    PipelineMode,
    StageLatencies,
    TrialTelemetry,
)
from vrag.retriever import LocalVectorRetriever
from vrag.verifier import (
    GroundTruthFn,
    NLIScorerFn,
    evaluate_unverified_summary,
    verify_and_prune_summary,
)


def _sample_std(values: list[float]) -> float:
    """Compute sample standard deviation (ddof=1) for n >= 2."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_val = sum(values) / n
    variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    return math.sqrt(max(0.0, variance))


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
                info = proc.info
                pname = str(info.get("name") or "").lower()
                if info.get("pid") != current_pid and "ollama" in pname:
                    total_rss += float(proc.memory_info().rss)
            except Exception:
                continue
        return total_rss / (1024.0 * 1024.0)
    except Exception:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS ru_maxrss is in bytes; on Linux it is in kilobytes
        if sys.platform == "darwin":
            return float(usage) / (1024.0 * 1024.0)
        return float(usage) / 1024.0


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

    mean_a = sum(baseline_scores) / len(baseline_scores)
    mean_b = sum(verified_scores) / len(verified_scores)
    delta = mean_b - mean_a

    diffs = [b - a for a, b in zip(baseline_scores, verified_scores)]
    std_a = _sample_std(baseline_scores)
    std_b = _sample_std(verified_scores)
    std_d = _sample_std(diffs)

    if unit == "%":
        scale_max = 1.0 if (max(baseline_scores + verified_scores) <= 1.0) else 100.0
        err_a = max(0.0, scale_max - mean_a)
        err_b = max(0.0, scale_max - mean_b)
        pct_err_red = ((err_a - err_b) / err_a) * 100.0 if err_a > 1e-12 else 0.0
    else:
        pct_err_red = ((mean_a - mean_b) / mean_a) * 100.0 if mean_a > 1e-12 else 0.0

    zero_var = all(abs(d - diffs[0]) < 1e-12 for d in diffs)
    normality_pval: float | None = None
    if len(diffs) >= 3 and not zero_var:
        try:
            shapiro_res = stats.shapiro(diffs)
            normality_pval = float(shapiro_res.pvalue)
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
                w_res = stats.wilcoxon(verified_scores, baseline_scores)
            wilcoxon_pval = float(w_res.pvalue)
        except ValueError:
            wilcoxon_pval = None

    return PairedStatisticalResult(
        metric_name=metric_name,
        mean_baseline_a=mean_a,
        mean_verified_b=mean_b,
        mean_delta=delta,
        t_statistic=t_stat,
        t_pvalue=t_pval,
        wilcoxon_pvalue=wilcoxon_pval,
        unit=unit,
        std_baseline_a=std_a,
        std_verified_b=std_b,
        std_delta=std_d,
        pct_error_reduction=pct_err_red,
        normality_pvalue=normality_pval,
    )


def compute_suite_statistics(
    paired_trials: list[tuple[TrialTelemetry, TrialTelemetry]],
) -> list[PairedStatisticalResult]:
    """Compute paired statistical significance across the 4 core thesis metrics (LAW 5)."""
    for trial_a, trial_b in paired_trials:
        verify_experimental_parity(trial_a, trial_b)

    metrics: list[tuple[str, list[float], list[float], str]] = [
        (
            "Factual Precision (FActScore)",
            [p[0].factual_precision_pct for p in paired_trials],
            [p[1].factual_precision_pct for p in paired_trials],
            "%",
        ),
        (
            "Entailment Consistency (NLI Score)",
            [p[0].entailment_consistency_pct for p in paired_trials],
            [p[1].entailment_consistency_pct for p in paired_trials],
            "%",
        ),
        (
            "Citation Accuracy (Precision/Recall)",
            [p[0].citation_accuracy_pct for p in paired_trials],
            [p[1].citation_accuracy_pct for p in paired_trials],
            "%",
        ),
        (
            "Computational Latency",
            [p[0].computational_latency_sec for p in paired_trials],
            [p[1].computational_latency_sec for p in paired_trials],
            "s",
        ),
    ]
    return [
        run_paired_significance_test(name, scores_a, scores_b, unit=unit)
        for name, scores_a, scores_b, unit in metrics
    ]


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

        # --- Condition A: Baseline Standard RAG ---
        t0 = time.perf_counter()
        chunks_a = chunk_pages_fixed(pages, token_size=self.token_size)
        t_chunk_a = time.perf_counter() - t0

        t1 = time.perf_counter()
        retriever_a = LocalVectorRetriever(embed_fn=self.embed_fn)
        retriever_a.index_chunks(chunks_a)
        passages_a = retriever_a.retrieve(query, top_k=top_k)
        t_retrieve_a = time.perf_counter() - t1

        t2 = time.perf_counter()
        raw_summary_a = self.generator.generate(
            model_name=model_name,
            query=query,
            passages=passages_a,
            mode=PipelineMode.BASELINE_A,
        )
        t_generate_a = time.perf_counter() - t2

        # Condition A has no active verification gate (t_verify = 0.0); evaluate post-hoc
        eval_report_a = evaluate_unverified_summary(
            unverified_summary=raw_summary_a,
            retrieved_chunks=[p.chunk for p in passages_a],
            entailment_threshold=self.entailment_threshold,
            nli_scorer=self.nli_scorer,
            ground_truth_fn=ground_truth_fn,
        )
        trial_a = TrialTelemetry(
            trial_id=trial_id,
            pdf_sha256=pdf_sha,
            prompt=query,
            model_name=model_name,
            mode=PipelineMode.BASELINE_A,
            latencies=StageLatencies(
                t_chunk=t_chunk_a,
                t_retrieve=t_retrieve_a,
                t_generate=t_generate_a,
                t_verify=0.0,
            ),
            atomic_factscore=eval_report_a.atomic_factscore,
            entailment_consistency=eval_report_a.confusion_matrix.entailment_consistency,
            citation_precision=eval_report_a.citation_precision,
            citation_recall=eval_report_a.citation_recall,
            confusion_matrix=eval_report_a.confusion_matrix,
            peak_ram_mb=get_peak_ram_mb(),
            output_summary=eval_report_a.verified_summary,
            claim_audit=eval_report_a.claim_results,
            intrinsic_hallucinations=eval_report_a.intrinsic_hallucinations,
            extrinsic_hallucinations=eval_report_a.extrinsic_hallucinations,
        )

        # --- Condition B: Proposed Verification-Driven RAG ---
        t3 = time.perf_counter()
        chunks_b = chunk_pages_semantic(
            pages,
            similarity_threshold=self.similarity_threshold,
            embed_fn=self.embed_fn,
        )
        t_chunk_b = time.perf_counter() - t3

        t4 = time.perf_counter()
        retriever_b = LocalVectorRetriever(embed_fn=self.embed_fn)
        retriever_b.index_chunks(chunks_b)
        passages_b = retriever_b.retrieve(query, top_k=top_k)
        t_retrieve_b = time.perf_counter() - t4

        t5 = time.perf_counter()
        draft_summary_b = self.generator.generate(
            model_name=model_name,
            query=query,
            passages=passages_b,
            mode=PipelineMode.VERIFIED_B,
        )
        t_generate_b = time.perf_counter() - t5

        t6 = time.perf_counter()
        ver_report_b = verify_and_prune_summary(
            generated_summary=draft_summary_b,
            retrieved_chunks=[p.chunk for p in passages_b],
            entailment_threshold=self.entailment_threshold,
            nli_scorer=self.nli_scorer,
            ground_truth_fn=ground_truth_fn,
        )
        t_verify_b = time.perf_counter() - t6

        trial_b = TrialTelemetry(
            trial_id=trial_id,
            pdf_sha256=pdf_sha,
            prompt=query,
            model_name=model_name,
            mode=PipelineMode.VERIFIED_B,
            latencies=StageLatencies(
                t_chunk=t_chunk_b,
                t_retrieve=t_retrieve_b,
                t_generate=t_generate_b,
                t_verify=t_verify_b,
            ),
            atomic_factscore=ver_report_b.atomic_factscore,
            entailment_consistency=ver_report_b.confusion_matrix.entailment_consistency,
            citation_precision=ver_report_b.citation_precision,
            citation_recall=ver_report_b.citation_recall,
            confusion_matrix=ver_report_b.confusion_matrix,
            peak_ram_mb=get_peak_ram_mb(),
            output_summary=ver_report_b.verified_summary,
            claim_audit=ver_report_b.claim_results,
            intrinsic_hallucinations=ver_report_b.intrinsic_hallucinations,
            extrinsic_hallucinations=ver_report_b.extrinsic_hallucinations,
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
            for item in documents:
                if len(item) == 4:
                    doc_id, pdf_bytes, query, gt_fn = item
                else:
                    doc_id, pdf_bytes, query = item
                    gt_fn = None
                trial_id = f"{doc_id}_{model_name.split(':')[0]}"
                pair = self.run_paired_trial(
                    trial_id=trial_id,
                    pdf_bytes=pdf_bytes,
                    query=query,
                    model_name=model_name,
                    top_k=top_k,
                    ground_truth_fn=gt_fn,
                )
                all_pairs.append(pair)
            self.generator.unload_model(model_name)
        return all_pairs


def export_telemetry_csv(trials: list[TrialTelemetry], output_path: Path | str) -> Path:
    """Export per-trial telemetry rows to a CSV file for thesis statistical analysis."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for t in trials:
            writer.writerow(
                {
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
            )
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


def _build_simulation_evaluator() -> ComparativeEvaluator:
    """Create a deterministic offline evaluator grounded on the uploaded PDF's actual sentences."""
    import re

    def sim_embed(texts: list[str]) -> list[list[float]]:
        vecs: list[list[float]] = []
        for idx, t in enumerate(texts):
            length_feat = float((len(t) % 17) + 1)
            word_feat = float((len(t.split()) % 13) + 1)
            vecs.append([length_feat, word_feat + (0.01 * idx)])
        return vecs

    def sim_http_post(_url: str, payload: dict[str, object]) -> dict[str, object]:
        if payload.get("keep_alive") == 0:
            return {"done": True}
        prompt = str(payload.get("prompt", ""))
        match = re.search(
            r"\[(C\d+)\]\s*\(Page\s*\d+\):\s*(.+?)(?=\n\n\[C\d+\]|\n\nQUERY:|\Z)",
            prompt,
            flags=re.DOTALL,
        )
        chunk_id = match.group(1) if match else "C1"
        raw_context = (
            " ".join(match.group(2).split()) if match else "The study evaluates RAG pipelines."
        )
        sents = split_sentences(raw_context)
        chosen_sent = next(
            (s for s in sents if re.search(r"\b(?:19|20)\d{2}\b", s)),
            sents[0] if sents else raw_context,
        )
        first_sent = chosen_sent.rstrip(".")

        if "VERIFIABLE SUMMARY:" in prompt:
            return {
                "response": (
                    f"{first_sent} [{chunk_id}]. "
                    f"According to (FabricatedAuthor, 2025), unverified speculative claims "
                    f"replace empirical grounding [{chunk_id}]."
                )
            }
        return {
            "response": (
                f"{first_sent} [{chunk_id}]. "
                f"Furthermore, (FabricatedAuthor, 2025) proved unverified speculative claims "
                f"replace empirical grounding [{chunk_id}]."
            )
        }

    def sim_nli(_premise: str, hypothesis: str) -> NLIScore:
        if "unverified speculative claims" in hypothesis.lower():
            return NLIScore(
                label=NLILabel.CONTRADICTION,
                entailment_prob=0.04,
                contradiction_prob=0.92,
                neutral_prob=0.04,
            )
        return NLIScore(
            label=NLILabel.ENTAILMENT,
            entailment_prob=0.95,
            contradiction_prob=0.02,
            neutral_prob=0.03,
        )

    return ComparativeEvaluator(
        generator=OllamaGenerator(http_post_fn=sim_http_post),
        embed_fn=sim_embed,
        nli_scorer=sim_nli,
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
    clean_stem = Path(pdf_name).stem or "student_paper"
    documents = [
        (f"{clean_stem}_{prompt_key}", pdf_bytes, prompt_text)
        for prompt_key, prompt_text in prompts
    ]
    pairs = evaluator.run_sequential_model_sweep(
        documents=documents,
        models=models,
        top_k=top_k,
    )
    stats = compute_suite_statistics(pairs)
    return pairs, stats
