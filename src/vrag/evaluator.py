"""Comparative evaluation & statistical significance harness enforcing LAW 3 and LAW 5."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from vrag.models import PipelineMode, TrialTelemetry


@dataclass(frozen=True)
class PairedStatisticalResult:
    """Paired t-test and Wilcoxon significance result across N comparative trials."""

    metric_name: str
    mean_baseline_a: float
    mean_verified_b: float
    mean_delta: float
    t_statistic: float
    t_pvalue: float
    wilcoxon_pvalue: float | None = None


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
) -> PairedStatisticalResult:
    """Enforce LAW 5: Compute paired t-test (scipy.stats.ttest_rel) & Wilcoxon p-value."""
    if len(baseline_scores) != len(verified_scores):
        raise ValueError("LAW 5 violation: Paired sample arrays must have identical length")
    if len(baseline_scores) < 2:
        raise ValueError("LAW 5 violation: At least 2 paired trials required for ttest_rel")

    from scipy import stats  # type: ignore[import-untyped]

    mean_a = sum(baseline_scores) / len(baseline_scores)
    mean_b = sum(verified_scores) / len(verified_scores)
    delta = mean_b - mean_a

    t_res = stats.ttest_rel(verified_scores, baseline_scores)
    t_stat = 0.0 if math.isnan(float(t_res.statistic)) else float(t_res.statistic)
    t_pval = 1.0 if math.isnan(float(t_res.pvalue)) else float(t_res.pvalue)

    wilcoxon_pval: float | None = None
    diffs = [b - a for a, b in zip(baseline_scores, verified_scores)]
    if any(d != 0.0 for d in diffs):
        try:
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
    )
