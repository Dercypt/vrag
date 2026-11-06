"""Immutable data contracts for the Verification-Driven RAG benchmarking harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

APPROVED_MODELS: tuple[str, ...] = (
    "llama3.1:8b-instruct-q4_K_M",
    "mistral:7b-instruct-v0.3-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
)

DEFAULT_STUDENT_BENCHMARK_PROMPTS: tuple[tuple[str, str], ...] = (
    (
        "contributions",
        "Summarize the core research problem and primary contributions of this paper.",
    ),
    (
        "methodology",
        "Describe the proposed methodology, pipeline architecture, and algorithms.",
    ),
    (
        "findings",
        "Summarize the experimental evaluation metrics, empirical results, and key findings.",
    ),
    (
        "limitations",
        "Identify the stated limitations, hardware constraints, and future work.",
    ),
)


class NLILabel(str, Enum):
    """Natural Language Inference classification label."""

    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


class PipelineMode(str, Enum):
    """Experimental condition identifier."""

    BASELINE_A = "condition_a_baseline"
    VERIFIED_B = "condition_b_verified"


@dataclass(frozen=True)
class PDFPage:
    """A single parsed page from an academic PDF document."""

    page_number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A text segment extracted from an academic PDF."""

    chunk_id: str
    text: str
    page_number: int = 1


@dataclass(frozen=True)
class RetrievedPassage:
    """A chunk retrieved from the vector index with its similarity score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class NLIScore:
    """Cross-encoder NLI score result for a (premise, hypothesis) pair."""

    label: NLILabel
    entailment_prob: float
    neutral_prob: float = 0.0
    contradiction_prob: float = 0.0


@dataclass(frozen=True)
class GroundTruthSpec:
    """Curated ground-truth claims, traps, and valid citations for benchmark evaluation."""

    supported_claims: tuple[str, ...] = ()
    intrinsic_traps: tuple[str, ...] = ()
    extrinsic_traps: tuple[str, ...] = ()
    valid_citations: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ClaimVerificationResult:
    """Audit record for a single atomic sentence evaluated in Condition B."""

    sentence: str
    cited_chunk_ids: tuple[str, ...]
    extracted_authors: tuple[str, ...]
    nli_label: NLILabel
    entailment_score: float
    citation_exists: bool
    author_exists: bool
    survived: bool
    prune_reason: str | None = None
    extracted_years: tuple[str, ...] = ()
    year_exists: bool = True


@dataclass(frozen=True)
class ConfusionMatrix:
    """Verification gate confusion matrix (TP, TN, FP, FN)."""

    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def entailment_consistency(self) -> float:
        """Chapter 3 (p. 53) classification accuracy: (TP + TN) / (TP + TN + FP + FN)."""
        denom = self.total
        return (self.tp + self.tn) / denom if denom > 0 else 0.0


@dataclass(frozen=True)
class VerificationReport:
    """Output of the Condition B deterministic verification gate."""

    verified_summary: str
    claim_results: tuple[ClaimVerificationResult, ...]
    confusion_matrix: ConfusionMatrix
    atomic_factscore: float
    citation_precision: float
    citation_recall: float
    intrinsic_hallucinations: int = 0
    extrinsic_hallucinations: int = 0


@dataclass(frozen=True)
class StageLatencies:
    """Per-stage execution timings in seconds (LAW 5)."""

    t_chunk: float
    t_retrieve: float
    t_generate: float
    t_verify: float = 0.0

    def __post_init__(self) -> None:
        for name, val in (
            ("t_chunk", self.t_chunk),
            ("t_retrieve", self.t_retrieve),
            ("t_generate", self.t_generate),
            ("t_verify", self.t_verify),
        ):
            if val < 0.0:
                raise ValueError(f"LAW 5 violation: {name} must be non-negative, got {val}")

    @property
    def total_seconds(self) -> float:
        return self.t_chunk + self.t_retrieve + self.t_generate + self.t_verify


@dataclass(frozen=True)
class TrialTelemetry:
    """Complete telemetry record for a single pipeline execution."""

    trial_id: str
    pdf_sha256: str
    prompt: str
    model_name: str
    mode: PipelineMode
    latencies: StageLatencies
    atomic_factscore: float
    entailment_consistency: float
    citation_precision: float
    citation_recall: float
    confusion_matrix: ConfusionMatrix
    peak_ram_mb: float
    output_summary: str
    claim_audit: tuple[ClaimVerificationResult, ...] = field(default_factory=tuple)
    intrinsic_hallucinations: int = 0
    extrinsic_hallucinations: int = 0

    @property
    def factual_precision_pct(self) -> float:
        """Metric 1: Factual Precision (FActScore) in percentage (0-100%)."""
        return self.atomic_factscore * 100.0

    @property
    def hallucination_error_rate_pct(self) -> float:
        """Overall hallucination error rate among surviving output claims (0-100%)."""
        denom = self.confusion_matrix.tp + self.confusion_matrix.fp
        return (self.confusion_matrix.fp / denom) * 100.0 if denom > 0 else 0.0

    @property
    def intrinsic_hallucination_rate_pct(self) -> float:
        """Intrinsic faithfulness contradiction rate among surviving output claims (0-100%)."""
        denom = self.confusion_matrix.tp + self.confusion_matrix.fp
        return (self.intrinsic_hallucinations / denom) * 100.0 if denom > 0 else 0.0

    @property
    def extrinsic_hallucination_rate_pct(self) -> float:
        """Extrinsic factuality/citation fabrication rate among surviving output claims (0-100%)."""
        denom = self.confusion_matrix.tp + self.confusion_matrix.fp
        return (self.extrinsic_hallucinations / denom) * 100.0 if denom > 0 else 0.0

    @property
    def entailment_consistency_pct(self) -> float:
        """Metric 2: Entailment Consistency (NLI Score) in percentage (0-100%)."""
        return self.entailment_consistency * 100.0

    @property
    def citation_precision_pct(self) -> float:
        """Citation Precision in percentage (0-100%)."""
        return self.citation_precision * 100.0

    @property
    def citation_recall_pct(self) -> float:
        """Citation Recall in percentage (0-100%)."""
        return self.citation_recall * 100.0

    @property
    def citation_accuracy_pct(self) -> float:
        """Metric 3: Citation Accuracy (F1 of Precision & Recall) in percentage (0-100%)."""
        denom = self.citation_precision + self.citation_recall
        if denom <= 0.0:
            return 0.0
        f1 = (2.0 * self.citation_precision * self.citation_recall) / denom
        return f1 * 100.0

    @property
    def computational_latency_sec(self) -> float:
        """Metric 4: End-to-end Computational Latency in seconds."""
        return self.latencies.total_seconds
