"""Immutable data contracts for the Verification-Driven RAG benchmarking harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

APPROVED_MODELS: tuple[str, ...] = (
    "llama3.1:8b-instruct-q4_K_M",
    "mistral:7b-instruct-v0.3-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
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
        """Proportion of surviving claims that are truly entailed: TP / (TP + FP)."""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 1.0


@dataclass(frozen=True)
class VerificationReport:
    """Output of the Condition B deterministic verification gate."""

    verified_summary: str
    claim_results: tuple[ClaimVerificationResult, ...]
    confusion_matrix: ConfusionMatrix
    atomic_factscore: float
    citation_precision: float
    citation_recall: float


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
