"""Minimal, modern Streamlit UI for Student PDF Benchmarking across the 4 Core Thesis Metrics.

Features a light/dark theme toggle, custom-rendered metric cards and tables, and a quiet
typographic system (Inter). All benchmark logic and row-builder helpers are unchanged.
"""

from __future__ import annotations

import csv
import html
import io
import json
from dataclasses import asdict

from vrag.cli import format_metric_value
from vrag.evaluator import (
    PairedStatisticalResult,
    run_student_pdf_benchmark,
)
from vrag.generator import OllamaHealthReport, diagnose_ollama
from vrag.models import (
    APPROVED_MODELS,
    DEFAULT_STUDENT_BENCHMARK_PROMPTS,
    NLILabel,
    TrialTelemetry,
)

METRIC_DESCRIPTIONS: dict[str, str] = {
    "Factual Precision (FActScore)": (
        "Percentage of generated claims actually supported by the uploaded PDF (%)."
    ),
    "Entailment Consistency (NLI Score)": (
        "Classification accuracy across retained and pruned claims: "
        "(TP + TN) / (TP + TN + FP + FN) (%)."
    ),
    "Citation Accuracy (Precision/Recall)": (
        "Whether real authors and chunk/page numbers are cited without inventing fake ones (%)."
    ),
    "Computational Latency": (
        "End-to-end execution time (in seconds) on local 2019 MacBook Pro hardware (s)."
    ),
}


def build_four_metrics_summary_rows(
    stats: list[PairedStatisticalResult],
) -> list[dict[str, str]]:
    """Format the 4 core comparative metrics into display rows with % and s units."""
    rows: list[dict[str, str]] = []
    for res in stats:
        rows.append(
            {
                "Metric": res.metric_name,
                "Description": METRIC_DESCRIPTIONS.get(res.metric_name, ""),
                "Unit": res.unit,
                "Base RAG (Condition A)": format_metric_value(res.mean_baseline_a, res.unit),
                "Verification-Driven RAG (Condition B)": format_metric_value(
                    res.mean_verified_b, res.unit
                ),
                "Delta (B - A)": format_metric_value(res.mean_delta, res.unit, signed=True),
                "SD (s_d)": f"{res.std_delta:.2f}",
                "Error Reduction": f"{res.pct_error_reduction:+.1f}%",
                "Paired t-stat": f"{res.t_statistic:.4f}",
                "p-value": f"{res.t_pvalue:.4e}",
                "Wilcoxon p": (
                    f"{res.wilcoxon_pvalue:.4e}" if res.wilcoxon_pvalue is not None else "N/A"
                ),
            }
        )
    return rows


def build_trial_comparison_rows(
    pairs: list[tuple[TrialTelemetry, TrialTelemetry]],
) -> list[dict[str, object]]:
    """Build per-trial side-by-side comparison rows for the 4 core metrics."""
    rows: list[dict[str, object]] = []
    for trial_a, trial_b in pairs:
        rows.append(
            {
                "Trial ID": trial_a.trial_id,
                "Model": trial_a.model_name,
                "Prompt": trial_a.prompt,
                "Base FActScore (%)": f"{trial_a.factual_precision_pct:.1f}%",
                "Verified FActScore (%)": f"{trial_b.factual_precision_pct:.1f}%",
                "Base NLI Consistency (%)": f"{trial_a.entailment_consistency_pct:.1f}%",
                "Verified NLI Consistency (%)": f"{trial_b.entailment_consistency_pct:.1f}%",
                "Base Citation Acc (%)": f"{trial_a.citation_accuracy_pct:.1f}%",
                "Verified Citation Acc (%)": f"{trial_b.citation_accuracy_pct:.1f}%",
                "Base Latency (s)": f"{trial_a.computational_latency_sec:.4f}s",
                "Verified Latency (s)": f"{trial_b.computational_latency_sec:.4f}s",
                "Verified TP/TN/FP/FN": (
                    f"{trial_b.confusion_matrix.tp}/{trial_b.confusion_matrix.tn}/"
                    f"{trial_b.confusion_matrix.fp}/{trial_b.confusion_matrix.fn}"
                ),
                "Peak RAM (MB)": f"{trial_b.peak_ram_mb:.1f}",
            }
        )
    return rows


def build_claim_audit_rows(trial_b: TrialTelemetry) -> list[dict[str, object]]:
    """Build sentence-by-sentence verification gate audit rows for Condition B."""
    rows: list[dict[str, object]] = []
    for claim in trial_b.claim_audit:
        rows.append(
            {
                "Atomic Sentence": claim.sentence,
                "Cited Chunks": ", ".join(claim.cited_chunk_ids) or "None",
                "Extracted Authors": ", ".join(claim.extracted_authors) or "None",
                "Extracted Years": ", ".join(claim.extracted_years) or "None",
                "NLI Label": claim.nli_label.value,
                "NLI Score (%)": f"{claim.entailment_score * 100.0:.1f}%",
                "Gate Decision": "SURVIVED (TP)" if claim.survived else "PRUNED (TN)",
                "Prune Reason": claim.prune_reason or "Verified & Retained",
            }
        )
    return rows


_FRIENDLY_MODEL_NAMES: dict[str, str] = {
    "llama3.1:8b-instruct-q4_K_M": "Llama-3.1-8B",
    "mistral:7b-instruct-v0.3-q4_K_M": "Mistral-7B",
    "qwen2.5:7b-instruct-q4_K_M": "Qwen-2.5-7B",
}


def format_model_label(model_name: str) -> str:
    """Return a concise, human-readable model label for UI tables and selectors."""
    return _FRIENDLY_MODEL_NAMES.get(model_name, model_name.split(":")[0])


def build_model_hallucination_rows(
    pairs: list[tuple[TrialTelemetry, TrialTelemetry]],
) -> list[dict[str, object]]:
    """Aggregate hallucination rates and counts per model across Condition A vs. Condition B."""
    grouped: dict[str, list[tuple[TrialTelemetry, TrialTelemetry]]] = {}
    for pair in pairs:
        grouped.setdefault(pair[0].model_name, []).append(pair)

    rows: list[dict[str, object]] = []
    for model_name, model_pairs in grouped.items():
        n = len(model_pairs)
        base_halluc_rate = sum(a.hallucination_error_rate_pct for a, _ in model_pairs) / n
        ver_halluc_rate = sum(b.hallucination_error_rate_pct for _, b in model_pairs) / n
        int_a = sum(a.intrinsic_hallucinations for a, _ in model_pairs)
        int_b = sum(b.intrinsic_hallucinations for _, b in model_pairs)
        ext_a = sum(a.extrinsic_hallucinations for a, _ in model_pairs)
        ext_b = sum(b.extrinsic_hallucinations for _, b in model_pairs)
        ver_fact = sum(b.factual_precision_pct for _, b in model_pairs) / n
        avg_lat = sum(b.computational_latency_sec for _, b in model_pairs) / n

        rows.append(
            {
                "Model": format_model_label(model_name),
                "Base Hallucination Rate": f"{base_halluc_rate:.1f}%",
                "Verified Hallucination Rate": f"{ver_halluc_rate:.1f}%",
                "Intrinsic (Contradictions)": f"{int_a} → {int_b}",
                "Extrinsic (Fabricated)": f"{ext_a} → {ext_b}",
                "Verified FActScore": f"{ver_fact:.1f}%",
                "Avg Latency": f"{avg_lat:.4f}s",
            }
        )
    return rows


def build_base_hallucination_evidence(trial_a: TrialTelemetry) -> list[dict[str, str]]:
    """Extract concrete hallucinated claims from Condition A for transparent proof."""
    evidence: list[dict[str, str]] = []
    for claim in trial_a.claim_audit:
        if (
            claim.citation_exists
            and claim.author_exists
            and claim.year_exists
            and claim.nli_label == NLILabel.ENTAILMENT
            and claim.entailment_score >= 0.70
        ):
            continue

        if claim.nli_label == NLILabel.CONTRADICTION:
            h_type = "Intrinsic (Contradicts PDF)"
            detail = f"NLI contradiction ({claim.entailment_score * 100:.0f}% entailment)"
        elif not claim.author_exists or not claim.year_exists:
            h_type = "Extrinsic (Fabricated Citation)"
            missing = []
            if not claim.author_exists and claim.extracted_authors:
                missing.append(f"author {', '.join(claim.extracted_authors)}")
            if not claim.year_exists and claim.extracted_years:
                missing.append(f"year {', '.join(claim.extracted_years)}")
            detail = f"Not found in cited PDF chunk ({'; '.join(missing) or 'invalid citation'})"
        elif not claim.citation_exists:
            h_type = "Extrinsic (Invalid Chunk Tag)"
            detail = (
                f"Cited {', '.join(claim.cited_chunk_ids) or 'no chunk'} " "which was not retrieved"
            )
        else:
            h_type = "Extrinsic (Unsupported Claim)"
            detail = f"NLI {claim.nli_label.value} ({claim.entailment_score * 100:.0f}% entailment)"

        evidence.append(
            {
                "Type": h_type,
                "Flagged Claim": claim.sentence,
                "Detail": detail,
            }
        )
    return evidence


def _trials_to_csv_bytes(trials: list[TrialTelemetry]) -> bytes:
    buf = io.StringIO()
    fieldnames = [
        "trial_id",
        "model_name",
        "mode",
        "factual_precision_pct",
        "entailment_consistency_pct",
        "citation_accuracy_pct",
        "citation_precision_pct",
        "citation_recall_pct",
        "hallucination_error_rate_pct",
        "intrinsic_hallucination_rate_pct",
        "extrinsic_hallucination_rate_pct",
        "computational_latency_sec",
        "tp",
        "tn",
        "fp",
        "fn",
        "peak_ram_mb",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for t in trials:
        writer.writerow(
            {
                "trial_id": t.trial_id,
                "model_name": t.model_name,
                "mode": t.mode.value,
                "factual_precision_pct": f"{t.factual_precision_pct:.2f}",
                "entailment_consistency_pct": f"{t.entailment_consistency_pct:.2f}",
                "citation_accuracy_pct": f"{t.citation_accuracy_pct:.2f}",
                "citation_precision_pct": f"{t.citation_precision_pct:.2f}",
                "citation_recall_pct": f"{t.citation_recall_pct:.2f}",
                "hallucination_error_rate_pct": f"{t.hallucination_error_rate_pct:.2f}",
                "intrinsic_hallucination_rate_pct": f"{t.intrinsic_hallucination_rate_pct:.2f}",
                "extrinsic_hallucination_rate_pct": f"{t.extrinsic_hallucination_rate_pct:.2f}",
                "computational_latency_sec": f"{t.computational_latency_sec:.6f}",
                "tp": t.confusion_matrix.tp,
                "tn": t.confusion_matrix.tn,
                "fp": t.confusion_matrix.fp,
                "fn": t.confusion_matrix.fn,
                "peak_ram_mb": f"{t.peak_ram_mb:.2f}",
            }
        )
    return buf.getvalue().encode("utf-8")


def is_connection_refused_error(err: Exception) -> bool:
    """Return True if exception indicates unreachable local Ollama daemon or connection refusal."""
    import errno

    if isinstance(err, (ConnectionRefusedError, ConnectionResetError)):
        return True
    if isinstance(err, OSError) and err.errno in (errno.ECONNREFUSED, errno.ECONNRESET):
        return True

    err_type_name = type(err).__name__
    if "ConnectError" in err_type_name or "ConnectTimeout" in err_type_name:
        return True

    cause = getattr(err, "__cause__", None) or getattr(err, "__context__", None)
    if cause is not None and isinstance(cause, Exception) and is_connection_refused_error(cause):
        return True

    msg = str(err).lower()
    return (
        "connection refused" in msg
        or "errno 61" in msg
        or "errno 111" in msg
        or "failed to establish a new connection" in msg
        or "all connection attempts failed" in msg
    )


def execute_benchmark_safely(
    pdf_bytes: bytes,
    pdf_name: str,
    prompts: tuple[tuple[str, str], ...],
    models: tuple[str, ...],
    simulation_mode: bool,
    auto_fallback: bool = True,
) -> tuple[
    list[tuple[TrialTelemetry, TrialTelemetry]] | None,
    list[PairedStatisticalResult] | None,
    str | None,
]:
    """Execute student PDF benchmark with robust exception catching and diagnostic message.

    If live execution fails due to an unreachable local Ollama daemon or connection refusal,
    and auto_fallback is enabled, execution automatically falls back to Deterministic Simulation
    Mode (Fast Offline) and returns the completed benchmark pairs, statistics, and an informative
    warning notice instead of terminating with an error.
    """
    try:
        pairs, stats = run_student_pdf_benchmark(
            pdf_bytes=pdf_bytes,
            pdf_name=pdf_name,
            prompts=prompts,
            models=models,
            simulation_mode=simulation_mode,
        )
        return pairs, stats, None
    except Exception as err:
        if not simulation_mode and auto_fallback and is_connection_refused_error(err):
            try:
                fallback_pairs, fallback_stats = run_student_pdf_benchmark(
                    pdf_bytes=pdf_bytes,
                    pdf_name=pdf_name,
                    prompts=prompts,
                    models=models,
                    simulation_mode=True,
                )
                warning_msg = (
                    f"Notice: Local Ollama daemon is unreachable ({err}). "
                    "Benchmark automatically executed in "
                    "Deterministic Simulation Mode (Fast Offline) "
                    "so that your comparative evaluation could complete without interruption."
                )
                return fallback_pairs, fallback_stats, warning_msg
            except Exception as fallback_err:
                err = fallback_err

        return (
            None,
            None,
            (
                f"Benchmark execution encountered an error: {err}\n\n"
                "Tip: If Ollama is unreachable or model weights encounter environment mismatches, "
                "check 'Use Deterministic Simulation Mode (Fast Offline)' in the sidebar."
            ),
        )


# ---------------------------------------------------------------------------
# Theme + HTML rendering helpers
# ---------------------------------------------------------------------------

_LIGHT_TOKENS: dict[str, str] = {
    "bg": "#F7F8FA",
    "surface": "#FFFFFF",
    "surface-2": "#F1F3F6",
    "border": "#E3E6EB",
    "text": "#16191F",
    "muted": "#667085",
    "accent": "#3B5BDB",
    "accent-soft": "#E8EDFD",
    "accent-text": "#FFFFFF",
    "good": "#0F8A5F",
    "good-soft": "#E3F5EC",
    "bad": "#C93B3B",
    "bad-soft": "#FBE9E9",
    "warn": "#A8620A",
    "warn-soft": "#FCF1DE",
    "shadow": "0 1px 2px rgba(16,24,40,.05)",
}

_DARK_TOKENS: dict[str, str] = {
    "bg": "#0D1117",
    "surface": "#151A22",
    "surface-2": "#1C222C",
    "border": "#262D39",
    "text": "#E6E9EF",
    "muted": "#8A94A6",
    "accent": "#7C93F5",
    "accent-soft": "#212A4A",
    "accent-text": "#0D1117",
    "good": "#4CC38A",
    "good-soft": "#13281F",
    "bad": "#F07178",
    "bad-soft": "#2E1A1D",
    "warn": "#E5A851",
    "warn-soft": "#2B2214",
    "shadow": "0 1px 2px rgba(0,0,0,.4)",
}

_BASE_CSS = """
html, body, .stApp, [class*="st-"], button, input, textarea {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.material-icons, .material-symbols-rounded, [data-testid="stIconMaterial"],
[class*="material-symbols"] {
  font-family: 'Material Symbols Rounded' !important;
}

/* App shell */
.stApp, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text);
}
[data-testid="stHeader"] {
  background: transparent !important;
}
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
  display: none !important;
}
.block-container {
  padding-top: 2.5rem !important;
  padding-bottom: 4rem !important;
  max-width: 1180px;
}

/* Text */
.stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
  color: var(--text);
}
[data-testid="stCaptionContainer"], .stApp small {
  color: var(--muted) !important;
}
[data-testid="stSpinner"] * {
  color: var(--muted) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child {
  padding-top: 0.5rem;
}
[data-testid="stSidebarCollapseButton"] button, [data-testid="stExpandSidebarButton"] {
  color: var(--muted) !important;
}
[data-testid="stSidebar"] hr {
  border-color: var(--border);
  margin: 1.1rem 0;
}

/* Inputs */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div {
  background: var(--surface-2) !important;
  border-color: var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
}
[data-baseweb="input"] input, [data-baseweb="select"] input, [data-baseweb="select"] div {
  color: var(--text) !important;
  -webkit-text-fill-color: var(--text) !important;
}
[data-baseweb="input"] input::placeholder {
  color: var(--muted) !important;
  -webkit-text-fill-color: var(--muted) !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
[data-baseweb="select"] svg, [data-baseweb="input"] svg {
  fill: var(--muted) !important;
  color: var(--muted) !important;
}
[data-baseweb="tag"] {
  background: var(--accent-soft) !important;
  border-radius: 6px !important;
}
[data-baseweb="tag"] span, [data-baseweb="tag"] svg {
  color: var(--accent) !important;
  fill: var(--accent) !important;
  -webkit-text-fill-color: var(--accent) !important;
}
[data-baseweb="popover"] > div, [data-baseweb="menu"], [data-baseweb="popover"] ul {
  background: var(--surface) !important;
  border: 1px solid var(--border);
  border-radius: 8px !important;
}
[data-baseweb="menu"] li, [data-baseweb="popover"] li {
  color: var(--text) !important;
  background: transparent !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="popover"] li:hover,
[data-baseweb="popover"] li[aria-selected="true"] {
  background: var(--surface-2) !important;
}

/* Checkbox + toggle labels */
[data-testid="stCheckbox"] label span, [data-testid="stToggle"] label span {
  color: var(--text) !important;
}
[data-testid="stCheckbox"] [data-baseweb="checkbox"] > div:first-child {
  border-color: var(--border) !important;
  background: var(--surface-2) !important;
}
[data-testid="stCheckbox"] [data-baseweb="checkbox"] input:checked + div,
[data-testid="stCheckbox"] [aria-checked="true"] > div:first-child {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}
[data-testid="stToggle"] [data-baseweb="checkbox"] > div:first-child {
  background: var(--border) !important;
}
[data-testid="stToggle"] [aria-checked="true"] > div:first-child {
  background: var(--accent) !important;
}
[data-testid="stToggle"] [data-baseweb="checkbox"] > div:last-child {
  background: #fff !important;
}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
  background: var(--surface-2) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"] * {
  color: var(--muted) !important;
}
[data-testid="stFileUploaderDropzone"] button {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
}
[data-testid="stFileUploaderFile"] * {
  color: var(--text) !important;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 8px !important;
  font-weight: 500 !important;
  padding: 0.55rem 1rem !important;
  border: 1px solid var(--border) !important;
  background: var(--surface) !important;
  color: var(--text) !important;
  box-shadow: var(--shadow);
  transition: border-color .15s, background .15s;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}
.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: var(--accent-text) !important;
  width: 100%;
}
.stButton > button[kind="primary"]:hover {
  filter: brightness(1.08);
  color: var(--accent-text) !important;
}
.stButton > button p, .stDownloadButton > button p {
  color: inherit !important;
}
button:focus-visible {
  outline: 2px solid var(--accent) !important;
  outline-offset: 2px;
}

/* Custom components */
.vr-hero h1 {
  font-size: 1.75rem;
  font-weight: 650;
  letter-spacing: -0.02em;
  margin: 0 0 .35rem 0;
  padding: 0;
  color: var(--text);
}
.vr-hero p {
  font-size: .95rem;
  color: var(--muted) !important;
  margin: 0 0 1.75rem 0;
  max-width: 62ch;
  line-height: 1.55;
}
.vr-section {
  font-size: 1.05rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin: 2.25rem 0 .25rem 0;
  color: var(--text);
}
.vr-sub {
  font-size: .85rem;
  color: var(--muted);
  margin: 0 0 1rem 0;
}
.vr-side-title {
  font-size: 1.05rem;
  font-weight: 650;
  letter-spacing: -0.01em;
  margin: .25rem 0 .1rem 0;
  color: var(--text);
}
.vr-side-sub {
  font-size: .8rem;
  color: var(--muted);
  margin: 0 0 .5rem 0;
}
.vr-label {
  font-size: .8rem;
  font-weight: 600;
  color: var(--muted);
  margin: 1.1rem 0 .4rem 0;
}

.vr-pill {
  display: inline-flex;
  align-items: center;
  gap: .5rem;
  font-size: .8rem;
  font-weight: 500;
  padding: .3rem .65rem;
  border-radius: 999px;
  margin: .15rem 0 .6rem 0;
}
.vr-pill .dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
}
.vr-pill.ok {
  background: var(--good-soft);
  color: var(--good);
}
.vr-pill.off {
  background: var(--warn-soft);
  color: var(--warn);
}

.vr-note {
  font-size: .85rem;
  line-height: 1.5;
  padding: .75rem 1rem;
  border-radius: 10px;
  margin: 0 0 1rem 0;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text);
  white-space: pre-wrap;
}
.vr-note.warn {
  background: var(--warn-soft);
  border-color: transparent;
  color: var(--warn);
}
.vr-note.error {
  background: var(--bad-soft);
  border-color: transparent;
  color: var(--bad);
}

.vr-guide {
  background: var(--surface-alt);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: .75rem .9rem;
  font-size: .82rem;
  line-height: 1.5;
  color: var(--text);
  margin: .3rem 0 .7rem 0;
}
.vr-guide .title {
  font-weight: 600;
  margin-bottom: .35rem;
  color: var(--text);
}
.vr-guide ol {
  margin: 0;
  padding-left: 1.2rem;
}
.vr-guide li {
  margin-bottom: .35rem;
}

.vr-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}
@media (max-width: 1000px) {
  .vr-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
@media (max-width: 560px) {
  .vr-grid {
    grid-template-columns: 1fr;
  }
}
.vr-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.1rem 1.2rem;
  box-shadow: var(--shadow);
}
.vr-card .name {
  font-size: .8rem;
  color: var(--muted);
  font-weight: 500;
  line-height: 1.35;
  min-height: 2.15em;
}
.vr-card .value {
  font-size: 1.9rem;
  font-weight: 650;
  letter-spacing: -0.03em;
  margin: .35rem 0 .55rem 0;
  color: var(--text);
  font-variant-numeric: tabular-nums;
}
.vr-card .row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: .5rem;
  flex-wrap: wrap;
}
.vr-card .base {
  font-size: .75rem;
  color: var(--muted);
}
.vr-chip {
  font-size: .75rem;
  font-weight: 600;
  padding: .15rem .5rem;
  border-radius: 6px;
  font-variant-numeric: tabular-nums;
}
.vr-chip.good {
  background: var(--good-soft);
  color: var(--good);
}
.vr-chip.bad {
  background: var(--bad-soft);
  color: var(--bad);
}
.vr-chip.neutral {
  background: var(--surface-2);
  color: var(--muted);
}

.vr-compare {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
@media (max-width: 800px) {
  .vr-compare {
    grid-template-columns: 1fr;
  }
}
.vr-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.1rem 1.2rem;
  box-shadow: var(--shadow);
}
.vr-panel.verified {
  border-top: 3px solid var(--accent);
}
.vr-panel.base {
  border-top: 3px solid var(--border);
}
.vr-panel h4 {
  font-size: .95rem;
  font-weight: 600;
  margin: 0;
  padding: 0;
  color: var(--text);
}
.vr-panel .tag {
  font-size: .75rem;
  color: var(--muted);
  margin: .1rem 0 .8rem 0;
}
.vr-panel .body {
  font-size: .9rem;
  line-height: 1.65;
  color: var(--text);
  white-space: pre-wrap;
  min-height: 4rem;
}
.vr-panel .body.empty {
  color: var(--muted);
  font-style: italic;
}
.vr-stats {
  display: flex;
  flex-wrap: wrap;
  gap: .4rem;
  margin-top: 1rem;
  padding-top: .9rem;
  border-top: 1px solid var(--border);
}
.vr-stat {
  font-size: .75rem;
  background: var(--surface-2);
  color: var(--muted);
  padding: .2rem .55rem;
  border-radius: 6px;
  font-variant-numeric: tabular-nums;
}
.vr-stat b {
  color: var(--text);
  font-weight: 600;
}

.vr-table-wrap {
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--surface);
  box-shadow: var(--shadow);
}
table.vr-table {
  width: 100%;
  border-collapse: collapse;
  font-size: .82rem;
}
table.vr-table th {
  text-align: left;
  font-weight: 600;
  color: var(--muted);
  background: var(--surface-2);
  padding: .65rem .9rem;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
table.vr-table td {
  padding: .65rem .9rem;
  border-bottom: 1px solid var(--border);
  color: var(--text);
  vertical-align: top;
  font-variant-numeric: tabular-nums;
}
table.vr-table tr:last-child td {
  border-bottom: none;
}
table.vr-table td.wide {
  min-width: 260px;
  line-height: 1.5;
}
.vr-badge {
  display: inline-block;
  font-size: .72rem;
  font-weight: 600;
  padding: .15rem .5rem;
  border-radius: 6px;
  white-space: nowrap;
}
.vr-badge.good {
  background: var(--good-soft);
  color: var(--good);
}
.vr-badge.bad {
  background: var(--bad-soft);
  color: var(--bad);
}

.vr-empty {
  background: var(--surface);
  border: 1px dashed var(--border);
  border-radius: 12px;
  padding: 2rem 1.75rem;
}
.vr-empty h3 {
  font-size: 1.05rem;
  font-weight: 600;
  margin: 0 0 .5rem 0;
  color: var(--text);
}
.vr-empty p {
  font-size: .9rem;
  color: var(--muted) !important;
  margin: 0 0 .35rem 0;
  line-height: 1.6;
  max-width: 60ch;
}
"""


def build_theme_css(dark: bool) -> str:
    """Return a full <style> block for the requested theme."""
    tokens = _DARK_TOKENS if dark else _LIGHT_TOKENS
    root = "".join(f"--{k}: {v};" for k, v in tokens.items())
    scheme = "dark" if dark else "light"
    return f"<style>:root {{ {root} color-scheme: {scheme}; }}{_BASE_CSS}</style>"


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _section(title: str, subtitle: str = "") -> str:
    sub = f'<div class="vr-sub">{_esc(subtitle)}</div>' if subtitle else ""
    return f'<div class="vr-section">{_esc(title)}</div>{sub}'


def build_metric_cards_html(stats: list[PairedStatisticalResult]) -> str:
    """Render the four headline metrics as compact cards with a colored delta chip."""
    cards: list[str] = []
    for res in stats:
        delta = res.mean_delta
        lower_is_better = "latency" in res.metric_name.lower()
        if abs(delta) < 1e-12:
            tone, arrow = "neutral", "–"
        else:
            improved = delta < 0 if lower_is_better else delta > 0
            tone = "good" if improved else "bad"
            arrow = "▲" if delta > 0 else "▼"
        delta_txt = format_metric_value(delta, res.unit, signed=True)
        base_txt = format_metric_value(res.mean_baseline_a, res.unit)
        cards.append(
            f'<div class="vr-card" title="{_esc(METRIC_DESCRIPTIONS.get(res.metric_name, ""))}">'
            f'<div class="name">{_esc(res.metric_name)}</div>'
            f'<div class="value">{_esc(format_metric_value(res.mean_verified_b, res.unit))}</div>'
            f'<div class="row"><span class="vr-chip {tone}">{arrow} {_esc(delta_txt)}</span>'
            f'<span class="base">Base {_esc(base_txt)}</span></div></div>'
        )
    return f'<div class="vr-grid">{"".join(cards)}</div>'


_WIDE_COLUMNS = {
    "Atomic Sentence",
    "Prompt",
    "Description",
    "Prune Reason",
    "Flagged Claim",
    "Detail",
}


def build_table_html(rows: list[dict[str, object]]) -> str:
    """Render rows as a styled, horizontally scrollable HTML table."""
    if not rows:
        return '<div class="vr-note">No rows to display.</div>'
    headers = list(rows[0].keys())
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body_rows: list[str] = []
    for row in rows:
        cells: list[str] = []
        for h in headers:
            text = str(row.get(h, ""))
            if text.startswith("SURVIVED"):
                inner = f'<span class="vr-badge good">{_esc(text)}</span>'
            elif text.startswith("PRUNED"):
                inner = f'<span class="vr-badge bad">{_esc(text)}</span>'
            else:
                inner = _esc(text)
            cls = ' class="wide"' if h in _WIDE_COLUMNS else ""
            cells.append(f"<td{cls}>{inner}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<div class="vr-table-wrap"><table class="vr-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def build_output_panel_html(trial: TrialTelemetry, *, verified: bool) -> str:
    """Render one condition's output and its four metrics as a panel."""
    title = "Verification-driven RAG" if verified else "Base RAG"
    tag = (
        "Condition B · unsupported claims pruned" if verified else "Condition A · unverified output"
    )
    fallback = "All sentences were pruned by the verification gate." if verified else "No output."
    text = trial.output_summary or ""
    body_cls = "body" if text else "body empty"
    stats = [
        ("FActScore", f"{trial.factual_precision_pct:.1f}%"),
        ("NLI", f"{trial.entailment_consistency_pct:.1f}%"),
        ("Citations", f"{trial.citation_accuracy_pct:.1f}%"),
        ("Latency", f"{trial.computational_latency_sec:.4f}s"),
    ]
    chips = "".join(f'<span class="vr-stat">{_esc(k)} <b>{_esc(v)}</b></span>' for k, v in stats)
    return (
        f'<div class="vr-panel {"verified" if verified else "base"}">'
        f'<h4>{_esc(title)}</h4><div class="tag">{_esc(tag)}</div>'
        f'<div class="{body_cls}">{_esc(text or fallback)}</div>'
        f'<div class="vr-stats">{chips}</div></div>'
    )


def build_ollama_troubleshooting_html(report: OllamaHealthReport) -> str:
    """Render structured troubleshooting guide for local Ollama daemon status."""
    if report.is_online and not report.missing_approved_models:
        return (
            '<div class="vr-guide">'
            '<div class="title">Daemon Ready</div>'
            f"<div>Ollama is reachable at <code>{_esc(report.base_url)}</code> and "
            f"all {len(APPROVED_MODELS)} models are ready.</div>"
            "</div>"
        )

    title = "How to Fix: Ollama Offline" if not report.is_online else "Missing Approved Models"
    steps_html = "".join(f"<li>{_esc(step)}</li>" for step in report.troubleshooting_steps)
    return (
        '<div class="vr-guide">'
        f'<div class="title">{_esc(title)}</div>'
        f'<div style="margin-bottom:.4rem;color:var(--muted);">{_esc(report.message)}</div>'
        f"<ol>{steps_html}</ol>"
        "</div>"
    )


def render_streamlit_app() -> None:  # pragma: no cover
    """Render the minimal Streamlit comparative benchmark UI."""
    import streamlit as st

    st.set_page_config(
        page_title="vrag · Verification-Driven RAG Benchmark",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.markdown(
            '<div class="vr-side-title">vrag</div>'
            '<div class="vr-side-sub">Verification-driven RAG benchmark</div>',
            unsafe_allow_html=True,
        )
        dark_mode = st.toggle("Dark mode", value=False, key="vr_dark_mode")

    st.markdown(build_theme_css(dark_mode), unsafe_allow_html=True)

    with st.sidebar:
        st.divider()
        st.markdown('<div class="vr-label">1. Document</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Student or academic paper",
            type=["pdf", "txt"],
            help="Upload any PDF to test how all 3 models hallucinate on it.",
            label_visibility="collapsed",
        )
        st.caption("Leave empty to test with a built-in sample paper.")

        st.markdown(
            '<div class="vr-label">2. Models (run sequentially)</div>', unsafe_allow_html=True
        )
        selected_models = st.multiselect(
            "Local Ollama models (run sequentially)",
            options=list(APPROVED_MODELS),
            default=list(APPROVED_MODELS),
            format_func=format_model_label,
            label_visibility="collapsed",
        )

        report = diagnose_ollama()
        ollama_online = report.is_online
        if ollama_online:
            st.markdown(
                '<div class="vr-pill ok">'
                '<span class="dot"></span>Ollama online · localhost:11434</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="vr-pill off">'
                '<span class="dot"></span>Ollama offline · localhost:11434</div>',
                unsafe_allow_html=True,
            )

        if not ollama_online or report.missing_approved_models:
            guide_label = (
                "How to fix: Ollama offline"
                if not ollama_online
                else f"Missing {len(report.missing_approved_models)} model(s)"
            )
            with st.expander(guide_label, expanded=False):
                st.markdown(
                    build_ollama_troubleshooting_html(report),
                    unsafe_allow_html=True,
                )

        ollama_ready = ollama_online and not report.missing_approved_models
        sim_mode = st.checkbox(
            "Offline simulation mode",
            value=not ollama_ready,
            help="Runs locally without Ollama using realistic simulated model outputs.",
        )
        if not ollama_online and not sim_mode:
            st.caption(
                "Ollama is offline. Live runs will fall back to simulation if the connection "
                "is refused."
            )
        elif report.missing_approved_models and not sim_mode:
            st.caption(
                "One or more approved models are not pulled yet. Pull them before running live."
            )

        st.markdown('<div class="vr-label">3. Prompt (optional)</div>', unsafe_allow_html=True)
        custom_query = st.text_input(
            "Custom prompt",
            value="",
            placeholder="Leave blank for 4-prompt suite",
            label_visibility="collapsed",
        )
        st.write("")
        run_btn = st.button("Run benchmark", type="primary")

    st.markdown(
        '<div class="vr-hero"><h1>Hallucination Benchmark</h1>'
        "<p>Compare standard RAG against verification-driven RAG across Llama-3.1-8B, "
        "Mistral-7B, and Qwen-2.5-7B on any uploaded PDF.</p></div>",
        unsafe_allow_html=True,
    )

    if run_btn:
        if uploaded_file is not None:
            pdf_bytes = uploaded_file.getvalue()
            pdf_name = uploaded_file.name
        else:
            pdf_name = "sample_student_thesis.pdf"
            pdf_bytes = (
                b"Bang et al. (2025) evaluated 7B to 8B open-source models on HalluLens and found "
                b"baseline hallucination rates between 48% and 85%. "
                b"Kadavath et al. (2022) showed smaller models act overconfident when wrong.\f"
                b"Wang et al. (2025) proposed dynamic semantic chunking using cosine similarity. "
                b"Qian et al. (2025) introduced VeriCite to prune unentailed claims via NLI."
            )

        models_tuple = tuple(selected_models) if selected_models else tuple(APPROVED_MODELS)
        if custom_query.strip():
            prompts = (
                ("custom_1", custom_query.strip()),
                ("custom_2", f"{custom_query.strip()} (Provide detailed methodology citations.)"),
            )
        else:
            prompts = DEFAULT_STUDENT_BENCHMARK_PROMPTS

        with st.spinner("Running sequential Condition A vs. Condition B benchmark..."):
            pairs, stats, error_msg = execute_benchmark_safely(
                pdf_bytes=pdf_bytes,
                pdf_name=pdf_name,
                prompts=prompts,
                models=models_tuple,
                simulation_mode=sim_mode,
            )

        st.session_state["vrag_results"] = {
            "pairs": pairs,
            "stats": stats,
            "pdf_name": pdf_name,
            "error_msg": error_msg,
        }

    saved = st.session_state.get("vrag_results")
    if not saved:
        st.markdown(
            '<div class="vr-empty"><h3>Ready to test</h3>'
            "<p>1. Upload any academic PDF in the sidebar (or use the built-in sample).</p>"
            "<p>2. Keep all 3 models selected to compare how each one hallucinates.</p>"
            "<p>3. Click <b>Run benchmark</b> to see fabricated claims and how NLI verification "
            "prunes them.</p></div>",
            unsafe_allow_html=True,
        )
        return

    pairs = saved["pairs"]
    stats = saved["stats"]
    pdf_name = saved["pdf_name"]
    error_msg = saved["error_msg"]

    if pairs is None or stats is None:
        st.markdown(
            f'<div class="vr-note error">{_esc(error_msg or "Benchmark execution failed.")}</div>',
            unsafe_allow_html=True,
        )
        return

    if error_msg:
        st.markdown(f'<div class="vr-note warn">{_esc(error_msg)}</div>', unsafe_allow_html=True)

    st.markdown(
        _section("Overview", f"{pdf_name} · mean across {len(pairs)} paired trials"),
        unsafe_allow_html=True,
    )
    st.markdown(build_metric_cards_html(stats), unsafe_allow_html=True)

    st.markdown(
        _section(
            "Hallucinations by model",
            "Condition A (Base RAG) vs. Condition B (Verification-Driven RAG)",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(build_table_html(build_model_hallucination_rows(pairs)), unsafe_allow_html=True)

    st.markdown(
        _section(
            "Inspect a trial",
            "Choose any model and prompt to see exact hallucinations and pruned claims",
        ),
        unsafe_allow_html=True,
    )
    trial_labels = [
        f"{format_model_label(a.model_name)} · {a.prompt[:65]}{'...' if len(a.prompt) > 65 else ''}"
        for a, _b in pairs
    ]
    selected_idx = st.selectbox(
        "Select trial to inspect",
        options=list(range(len(pairs))),
        format_func=lambda idx: trial_labels[idx],
        label_visibility="collapsed",
    )
    sel_a, sel_b = pairs[selected_idx]

    st.markdown(
        '<div class="vr-compare">'
        + build_output_panel_html(sel_a, verified=False)
        + build_output_panel_html(sel_b, verified=True)
        + "</div>",
        unsafe_allow_html=True,
    )

    evidence_rows = build_base_hallucination_evidence(sel_a)
    with st.expander(
        f"Flagged hallucinations in Base RAG ({len(evidence_rows)} caught)", expanded=True
    ):
        if evidence_rows:
            st.markdown(build_table_html(evidence_rows), unsafe_allow_html=True)
        else:
            st.caption("No hallucinations detected in Base RAG for this trial.")

    with st.expander("Sentence-by-sentence verification log (Condition B)", expanded=False):
        st.markdown(build_table_html(build_claim_audit_rows(sel_b)), unsafe_allow_html=True)

    with st.expander("Statistical significance & per-trial telemetry", expanded=False):
        st.markdown(
            build_table_html(build_four_metrics_summary_rows(stats)), unsafe_allow_html=True
        )
        st.write("")
        st.markdown(build_table_html(build_trial_comparison_rows(pairs)), unsafe_allow_html=True)

    st.write("")
    all_trials = [t for pair in pairs for t in pair]
    dl_col1, dl_col2, _spacer = st.columns([1, 1, 2])
    with dl_col1:
        st.download_button(
            "Download telemetry CSV",
            data=_trials_to_csv_bytes(all_trials),
            file_name="vrag_telemetry.csv",
            mime="text/csv",
        )
    with dl_col2:
        json_payload = json.dumps(
            {
                "trials": [asdict(t) for t in all_trials],
                "statistical_tests": [asdict(s) for s in stats],
            },
            indent=2,
        ).encode("utf-8")
        st.download_button(
            "Download summary JSON",
            data=json_payload,
            file_name="vrag_summary.json",
            mime="application/json",
        )


if __name__ == "__main__":  # pragma: no cover
    render_streamlit_app()
