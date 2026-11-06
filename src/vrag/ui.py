"""Lightweight Streamlit UI for Student PDF Benchmarking across the 4 Core Thesis Metrics."""

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
from vrag.generator import check_ollama_status
from vrag.models import (
    APPROVED_MODELS,
    DEFAULT_STUDENT_BENCHMARK_PROMPTS,
    TrialTelemetry,
)

METRIC_DESCRIPTIONS: dict[str, str] = {
    "Factual Precision (FActScore)": (
        "Percentage of generated claims actually supported by the uploaded PDF (%)."
    ),
    "Entailment Consistency (NLI Score)": (
        "Accuracy of classifying sentences as supported vs. hallucinated (TP vs. FP) (%)."
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
                "Paired t-stat": f"{res.t_statistic:.4f}",
                "p-value": f"{res.t_pvalue:.4e}",
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
                "NLI Label": claim.nli_label.value,
                "NLI Score (%)": f"{claim.entailment_score * 100.0:.1f}%",
                "Gate Decision": "SURVIVED (TP)" if claim.survived else "PRUNED (TN)",
                "Prune Reason": claim.prune_reason or "Verified & Retained",
            }
        )
    return rows


def _trials_to_csv_bytes(trials: list[TrialTelemetry]) -> bytes:
    buf = io.StringIO()
    fieldnames = [
        "trial_id",
        "model_name",
        "mode",
        "factual_precision_pct",
        "entailment_consistency_pct",
        "citation_accuracy_pct",
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

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


def render_streamlit_app() -> None:  # pragma: no cover
    """Render the minimal Streamlit comparative benchmark UI."""
    import streamlit as st

    st.set_page_config(
        page_title="vrag — Verification-Driven RAG Benchmark",
        layout="wide",
    )
    st.title("Verification-Driven RAG vs. Base RAG — Comparative Benchmark")
    st.caption(
        "Thesis Harness: Evaluating Factual Precision (FActScore %), Entailment Consistency "
        "(NLI Score %), Citation Accuracy (%), and Computational Latency (s)."
    )

    with st.sidebar:
        st.header("1. Student PDF Input")
        uploaded_file = st.file_uploader(
            "Upload Student / Academic PDF",
            type=["pdf", "txt"],
            help="Upload a student paper PDF to run the 4-metric comparative benchmark.",
        )

        st.header("2. Benchmark Configuration")
        selected_models = st.multiselect(
            "Local Ollama Models (Sequential Execution)",
            options=list(APPROVED_MODELS),
            default=[APPROVED_MODELS[0]],
        )
        ollama_online = check_ollama_status()
        if ollama_online:
            st.success("🟢 Local Ollama daemon is online (http://localhost:11434)")
        else:
            st.warning("⚠️ Local Ollama daemon is offline (http://localhost:11434)")

        sim_mode = st.checkbox(
            "Use Deterministic Simulation Mode (Fast Offline)",
            value=not ollama_online,
            help="When enabled, simulates LLM responses grounded on your uploaded PDF chunks.",
        )
        if not ollama_online and not sim_mode:
            st.info(
                "💡 Notice: Ollama daemon is offline. Running live mode will automatically "
                "fall back to Simulation Mode if connection is refused."
            )
        custom_query = st.text_input(
            "Optional Custom Prompt (leave blank to run 4-prompt benchmark suite)",
            value="",
        )
        run_btn = st.button("Run Comparative Benchmark", type="primary")

    if not run_btn:
        st.info(
            "Upload a student PDF in the sidebar (or use the built-in sample paper) and click "
            "**Run Comparative Benchmark** to compare Base RAG vs. Verification-Driven RAG."
        )
        return

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        pdf_name = uploaded_file.name
    else:
        pdf_name = "sample_student_thesis.pdf"
        pdf_bytes = (
            b"Vaswani et al. (2017) introduced the Transformer architecture based on "
            b"self-attention mechanisms.\f"
            b"Lewis et al. (2020) proposed Retrieval-Augmented Generation to ground LLMs on "
            b"retrieved passages and reduce factual hallucinations."
        )

    models_tuple = tuple(selected_models) if selected_models else (APPROVED_MODELS[0],)
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

    if pairs is None or stats is None:
        st.error(error_msg or "Benchmark execution failed.")
        return

    if error_msg:
        st.warning(error_msg)

    st.subheader("1. Core 4-Metric Comparative Results (Base RAG vs. Verification-Driven RAG)")
    cols = st.columns(4)
    for col, res in zip(cols, stats):
        col.metric(
            label=res.metric_name,
            value=format_metric_value(res.mean_verified_b, res.unit),
            delta=format_metric_value(res.mean_delta, res.unit, signed=True),
        )

    st.table(build_four_metrics_summary_rows(stats))

    st.subheader("2. Side-by-Side Output & Verification Gate Audit (First Trial)")
    first_a, first_b = pairs[0]
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Condition A: Base RAG (Unverified Output)")
        st.warning(first_a.output_summary or "(Empty output)")
        st.caption(
            f"FActScore: {first_a.factual_precision_pct:.1f}% | "
            f"NLI Consistency: {first_a.entailment_consistency_pct:.1f}% | "
            f"Citation Accuracy: {first_a.citation_accuracy_pct:.1f}% | "
            f"Latency: {first_a.computational_latency_sec:.4f}s"
        )
    with col_b:
        st.markdown("#### Condition B: Verification-Driven RAG (Pruned Output)")
        st.success(first_b.output_summary or "(All sentences pruned by verification gate)")
        st.caption(
            f"FActScore: {first_b.factual_precision_pct:.1f}% | "
            f"NLI Consistency: {first_b.entailment_consistency_pct:.1f}% | "
            f"Citation Accuracy: {first_b.citation_accuracy_pct:.1f}% | "
            f"Latency: {first_b.computational_latency_sec:.4f}s"
        )

    st.markdown("##### Atomic Claim Verification & Pruning Log (Condition B)")
    st.dataframe(build_claim_audit_rows(first_b), use_container_width=True)

    st.subheader("3. Per-Trial Breakdown & Export")
    st.dataframe(build_trial_comparison_rows(pairs), use_container_width=True)

    all_trials = [t for pair in pairs for t in pair]
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "Download Telemetry CSV",
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
            "Download Statistical Summary JSON",
            data=json_payload,
            file_name="vrag_summary.json",
            mime="application/json",
        )


if __name__ == "__main__":  # pragma: no cover
    render_streamlit_app()
