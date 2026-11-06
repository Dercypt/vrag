"""Thesis Writing 1 curated academic PDF benchmark corpus loader and ground-truth oracle."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from vrag.evaluator import (
    ComparativeEvaluator,
    PairedStatisticalResult,
    _build_simulation_evaluator,
    compute_suite_statistics,
)
from vrag.models import APPROVED_MODELS, GroundTruthSpec, TrialTelemetry
from vrag.verifier import GroundTruthFn, extract_author_year_citations

DEFAULT_CORPUS_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "benchmark_corpus"

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
    }
)


@dataclass(frozen=True)
class BenchmarkCorpusItem:
    """Single curated benchmark prompt paired with its digital PDF and ground-truth spec."""

    item_id: str
    pdf_filename: str
    pdf_bytes: bytes
    benchmark_source: str
    query: str
    ground_truth: GroundTruthSpec


def _content_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zà-öø-ÿ0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def build_ground_truth_oracle(spec: GroundTruthSpec) -> GroundTruthFn:
    """Create a deterministic claim evaluator from a curated GroundTruthSpec."""
    valid_cites = {(a.lower(), y.lower()) for a, y in spec.valid_citations}
    supported_sets = [
        (_content_tokens(c), _numeric_tokens(c)) for c in spec.supported_claims if c.strip()
    ]
    trap_sets = [
        _content_tokens(t) for t in (*spec.intrinsic_traps, *spec.extrinsic_traps) if t.strip()
    ]

    def _oracle(claim: str) -> bool:
        claim_clean = claim.strip()
        if not claim_clean:
            return False

        cited_pairs = extract_author_year_citations(claim_clean)
        if valid_cites and cited_pairs:
            for author, year in cited_pairs:
                if (author.lower(), year.lower()) not in valid_cites:
                    return False

        claim_tokens = _content_tokens(claim_clean)
        if not claim_tokens:
            return False

        best_trap_overlap = 0.0
        for trap_tokens in trap_sets:
            if trap_tokens:
                overlap = len(claim_tokens & trap_tokens) / len(claim_tokens)
                best_trap_overlap = max(best_trap_overlap, overlap)

        claim_nums = _numeric_tokens(claim_clean)
        best_support_overlap = 0.0
        for sup_tokens, sup_nums in supported_sets:
            if not sup_tokens:
                continue
            if claim_nums and sup_nums and not claim_nums.issubset(sup_nums):
                continue
            overlap = len(claim_tokens & sup_tokens) / len(claim_tokens)
            best_support_overlap = max(best_support_overlap, overlap)

        if best_trap_overlap >= 0.65 and best_trap_overlap >= best_support_overlap:
            return False
        return best_support_overlap >= 0.45

    return _oracle


def _render_academic_pdf_bytes(title: str, pages: list[str]) -> bytes:
    """Render clean multi-page digital PDF bytes via PyMuPDF (fitz)."""
    import fitz  # type: ignore[import-untyped]

    doc = fitz.open()
    for page_idx, body in enumerate(pages, start=1):
        page = doc.new_page(width=595, height=842)
        header = f"{title}." if page_idx == 1 else ""
        text_block = f"{header}\n\n{body}" if header else body
        rect = fitz.Rect(54, 60, 541, 780)
        page.insert_textbox(rect, text_block, fontsize=11, fontname="helv")
    pdf_bytes: bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def ensure_corpus_pdfs(corpus_dir: Path | str = DEFAULT_CORPUS_DIR) -> list[Path]:
    """Ensure every document in manifest.json exists as a real digital PDF on disk."""
    base_dir = Path(corpus_dir)
    manifest_path = base_dir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    pdfs_dir = base_dir / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    for doc_entry in data.get("documents", []):
        pdf_path = pdfs_dir / str(doc_entry["pdf_filename"])
        if not pdf_path.exists():
            pdf_bytes = _render_academic_pdf_bytes(
                title=str(doc_entry["title"]),
                pages=[str(p) for p in doc_entry["pages"]],
            )
            pdf_path.write_bytes(pdf_bytes)
        written_paths.append(pdf_path)
    return written_paths


def load_benchmark_corpus(
    corpus_dir: Path | str = DEFAULT_CORPUS_DIR,
) -> list[BenchmarkCorpusItem]:
    """Load all benchmark items and real PDFs from the curated corpus manifest."""
    base_dir = Path(corpus_dir)
    ensure_corpus_pdfs(base_dir)
    manifest_path = base_dir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    pdfs_dir = base_dir / "pdfs"

    items: list[BenchmarkCorpusItem] = []
    for doc_entry in data.get("documents", []):
        pdf_filename = str(doc_entry["pdf_filename"])
        pdf_bytes = (pdfs_dir / pdf_filename).read_bytes()
        gt_spec = GroundTruthSpec(
            supported_claims=tuple(str(x) for x in doc_entry.get("supported_claims", [])),
            intrinsic_traps=tuple(str(x) for x in doc_entry.get("intrinsic_traps", [])),
            extrinsic_traps=tuple(str(x) for x in doc_entry.get("extrinsic_traps", [])),
            valid_citations=tuple(
                (str(pair[0]), str(pair[1])) for pair in doc_entry.get("valid_citations", [])
            ),
        )
        for prompt_entry in doc_entry.get("prompts", []):
            items.append(
                BenchmarkCorpusItem(
                    item_id=str(prompt_entry["item_id"]),
                    pdf_filename=pdf_filename,
                    pdf_bytes=pdf_bytes,
                    benchmark_source=str(prompt_entry.get("benchmark_source", "Curated")),
                    query=str(prompt_entry["query"]),
                    ground_truth=gt_spec,
                )
            )
    return items


def run_curated_corpus_benchmark(
    corpus_dir: Path | str = DEFAULT_CORPUS_DIR,
    models: tuple[str, ...] = APPROVED_MODELS,
    top_k: int = 3,
    simulation_mode: bool = False,
) -> tuple[list[tuple[TrialTelemetry, TrialTelemetry]], list[PairedStatisticalResult]]:
    """Execute the curated Thesis Writing 1 benchmark suite with ground-truth claim oracles."""
    items = load_benchmark_corpus(corpus_dir)
    evaluator = _build_simulation_evaluator() if simulation_mode else ComparativeEvaluator()
    sweep_docs: list[tuple[str, bytes, str, GroundTruthFn | None]] = [
        (
            item.item_id,
            item.pdf_bytes,
            item.query,
            build_ground_truth_oracle(item.ground_truth),
        )
        for item in items
    ]
    pairs = evaluator.run_sequential_model_sweep(
        documents=sweep_docs,
        models=models,
        top_k=top_k,
    )
    stats = compute_suite_statistics(pairs)
    return pairs, stats
