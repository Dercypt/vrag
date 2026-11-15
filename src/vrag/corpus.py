"""Thesis Writing 1 curated academic PDF benchmark corpus loader and ground-truth oracle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vrag.evaluator import (
    ComparativeEvaluator,
    PairedStatisticalResult,
    _build_simulation_evaluator,
    compute_suite_statistics,
)
from vrag.models import APPROVED_MODELS, GroundTruthSpec, TrialTelemetry
from vrag.verifier import (
    GroundTruthFn,
    content_tokens,
    extract_author_year_citations,
    numeric_tokens,
)

DEFAULT_CORPUS_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "benchmark_corpus"


@dataclass(frozen=True)
class BenchmarkCorpusItem:
    """Single curated benchmark prompt paired with its digital PDF and ground-truth spec."""

    item_id: str
    pdf_filename: str
    pdf_bytes: bytes
    benchmark_source: str
    query: str
    ground_truth: GroundTruthSpec


# --- Ground-Truth Claim Oracle ---


def build_ground_truth_oracle(spec: GroundTruthSpec) -> GroundTruthFn:
    """Create a deterministic claim evaluator from a curated GroundTruthSpec."""
    valid_cites = {(a.lower(), y.lower()) for a, y in spec.valid_citations}
    supported_sets = [
        (content_tokens(c), numeric_tokens(c)) for c in spec.supported_claims if c.strip()
    ]
    trap_sets = [
        content_tokens(t) for t in (*spec.intrinsic_traps, *spec.extrinsic_traps) if t.strip()
    ]

    def _oracle(claim: str) -> bool:
        claim_clean = claim.strip()
        if not claim_clean:
            return False

        # Reject fabricated or mismatched author-year pairs immediately when valid_cites exist.
        if valid_cites and any(
            (a.lower(), y.lower()) not in valid_cites
            for a, y in extract_author_year_citations(claim_clean)
        ):
            return False

        claim_tokens = content_tokens(claim_clean)
        if not claim_tokens:
            return False

        best_trap = max(
            (len(claim_tokens & t) / len(claim_tokens) for t in trap_sets if t),
            default=0.0,
        )
        claim_nums = numeric_tokens(claim_clean)
        best_support = max(
            (
                len(claim_tokens & sup_toks) / len(claim_tokens)
                for sup_toks, sup_nums in supported_sets
                if sup_toks and not (claim_nums and sup_nums and not claim_nums.issubset(sup_nums))
            ),
            default=0.0,
        )
        if best_trap >= 0.65 and best_trap >= best_support:
            return False
        return best_support >= 0.45

    return _oracle


# --- Curated PDF Corpus Loader ---


def _render_academic_pdf_bytes(title: str, pages: list[str]) -> bytes:
    """Render clean multi-page digital PDF bytes via PyMuPDF (fitz)."""
    import fitz  # type: ignore[import-untyped]

    with fitz.open() as doc:
        for page_idx, body in enumerate(pages, start=1):
            page = doc.new_page(width=595, height=842)
            text_block = f"{title}.\n\n{body}" if page_idx == 1 else body
            page.insert_textbox(
                fitz.Rect(54, 60, 541, 780), text_block, fontsize=11, fontname="helv"
            )
        return bytes(doc.tobytes())


def _read_manifest(base_dir: Path) -> dict[str, Any]:
    return dict(json.loads((base_dir / "manifest.json").read_text(encoding="utf-8")))


def _ensure_pdfs_from_manifest(base_dir: Path, data: dict[str, Any]) -> list[Path]:
    pdfs_dir = base_dir / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for doc_entry in data.get("documents", []):
        pdf_path = pdfs_dir / str(doc_entry["pdf_filename"])
        if not pdf_path.exists():
            pdf_path.write_bytes(
                _render_academic_pdf_bytes(
                    title=str(doc_entry["title"]),
                    pages=[str(p) for p in doc_entry["pages"]],
                )
            )
        written.append(pdf_path)
    return written


def ensure_corpus_pdfs(corpus_dir: Path | str = DEFAULT_CORPUS_DIR) -> list[Path]:
    """Ensure every document in manifest.json exists as a real digital PDF on disk."""
    base_dir = Path(corpus_dir)
    return _ensure_pdfs_from_manifest(base_dir, _read_manifest(base_dir))


def load_benchmark_corpus(
    corpus_dir: Path | str = DEFAULT_CORPUS_DIR,
) -> list[BenchmarkCorpusItem]:
    """Load all benchmark items and real PDFs from the curated corpus manifest."""
    base_dir = Path(corpus_dir)
    data = _read_manifest(base_dir)
    _ensure_pdfs_from_manifest(base_dir, data)
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
                (str(a), str(y)) for a, y in doc_entry.get("valid_citations", [])
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


# --- Curated Corpus Benchmark Runner ---


def run_curated_corpus_benchmark(
    corpus_dir: Path | str = DEFAULT_CORPUS_DIR,
    models: tuple[str, ...] = APPROVED_MODELS,
    top_k: int = 3,
    simulation_mode: bool = False,
) -> tuple[list[tuple[TrialTelemetry, TrialTelemetry]], list[PairedStatisticalResult]]:
    """Execute the curated Thesis Writing 1 benchmark suite with ground-truth claim oracles."""
    items = load_benchmark_corpus(corpus_dir)
    sweep_docs: list[tuple[str, bytes, str, GroundTruthFn | None]] = [
        (item.item_id, item.pdf_bytes, item.query, build_ground_truth_oracle(item.ground_truth))
        for item in items
    ]
    evaluator = _build_simulation_evaluator() if simulation_mode else ComparativeEvaluator()
    with evaluator:
        pairs = evaluator.run_sequential_model_sweep(
            documents=sweep_docs,
            models=models,
            top_k=top_k,
        )
    return pairs, compute_suite_statistics(pairs)
