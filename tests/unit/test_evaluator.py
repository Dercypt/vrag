"""Unit tests for ComparativeEvaluator end-to-end execution, RAM telemetry, and CSV/JSON export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from vrag.evaluator import (
    ComparativeEvaluator,
    compute_suite_statistics,
    export_summary_json,
    export_telemetry_csv,
    get_peak_ram_mb,
)
from vrag.generator import OllamaGenerator
from vrag.models import NLILabel, NLIScore, PipelineMode


def test_get_peak_ram_mb_returns_positive_value() -> None:
    ram = get_peak_ram_mb()
    assert ram > 0.0


def test_comparative_evaluator_runs_paired_trial_and_exports_csv_json(tmp_path: Path) -> None:
    pdf_bytes = (
        b"Vaswani et al. (2017) introduced the Transformer architecture using self-attention. "
        b"Self-attention computes pairwise token interactions in parallel.\f"
        b"Lewis et al. (2020) proposed Retrieval-Augmented Generation for knowledge-intensive NLP."
    )

    def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.5] if "Transformer" in t else [0.5, 1.0] for t in texts]

    def fake_http_post(_url: str, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("keep_alive") == 0:
            return {"done": True}
        prompt_text = str(payload.get("prompt", ""))
        if "VERIFIABLE SUMMARY:" in prompt_text:
            # Condition B draft: 1 valid entailed sentence + 1 fabricated citation sentence
            return {
                "response": (
                    "Vaswani et al. (2017) introduced the Transformer architecture [C1]. "
                    "According to (FakeAuthor, 2025), RNNs are faster than attention [C1]."
                )
            }
        # Condition A unverified draft: 1 valid sentence + 1 unentailed hallucination
        return {
            "response": (
                "Vaswani et al. (2017) introduced the Transformer architecture. "
                "Transformers completely eliminate all weights and parameters."
            )
        }

    def fake_nli(_premise: str, hypothesis: str) -> NLIScore:
        if "eliminate all weights" in hypothesis or "RNNs are faster" in hypothesis:
            return NLIScore(
                label=NLILabel.CONTRADICTION,
                entailment_prob=0.03,
                contradiction_prob=0.94,
            )
        return NLIScore(label=NLILabel.ENTAILMENT, entailment_prob=0.96)

    generator = OllamaGenerator(http_post_fn=fake_http_post)
    evaluator = ComparativeEvaluator(
        generator=generator,
        embed_fn=fake_embed,
        nli_scorer=fake_nli,
    )

    pair_1 = evaluator.run_paired_trial(
        trial_id="trial_01",
        pdf_bytes=pdf_bytes,
        query="Summarize the Transformer and RAG papers.",
        model_name="llama3.1:8b-instruct-q4_K_M",
        top_k=2,
    )
    pair_2 = evaluator.run_paired_trial(
        trial_id="trial_02",
        pdf_bytes=pdf_bytes,
        query="Explain self-attention mechanisms.",
        model_name="llama3.1:8b-instruct-q4_K_M",
        top_k=2,
    )

    trial_a, trial_b = pair_1
    assert trial_a.mode == PipelineMode.BASELINE_A
    assert trial_b.mode == PipelineMode.VERIFIED_B
    assert trial_a.latencies.t_verify == 0.0
    assert trial_b.latencies.t_verify >= 0.0
    # Condition A has 1 TP and 1 FP -> FActScore = 0.5
    assert trial_a.atomic_factscore == 0.5
    # Condition B pruned the fabricated claim -> 1 TP, 1 TN, 0 FP -> FActScore = 1.0
    assert trial_b.atomic_factscore == 1.0
    assert "FakeAuthor" not in trial_b.output_summary

    stats = compute_suite_statistics([pair_1, pair_2])
    assert [s.metric_name for s in stats] == [
        "Factual Precision (FActScore)",
        "Entailment Consistency (NLI Score)",
        "Citation Accuracy (Precision/Recall)",
        "Computational Latency",
    ]
    assert [s.unit for s in stats] == ["%", "%", "%", "s"]
    factscore_stat = stats[0]
    assert factscore_stat.mean_baseline_a == 50.0
    assert factscore_stat.mean_verified_b == 100.0
    assert factscore_stat.mean_delta == 50.0

    csv_path = tmp_path / "telemetry.csv"
    json_path = tmp_path / "summary.json"
    all_trials = [trial_a, trial_b, pair_2[0], pair_2[1]]
    export_telemetry_csv(all_trials, csv_path)
    export_summary_json(all_trials, stats, json_path)

    with csv_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert rows[0]["trial_id"] == "trial_01"
    assert rows[1]["mode"] == "condition_b_verified"
    assert rows[1]["factual_precision_pct"] == "100.00"

    summary_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(summary_data["trials"]) == 4
    assert len(summary_data["statistical_tests"]) == 4
