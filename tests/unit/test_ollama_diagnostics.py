"""Unit tests for local Ollama health diagnostics and troubleshooting helpers."""

from __future__ import annotations

from typing import Any

import pytest

from vrag.generator import OllamaHealthReport, diagnose_ollama
from vrag.models import APPROVED_MODELS


def test_diagnose_ollama_when_offline_returns_actionable_report() -> None:
    """When daemon is unreachable, diagnose_ollama must return offline report with fix steps."""

    def offline_get(_url: str) -> dict[str, Any]:
        raise ConnectionRefusedError("[Errno 61] Connection refused")

    report = diagnose_ollama(
        base_url="http://127.0.0.1:11434",
        http_get_fn=offline_get,
    )
    assert isinstance(report, OllamaHealthReport)
    assert report.is_online is False
    assert report.base_url == "http://127.0.0.1:11434"
    assert report.installed_models == ()
    assert report.missing_approved_models == APPROVED_MODELS
    assert "unreachable" in report.message.lower()
    assert any("ollama serve" in step for step in report.troubleshooting_steps)
    assert any("Simulation Mode" in step for step in report.troubleshooting_steps)


def test_diagnose_ollama_when_online_with_all_models() -> None:
    """When daemon is reachable with all approved models, report shows all models ready."""

    def online_get(_url: str) -> dict[str, Any]:
        return {"models": [{"name": f"{model}:latest"} for model in APPROVED_MODELS]}

    report = diagnose_ollama(
        base_url="http://localhost:11434",
        http_get_fn=online_get,
    )
    assert report.is_online is True
    assert len(report.missing_approved_models) == 0
    assert "ready" in report.message.lower() or "available" in report.message.lower()


def test_diagnose_ollama_when_online_with_missing_models() -> None:
    """When daemon is online but missing models, report lists pull commands."""

    def partial_get(_url: str) -> dict[str, Any]:
        return {"models": [{"name": APPROVED_MODELS[0]}]}

    report = diagnose_ollama(
        base_url="http://localhost:11434",
        http_get_fn=partial_get,
    )
    assert report.is_online is True
    assert APPROVED_MODELS[0] in report.installed_models
    assert len(report.missing_approved_models) == len(APPROVED_MODELS) - 1
    assert any(APPROVED_MODELS[1] in step for step in report.troubleshooting_steps)


def test_diagnose_ollama_rejects_external_hosts_per_law2() -> None:
    """Per LAW 2, non-loopback endpoints must be rejected."""
    with pytest.raises(ValueError, match="LAW 2 violation"):
        diagnose_ollama(base_url="https://remote.ollama.com")
