"""Offline smoke test for the VQ-VAE comparison artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.vq_vae_digits_evidence import main


def test_vq_vae_evidence_script_writes_reproducible_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The benchmark runs without network access and records both geometry paths."""

    monkeypatch.chdir(tmp_path)
    main()
    artifact = tmp_path / "artifacts" / "vq_vae_digits_evidence.json"
    config = tmp_path / "artifacts" / "vq_vae_digits_evidence_config.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["dataset_revision"] == "sklearn-digits-8x8@scikit-learn==1.9.0"
    assert payload["model_revision"] == "compact-vq-vae-v1"
    assert payload["acceptance"]["reconstruction_mse_finite"] is True
    assert payload["acceptance"]["continuous_path_is_comparison_only"] is True
    assert json.loads(config.read_text(encoding="utf-8"))["offline"] is True
