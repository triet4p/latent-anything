"""Regression tests for the Sprint 34 held-out ConvVAE evidence lane."""

from __future__ import annotations

import json
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pytest

from scripts.conv_vae_heldout_benchmark import main, split_digits


def test_split_is_deterministic_and_disjoint() -> None:
    images = np.zeros((10, 1, 8, 8), dtype=np.float64)
    labels = np.arange(10)
    first = split_digits(images, labels, seed=42)
    second = split_digits(images, labels, seed=42)
    np.testing.assert_array_equal(first[4], second[4])
    np.testing.assert_array_equal(first[5], second[5])
    assert np.intersect1d(first[4], first[5]).size == 0
    assert len(first[4]) == 8
    assert len(first[5]) == 2


def test_heldout_benchmark_meets_predeclared_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    payload = main(tmp_path / "artifacts")
    artifact = json.loads((tmp_path / "artifacts" / "conv_vae_heldout_benchmark.json").read_text())
    assert payload["accepted"] is True
    assert artifact["dataset_revision"] == f"scikit-learn=={package_version('scikit-learn')}"
    assert artifact["dataset_license"] == "BSD-3-Clause (scikit-learn bundled digits dataset)"
    assert artifact["split"]["train_samples"] + artifact["split"]["heldout_samples"] == 1797
    assert set(artifact["split"]["train_indices"]).isdisjoint(artifact["split"]["heldout_indices"])
    metrics = artifact["metrics"]
    assert metrics["heldout_reconstruction_mse"] < 0.9 * metrics["zero_baseline_mse"]
    assert metrics["latent_utilization_train"] >= 1e-3
    assert artifact["composition"]["heldout_pca_shape"] == [360, 2]
    assert artifact["composition"]["heldout_sae_shape"] == [360, 3]
