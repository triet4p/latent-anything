"""Regression tests for the cached, revision-pinned Diffusers fidelity lane."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.diffusers_vae_fidelity import (
    MODEL_REVISION,
    SAFE_WEIGHTS_SHA256,
    metrics,
    portable_snapshot_label,
    run_evidence,
)


def test_fidelity_metric_is_zero_for_identical_arrays() -> None:
    import numpy as np

    values = np.ones((2, 3), dtype=np.float32)
    assert metrics(values, values) == {"max_abs_error": 0.0, "rmse": 0.0, "max_relative_error": 0.0}


def test_fidelity_snapshot_label_is_repository_relative() -> None:
    snapshot = Path(r"C:\runner\workspace\latent-anything\.cache") / f"hf-sd-vae-ft-mse-{MODEL_REVISION}"

    assert portable_snapshot_label(snapshot) == f".cache/hf-sd-vae-ft-mse-{MODEL_REVISION}"
    assert not Path(portable_snapshot_label(snapshot)).is_absolute()


@pytest.mark.large_download
@pytest.mark.skipif(
    os.environ.get("LATENT_ANYTHING_RUN_REAL_CHECKPOINT") != "1",
    reason="set LATENT_ANYTHING_RUN_REAL_CHECKPOINT=1 for the cached checkpoint lane",
)
def test_cached_checkpoint_fidelity_artifact() -> None:
    root = Path(__file__).resolve().parents[1]
    snapshot = root / ".cache" / f"hf-sd-vae-ft-mse-{MODEL_REVISION}"
    if not snapshot.is_dir():
        pytest.skip("verified local checkpoint snapshot is absent")
    artifact = root / "artifacts" / "diffusers_vae_fidelity.json"
    payload = run_evidence(snapshot, artifact)
    assert payload["accepted"] is True
    assert payload["network_attempts"] == []
    assert payload["weights"]["sha256"] == SAFE_WEIGHTS_SHA256
    assert payload["modes"]["mean"]["latent_metrics"]["max_abs_error"] <= 1e-6
    assert payload["modes"]["sample"]["decode_metrics"]["max_abs_error"] <= 1e-6
    assert json.loads(artifact.read_text(encoding="utf-8"))["accepted"] is True
