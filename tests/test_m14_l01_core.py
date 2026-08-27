"""Focused regression tests for the M14 L01 evidence runner and schema."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from scripts.m14_l01_core import main, split_digits, validate_l01_artifact


def test_split_digits_is_deterministic_disjoint_and_shape_safe() -> None:
    """The declared split is stable and contains every source row once."""

    images = np.zeros((10, 1, 8, 8), dtype=np.float64)
    labels = np.arange(10, dtype=np.int64)
    first = split_digits(images, labels)
    second = split_digits(images, labels)
    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))
    train_indices, heldout_indices = first[4], first[5]
    assert np.intersect1d(train_indices, heldout_indices).size == 0
    assert sorted(np.concatenate((train_indices, heldout_indices)).tolist()) == list(range(10))


def test_l01_runner_writes_accepted_schema_and_matching_run_record(tmp_path: Path) -> None:
    """A real ConvVAE/pipeline run produces a validator-recognized D2 artifact."""

    payload = main(tmp_path)
    artifact = cast(dict[str, object], json.loads((tmp_path / "l01-core.json").read_text(encoding="utf-8")))
    run_record = cast(dict[str, object], json.loads((tmp_path / "l01-core.run.json").read_text(encoding="utf-8")))
    assert payload == artifact
    assert payload["accepted"] is True
    assert payload["evidence_level"] == "D2"
    assert validate_l01_artifact(payload) == []
    assert run_record["artifact_sha256"] == payload["artifact_sha256"]
    assert run_record["status"] == "accepted"
    split = cast(dict[str, object], payload["split"])
    metrics = cast(dict[str, float], payload["metrics"])
    assert split["train_samples"] == 1437
    assert split["heldout_samples"] == 360
    assert metrics["heldout_reconstruction_mse"] < metrics["zero_baseline_mse"]


def test_l01_schema_rejects_missing_field_and_tampered_digest() -> None:
    """Malformed or modified evidence cannot be mistaken for a valid artifact."""

    with pytest.raises(ValueError, match="shape"):
        split_digits(np.zeros((10, 8, 8)), np.arange(10, dtype=np.int64))
    payload = {
        "schema_version": 1,
        "lane": "M14-L01",
        "capability_id": "THY-T01-METRIC-SPACE-VA-VECTOR-SPACE",
        "evidence_level": "D2",
    }
    missing_errors = validate_l01_artifact(payload)
    assert any("missing required field: artifact_sha256" in error for error in missing_errors)
    complete = cast(dict[str, object], copy.deepcopy(payload))
    placeholder_fields: dict[str, object] = {
        field: {}
        for field in (
            "seed",
            "dataset",
            "split",
            "model",
            "backend",
            "contracts",
            "metrics",
            "controls",
            "acceptance",
            "provenance",
        )
    }
    complete.update(placeholder_fields)
    complete["artifact_sha256"] = "0" * 64
    assert any("artifact_sha256" in error for error in validate_l01_artifact(complete))
