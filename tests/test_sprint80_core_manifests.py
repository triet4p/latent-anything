"""Consumer-observable tests for the two frozen Sprint 80.5 core manifests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from latent_anything._benchmark_manifest import (
    BenchmarkManifestValidationError,
    manifest_digest,
    validate_manifest,
)

ENCODER_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts" / "benchmark_manifest_sprint80_encoder_autoencoder_v1.json"
)
TRANSFORMER_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts" / "benchmark_manifest_sprint80_transformer_hidden_state_v1.json"
)


def _load(path: Path) -> dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def test_both_core_manifests_are_validator_clean() -> None:
    for path in (ENCODER_PATH, TRANSFORMER_PATH):
        validate_manifest(_load(path))


def test_manifests_carry_distinct_core_identities() -> None:
    encoder = _load(ENCODER_PATH)
    transformer = _load(TRANSFORMER_PATH)

    assert encoder["manifest_id"] == "sprint80-core-encoder-autoencoder-collapse-v1"
    assert transformer["manifest_id"] == "sprint80-core-transformer-hidden-state-probe-v1"
    assert encoder["manifest_id"] != transformer["manifest_id"]

    encoder_metrics = encoder["metrics"]
    transformer_metrics = transformer["metrics"]
    assert isinstance(encoder_metrics, list) and isinstance(transformer_metrics, list)
    assert {metric["taxonomy_family_id"] for metric in encoder_metrics if isinstance(metric, dict)} == {
        "collapse_rank_loss"
    }
    assert {metric["taxonomy_family_id"] for metric in transformer_metrics if isinstance(metric, dict)} == {
        "separability_probe_leakage"
    }

    encoder_model = encoder["model"]
    transformer_model = transformer["model"]
    assert isinstance(encoder_model, dict) and isinstance(transformer_model, dict)
    assert encoder_model["revision"] == "conv-vae-8x8-latent4-seed0-epochs5"
    assert transformer_model["revision"] == "e7da7f221d5bf496a48136c0cd264e630fe9fcc8"

    encoder_dataset = encoder["dataset"]
    transformer_dataset = transformer["dataset"]
    assert isinstance(encoder_dataset, dict) and isinstance(transformer_dataset, dict)
    assert encoder_dataset["revision"] == "scikit-learn==1.9.0"
    assert transformer_dataset["revision"] == "f776294184f13b8ff2337b3841cf9269a6216d1e"

    for manifest in (encoder, transformer):
        assert manifest["status"] == "predeclared"
        commitment = manifest["commitment"]
        assert isinstance(commitment, dict)
        assert commitment["locked"] is True
        thresholds = manifest["thresholds"]
        assert isinstance(thresholds, list) and thresholds
        for threshold in thresholds:
            assert isinstance(threshold, dict)
            assert threshold["predeclared"] is True
        causal = manifest["causal_expectation"]
        assert isinstance(causal, dict)
        assert causal["applicable"] is True
        assert causal["target_metric_ids"]
        assert causal["falsification_rule"] not in ("", "not_applicable")


def test_transformer_manifest_is_leakage_safe_by_declaration() -> None:
    manifest = _load(TRANSFORMER_PATH)
    controls = manifest["controls"]
    assert isinstance(controls, list)
    kinds = {control["id"]: control["kind"] for control in controls if isinstance(control, dict)}
    assert "control-label-randomization" in kinds
    assert "control-nonseparable-negative" in kinds
    assert "control-capacity" in kinds
    dataset = manifest["dataset"]
    assert isinstance(dataset, dict)
    assert dataset["split"] == "validation"
    defect = manifest["defect"]
    assert isinstance(defect, dict)
    declaration = defect["declaration"]
    assert isinstance(declaration, str)
    assert "heldout" in declaration


def test_locked_manifests_reject_post_hoc_mutation() -> None:
    manifest = _load(ENCODER_PATH)
    mutated = deepcopy(manifest)
    thresholds = mutated["thresholds"]
    assert isinstance(thresholds, list)
    threshold = thresholds[0]
    assert isinstance(threshold, dict)
    threshold["value"] = 0.01

    with pytest.raises(BenchmarkManifestValidationError, match="digest does not match"):
        validate_manifest(mutated)


def test_locked_manifests_reject_incomplete_thresholds() -> None:
    manifest = _load(TRANSFORMER_PATH)
    mutated = deepcopy(manifest)
    thresholds = mutated["thresholds"]
    assert isinstance(thresholds, list)
    thresholds.pop()
    commitment = mutated["commitment"]
    assert isinstance(commitment, dict)
    commitment["manifest_sha256"] = manifest_digest(mutated)

    with pytest.raises(BenchmarkManifestValidationError, match="every metric must have exactly one"):
        validate_manifest(mutated)
