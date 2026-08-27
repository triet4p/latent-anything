"""Unit and contract tests for the executable M14 L02 runner.

These tests use tiny synthetic arrays and never fit ConvVAE or call ``main``.
"""

# The tests intentionally exercise cohesive private helpers; the public entry
# point is reserved for a future explicitly approved run.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts.m14_l02_data import build_heldout_latent_paths
from scripts.m14_l02_envelope import artifact_digest, validate_artifact, validate_run_record
from scripts.m14_l02_metrics import (  # pyright: ignore[reportPrivateUsage]
    _evaluate_lerp,
    _evaluate_slerp,
    _evaluate_trajectory,
    record_spec,
    verdict,
)
from scripts.m14_l02_plan import (
    EXPECTED_GAP_IDS,
    EXPECTED_RECORD_IDS,
    load_plan,
    plan_digest,
    validate_plan,
)


def test_checked_in_plan_is_the_valid_design_source_of_truth() -> None:
    plan = load_plan()

    assert validate_plan(plan) == []
    assert plan["plan_sha256"] == plan_digest(plan)
    assert tuple(plan["gap_ids"]) == EXPECTED_GAP_IDS
    assert tuple(record["record_id"] for record in plan["records"]) == EXPECTED_RECORD_IDS
    assert plan["accepted"] is False
    assert plan["evidence_level"] == "not-run"


def test_design_plan_rejects_tampered_threshold_and_fabricated_provenance() -> None:
    plan = deepcopy(load_plan())
    plan["records"][0]["acceptance"]["real_pair_auc_min"] = 0.99
    plan["provenance_contract"]["git_sha"] = "a" * 40

    errors = validate_plan(plan)

    assert any("plan_sha256" in error for error in errors)
    assert any("provenance" in error for error in errors)


def test_future_artifact_and_run_record_reject_placeholders() -> None:
    assert validate_artifact({"lane": "M14-L02"})
    assert validate_run_record({"lane": "M14-L02", "schema_version": "m14-l02-geometry-run-v1"})


def test_artifact_self_digest_and_independent_gap_envelope() -> None:
    plan = load_plan()
    records = [
        {"record_id": record_id, "gap_ids": [gap_id], "accepted": False, "verdict": "failed"}
        for record_id, gap_id in zip(EXPECTED_RECORD_IDS, EXPECTED_GAP_IDS)
    ]
    artifact = {
        "schema_version": "m14-l02-geometry-artifact-v1",
        "lane": "M14-L02",
        "evidence_level": "D1",
        "accepted_record_ids": [],
        "accepted_gap_ids": [],
        "partial_promotion": True,
        "records": records,
        "plan_sha256": plan["plan_sha256"],
        "dataset": {
            "dataset": "sklearn.datasets.load_digits",
            "license": "BSD-3-Clause",
            "content_sha256": "a" * 64,
            "total_samples": 2,
            "train_samples": 1,
            "heldout_samples": 1,
            "train_index_sha256": "b" * 64,
            "heldout_index_sha256": "c" * 64,
        },
        "model": {"config": {}, "fit_scope": "train images only"},
        "density": {"config": {}, "fit_scope": "train latents only"},
        "backend_versions": {"numpy": "test"},
        "input_digests": {"before": {"x": "d" * 64}, "after": {"x": "d" * 64}},
        "provenance": {"git_sha": "e" * 40, "runner_source_sha256": "f" * 64, "contract_source_sha256": "0" * 64},
    }
    artifact["artifact_sha256"] = artifact_digest(artifact)

    assert validate_artifact(artifact, plan=plan) == []
    artifact["artifact_sha256"] = "0" * 64
    assert any("canonical artifact" in error for error in validate_artifact(artifact, plan=plan))


def test_heldout_path_helper_is_deterministic_and_does_not_mutate_inputs() -> None:
    plan = deepcopy(load_plan())
    plan["records"][0]["acceptance"]["pair_count_min"] = 4
    latents = np.arange(32, dtype=np.float64).reshape(8, 4) / 10.0
    images = np.arange(8 * 64, dtype=np.float64).reshape(8, 1, 8, 8) / 64.0
    labels = np.arange(8, dtype=np.int64) % 2
    before_latents, before_images, before_labels = latents.copy(), images.copy(), labels.copy()

    first = build_heldout_latent_paths(latents, images, labels, plan)
    second = build_heldout_latent_paths(latents, images, labels, plan)

    assert len(first["pairs"]) == 4
    assert [pair["same_label"] for pair in first["pairs"]] == [pair["same_label"] for pair in second["pairs"]]
    assert np.array_equal(latents, before_latents)
    assert np.array_equal(images, before_images)
    assert np.array_equal(labels, before_labels)


def test_interpolation_helpers_enforce_finite_float64_contracts() -> None:
    plan = load_plan()
    latents = np.arange(32, dtype=np.float64).reshape(8, 4) / 10.0
    images = np.zeros((8, 1, 8, 8), dtype=np.float64)
    labels = np.arange(8, dtype=np.int64) % 2
    local_plan = deepcopy(plan)
    local_plan["records"][0]["acceptance"]["pair_count_min"] = 2
    paths = build_heldout_latent_paths(latents, images, labels, local_plan)

    lerp_metrics = _evaluate_lerp(paths)
    slerp_metrics = _evaluate_slerp(paths)

    assert lerp_metrics["finite"] and lerp_metrics["no_input_mutation"]
    assert slerp_metrics["finite"] and slerp_metrics["no_input_mutation"]
    assert record_spec(plan, "slerp_spherical")["gap_ids"] != record_spec(plan, "slerp_latent_operation")["gap_ids"]


def test_slerp_records_have_independent_verdicts() -> None:
    plan = load_plan()
    metrics = {
        "endpoint_error": 0.0,
        "norm_error": 0.0,
        "angular_additivity_error": 0.0,
        "finite": True,
        "no_input_mutation": True,
    }

    assert verdict(record_spec(plan, "slerp_spherical"), metrics)
    assert verdict(record_spec(plan, "slerp_latent_operation"), metrics)


def test_trajectory_uses_independent_pair_paths_and_derangement() -> None:
    plan = deepcopy(load_plan())
    plan["records"][0]["acceptance"]["pair_count_min"] = 4
    plan["records"][5]["acceptance"]["independent_pair_trials_min"] = 4
    plan["execution"]["path_points"] = 4
    plan["execution"]["trajectory_query_points"] = 5
    plan["execution"]["smoothing"]["window"] = 3
    latents = np.arange(32, dtype=np.float64).reshape(8, 4) / 10.0
    images = np.zeros((8, 1, 8, 8), dtype=np.float64)
    labels = np.arange(8, dtype=np.int64) % 2
    paths = build_heldout_latent_paths(latents, images, labels, plan)

    metrics = _evaluate_trajectory(paths, plan)

    assert metrics["independent_pair_trials"] == 4
    assert metrics["no_self_mapping"] is True
    assert metrics["unequal_lengths"] is True
    assert metrics["pair_path_digest"]
    assert metrics["unrelated_pair_permutation_digest"]
    assert metrics["positive_scores_digest"] != metrics["negative_scores_digest"]


def test_importing_runner_does_not_create_evidence() -> None:
    artifact = Path(__file__).resolve().parents[1] / "artifacts/m14/l02-geometry.json"

    assert not artifact.exists()


@pytest.mark.parametrize("record_id", EXPECTED_RECORD_IDS)
def test_every_declared_record_has_a_local_not_run_verdict(record_id: str) -> None:
    record = record_spec(load_plan(), record_id)

    assert record["verdict"] == "not-run"
    assert record["accepted"] is None
