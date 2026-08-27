"""No-network contract tests for the M14 L03 design and support code."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.m14_l03_analysis import check
from scripts.m14_l03_data import glyph_prompt, grouped_digit_split, validate_group_labels
from scripts.m14_l03_envelope import (
    apply_dependency_blocking,
    build_artifact,
    build_run_record,
    failure_envelope,
    source_digests,
    validate_artifact,
)
from scripts.m14_l03_metrics import compression_ok, paired_bootstrap, wilson_95
from scripts.m14_l03_plan import load_plan, plan_digest, validate_plan


def test_plan_is_immutable_and_check_has_no_result_artifact() -> None:
    plan = load_plan()
    assert plan["plan_sha256"] == plan_digest(plan)
    assert not (Path(__file__).parents[1] / "artifacts/m14/l03-analysis.json").exists()
    assert not (Path(__file__).parents[1] / "artifacts/m14/l03-analysis.run.json").exists()
    assert check()["plan_sha256"] == plan["plan_sha256"]


def test_grouped_split_has_zero_prompt_overlap_and_auditable_counts() -> None:
    split = grouped_digit_split()
    masks = split["partitions"]
    digests = split["prompt_digests"]
    groups = {name: set(digests[mask].tolist()) for name, mask in masks.items()}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    assert sum(split["metadata"]["partition_counts"].values()) == 1797
    assert all(count > 0 for count in split["metadata"]["partition_counts"].values())


def test_conflicting_prompt_digest_fails_in_protocol() -> None:
    image = np.zeros((8, 8))
    assert glyph_prompt(image) == glyph_prompt(image.copy())
    with pytest.raises(ValueError, match="multiple labels"):
        validate_group_labels(np.array(["same", "same"]), np.array([0, 1]))
    plan = load_plan()
    assert validate_plan(plan) == []


def test_dependency_predicates_are_independent_but_block_downstream() -> None:
    records = [
        {"record_id": "a", "accepted": False},
        {"record_id": "b", "accepted": True},
        {"record_id": "c", "accepted": True},
    ]
    result = apply_dependency_blocking(records)
    assert [item["accepted"] for item in result] == [False, False, False]
    assert result[1]["blocked_by_dependency"] is True
    assert result[2]["blocked_by_dependency"] is True


def test_source_provenance_digests_are_sha256() -> None:
    digests = source_digests()
    assert set(digests) == {"runner_source_sha256", "contract_source_sha256", "implementation_source_sha256"}
    assert all(len(value) == 64 for value in digests.values())


def test_intervals_are_explicit_and_deterministic() -> None:
    real = np.array([True, True, False, True, False])
    control = np.array([False, True, False, False, False])
    first = paired_bootstrap(real, control, resamples=100, seed=7901)
    assert first == paired_bootstrap(real, control, resamples=100, seed=7901)
    assert first["sampling_unit"] == "held-out example"
    assert wilson_95(3, 5)[0] < 0.6 < wilson_95(3, 5)[1]
    assert compression_ok(0.65, 0.7)
    assert not compression_ok(0.59, 0.7)


def test_failure_envelope_never_promotes() -> None:
    plan = load_plan()
    envelope = failure_envelope(plan, RuntimeError("no CUDA"))
    assert envelope["artifact_written"] is False
    assert envelope["accepted_record_ids"] == []


def test_artifact_validation_rejects_bad_digest_and_dependency() -> None:
    plan = load_plan()
    artifact = {
        "schema_version": "m14-l03-analysis-artifact-v1",
        "lane": "M14-L03",
        "records": [],
        "plan_sha256": plan_digest(plan),
        "artifact_sha256": "0" * 64,
        "provenance": {"git_sha": "0" * 40},
        "split": {"group_overlap": {"train_val": 0, "train_test": 0, "val_test": 0}},
    }
    errors = validate_artifact(artifact, plan)
    assert any("self-digest" in error for error in errors)


def test_build_artifact_has_valid_self_digest_without_writing() -> None:
    plan = load_plan()
    records = [
        {"record_id": record["record_id"], "gap_id": record["gap_id"], "accepted": False, "verdict": "failed"}
        for record in plan["records"]
    ]
    split = {"group_overlap": {"train_val": 0, "train_test": 0, "val_test": 0}}
    artifact = build_artifact(plan, records, split, {"git_sha": "0" * 40})
    assert validate_artifact(artifact, plan) == []
    run = build_run_record(plan, artifact, {"gpu": "not-measured-in-unit-test"})
    assert run["plan_sha256"] == plan["plan_sha256"]
    assert run["artifact_sha256"] == artifact["artifact_sha256"]
    assert not (Path(__file__).parents[1] / "artifacts/m14/l03-analysis.json").exists()
