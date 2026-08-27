"""No-network contract tests for the M14 L03 design and support code."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.m14_l03_analysis import check
from scripts.m14_l03_data import glyph_prompt, grouped_digit_split, validate_group_labels
from scripts.m14_l03_envelope import (
    apply_dependency_blocking,
    build_artifact,
    build_report,
    build_run_record,
    failure_envelope,
    source_digests,
    validate_artifact,
    validate_report,
)
from scripts.m14_l03_features import extract_batched
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
    assert len(envelope["git_sha"]) == 40
    assert envelope["source_digests"]["implementation_source_sha256"]
    assert envelope["error_type"] == "RuntimeError"


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
    assert any("provenance" in error for error in errors)


def test_artifact_validation_detects_implementation_digest_mismatch() -> None:
    plan = load_plan()
    records = [
        {"record_id": item["record_id"], "gap_id": item["gap_id"], "accepted": False, "verdict": "failed"}
        for item in plan["records"]
    ]
    split = {"group_overlap": {"train_val": 0, "train_test": 0, "val_test": 0}}
    provenance = {
        "git_sha": "0" * 40,
        **source_digests(),
        "model_id": "test",
        "model_revision": "0" * 40,
        "tokenizer": {},
        "runtime_versions": {},
        "resources": {},
        "cleanup": "test-only",
    }
    artifact = build_artifact(plan, records, split, provenance)
    artifact["provenance"]["implementation_source_sha256"] = "f" * 64
    assert any("implementation_source_sha256" in error for error in validate_artifact(artifact, plan, source_digests()))


def test_build_artifact_has_valid_self_digest_without_writing() -> None:
    plan = load_plan()
    records = [
        {"record_id": record["record_id"], "gap_id": record["gap_id"], "accepted": False, "verdict": "failed"}
        for record in plan["records"]
    ]
    split = {"group_overlap": {"train_val": 0, "train_test": 0, "val_test": 0}}
    provenance = {
        "git_sha": "0" * 40,
        **source_digests(),
        "model_id": "test",
        "model_revision": "0" * 40,
        "tokenizer": {},
        "runtime_versions": {},
        "resources": {},
        "cleanup": "test-only",
    }
    artifact = build_artifact(plan, records, split, provenance)
    assert validate_artifact(artifact, plan) == []
    run = build_run_record(plan, artifact, {"gpu": "not-measured-in-unit-test"})
    assert run["plan_sha256"] == plan["plan_sha256"]
    assert run["artifact_sha256"] == artifact["artifact_sha256"]
    report = build_report(plan, artifact, run)
    assert validate_report(report, plan) == []
    assert not (Path(__file__).parents[1] / "artifacts/m14/l03-analysis.json").exists()


def test_batched_feature_extraction_preserves_order_pooling_and_release() -> None:
    class FakeIntegration:
        def __init__(self) -> None:
            self.max_batch = 0
            self.released = 0

        def tokenize(self, prompts: tuple[str, ...], **_: object) -> dict[str, np.ndarray]:
            return {"attention_mask": np.ones((len(prompts), 3), dtype=bool)}

        def generate(self, request: object) -> object:
            prompts = request["prompt"]  # type: ignore[index]
            self.max_batch = max(self.max_batch, len(prompts))
            values = np.asarray(
                [[[float(ord(prompt) - 97), 1.0], [float(ord(prompt) - 95), 3.0], [99.0, 99.0]] for prompt in prompts]
            )
            owner = self

            class Result:
                attention_mask = np.asarray([[True, True, False] for _ in prompts])
                hidden_states = (SimpleNamespace(layer=12, values=values),)

                def __del__(self) -> None:
                    owner.released += 1

            return Result()

    fake = FakeIntegration()
    hidden, metadata = extract_batched(
        fake,
        ("a", "b", "c", "d", "e"),
        layers=(12,),
        max_length=64,
        batch_size=2,
        request_factory=lambda prompt, **_: {"prompt": prompt},
    )
    assert fake.max_batch == 2
    np.testing.assert_allclose(hidden[12][:, 0], [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(hidden[12][:, 1], [2.0] * 5)
    assert metadata["inference_batch_count"] == 3
    assert fake.released == 3
