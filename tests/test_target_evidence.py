"""Behavioral tests for versioned, content-addressed target-label provenance."""

from __future__ import annotations

import numpy as np
import pytest

from latent_anything._redundancy_separability_detection import LabeledBatch
from latent_anything._target_evidence import (
    TargetEvidenceError,
    bind_target_from_batch,
    build_target_record,
    target_record_digest,
    validate_target_record,
)
from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue


def _record(rule: str, *, eval_indices: tuple[int, ...] = (2, 3)) -> dict[str, object]:
    return build_target_record(
        target_id="fixture-target",
        rule=rule,
        rule_kind="declared-test-rule",
        representation_identity="fixture-representation",
        dataset_split_identity="fixture-dataset-split",
        train_split_identity="fixture-train-split",
        eval_split_identity="fixture-eval-split",
        sample_ids=("sample-a", "sample-b", "sample-c", "sample-d"),
        labels=(0, 1, 0, 1),
        train_indices=(0, 1),
        eval_indices=eval_indices,
        capacity="bounded-linear-probe",
    )


def test_target_record_content_address_binds_the_predeclared_rule() -> None:
    declared = _record("header matches frozen expression")
    changed = _record("different target rule")

    validate_target_record(declared)

    assert target_record_digest(declared) != target_record_digest(changed)


def test_target_record_rejects_unassigned_samples() -> None:
    with pytest.raises(TargetEvidenceError, match="assign every target sample exactly once"):
        _record("header matches frozen expression", eval_indices=(2,))


def test_bind_target_from_batch_preserves_both_split_identities() -> None:
    batch = LabeledBatch(
        value=LatentValue(
            np.asarray([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
            LatentSpace(dim=2),
        ),
        labels=(0, 1, 0, 1),
        sample_ids=("sample-a", "sample-b", "sample-c", "sample-d"),
        train_indices=(0, 1),
        eval_indices=(2, 3),
        train_split_identity="fixture-train-split",
        eval_split_identity="fixture-eval-split",
        capacity="bounded-linear-probe",
    )

    record = bind_target_from_batch(
        batch,
        target_id="fixture-target",
        rule="header matches frozen expression",
        rule_kind="declared-test-rule",
        representation_identity="fixture-representation",
        dataset_split_identity="fixture-dataset-split",
    )
    validate_target_record(record)

    assert record["train_split_identity"] == "fixture-train-split"
    assert record["eval_split_identity"] == "fixture-eval-split"


def test_target_record_rejects_a_non_mapping() -> None:
    with pytest.raises(TargetEvidenceError, match="target record must be a mapping"):
        validate_target_record(None)
