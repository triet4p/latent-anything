"""Thin public facade for the independent L04.9 v2 stage validators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scripts._m14_l049_v2_fixture import TRAIN_FIXTURE_PATH, read_rows
from scripts._m14_l049_v2_schema import pinned_commitment_policy
from scripts._m14_l049_v2_validate_stage_a import validate_stage_a_impl
from scripts._m14_l049_v2_validate_stage_b import validate_stage_b_impl


def validate_stage_a(
    artifact: Mapping[str, Any], train_rows: Sequence[Mapping[str, Any]], addendum: Mapping[str, Any]
) -> list[str]:
    """Validate Stage A and return errors; malformed input never escapes."""
    try:
        return validate_stage_a_impl(artifact, train_rows, addendum, policy=pinned_commitment_policy())
    except Exception as exc:  # fail closed at the untrusted artifact boundary
        return [f"Stage A malformed input: {type(exc).__name__}"]


def validate_stage_b(
    artifact: Mapping[str, Any],
    holdout_rows: Sequence[Mapping[str, Any]],
    holdout_seed: bytes,
    candidate_artifact: Mapping[str, Any],
    addendum: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    """Validate Stage B and return errors; malformed input never escapes."""
    try:
        rows = list(train_rows) if train_rows is not None else read_rows(TRAIN_FIXTURE_PATH)[1]
        return validate_stage_b_impl(
            artifact,
            holdout_rows,
            holdout_seed,
            candidate_artifact,
            addendum,
            rows,
            policy=pinned_commitment_policy(),
        )
    except Exception as exc:  # fail closed at the untrusted artifact boundary
        return [f"Stage B malformed input: {type(exc).__name__}"]


__all__ = ["validate_stage_a", "validate_stage_b"]
