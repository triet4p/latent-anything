"""Tests for the Sprint 73 external-plugin group vocabulary."""

from __future__ import annotations

from latent_anything.plugin_groups import (
    CANONICAL_ENTRY_POINT_GROUPS,
    ENTRY_POINT_GROUP_ADAPTER,
    ENTRY_POINT_GROUP_ANALYSIS,
    ENTRY_POINT_GROUP_INTERVENTION,
    ENTRY_POINT_GROUP_PLANNER,
    ENTRY_POINT_GROUP_TRANSITION,
)


def test_canonical_entry_point_groups_are_stable_and_unique() -> None:
    """The public group vocabulary has one deterministic order."""

    assert CANONICAL_ENTRY_POINT_GROUPS == (
        ENTRY_POINT_GROUP_ADAPTER,
        ENTRY_POINT_GROUP_ANALYSIS,
        ENTRY_POINT_GROUP_INTERVENTION,
        ENTRY_POINT_GROUP_TRANSITION,
        ENTRY_POINT_GROUP_PLANNER,
    )
    assert len(CANONICAL_ENTRY_POINT_GROUPS) == len(set(CANONICAL_ENTRY_POINT_GROUPS))
    assert all(group.startswith("latent_anything.") for group in CANONICAL_ENTRY_POINT_GROUPS)


def test_intervention_is_the_canonical_transformation_group() -> None:
    """Transformation-like plugins use RFC 0001's intervention vocabulary."""

    assert ENTRY_POINT_GROUP_INTERVENTION == "latent_anything.intervention"
    assert "latent_anything.transformation" not in CANONICAL_ENTRY_POINT_GROUPS
