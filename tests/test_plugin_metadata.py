"""Tests for metadata-only external plugin declarations."""

from __future__ import annotations

from importlib.metadata import EntryPoint

import pytest

from latent_anything.plugin_groups import ENTRY_POINT_GROUP_ADAPTER, ENTRY_POINT_GROUP_TRANSITION
from latent_anything.plugin_metadata import EntryPointMetadata
from latent_anything.registry_aliases import KIND_ADAPTER, KIND_RUNTIME


def test_entry_point_metadata_does_not_load_target() -> None:
    """Metadata extraction records declaration/provenance only."""

    point = EntryPoint(
        name="hello",
        value="fixture_package:Hello",
        group=ENTRY_POINT_GROUP_ADAPTER,
    )
    metadata = EntryPointMetadata.from_entry_point(point)

    assert metadata.name == "hello"
    assert metadata.value == "fixture_package:Hello"
    assert metadata.distribution is None
    assert metadata.version is None
    assert metadata.registry_kind == KIND_ADAPTER


def test_transition_group_preserves_existing_runtime_kind() -> None:
    """Capability groups do not prematurely add a registry kind."""

    point = EntryPoint(name="transition", value="fixture_package:Transition", group=ENTRY_POINT_GROUP_TRANSITION)
    assert EntryPointMetadata.from_entry_point(point).registry_kind == KIND_RUNTIME


def test_unsupported_group_is_actionable() -> None:
    """Unknown declarations fail with supported-group guidance."""

    point = EntryPoint(name="bad", value="fixture_package:Bad", group="latent_anything.unknown")
    with pytest.raises(ValueError, match="Unsupported latent-anything entry-point group"):
        EntryPointMetadata.from_entry_point(point)
