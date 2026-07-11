"""Tests for canonical registry kinds and legacy migration diagnostics."""

from __future__ import annotations

import warnings

from latent_anything import GLOBAL_REGISTRY, ObjectSpec, build_from_config
from latent_anything.registry import KIND_ANALYSIS, KIND_INTERVENTION


def test_builtin_entries_use_canonical_semantic_kinds() -> None:
    assert {entry.kind for entry in GLOBAL_REGISTRY.list(KIND_ANALYSIS)} == {KIND_ANALYSIS}
    assert {entry.kind for entry in GLOBAL_REGISTRY.list(KIND_INTERVENTION)} == {KIND_INTERVENTION}


def test_legacy_config_kind_builds_with_one_deprecation_warning() -> None:
    spec = ObjectSpec(kind="method_a", name="pca", params={"n_components": 2})
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        built = build_from_config(spec)
    assert built.__class__.__name__ == "PCA"
    assert len(captured) == 1
    assert "analysis" in str(captured[0].message)


def test_canonical_nested_config_has_no_migration_warning() -> None:
    spec = ObjectSpec(kind="analysis", name="pca", params={"n_components": 2})
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        build_from_config(spec)
    assert captured == []
