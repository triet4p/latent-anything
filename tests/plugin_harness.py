"""Small test-only assertions for external plugin provenance."""

from __future__ import annotations

from latent_anything.registry import Registry


def assert_external_entry_provenance(
    registry: Registry,
    name: str,
    *,
    group: str,
    distribution: str | None,
    version: str | None,
    api_version: str,
) -> None:
    """Assert the stable provenance fields on one loaded external entry."""

    entry = registry.lookup(name)
    assert entry.metadata["source"] == "external"
    assert entry.metadata["entry_point_group"] == group
    assert entry.metadata["distribution"] == distribution
    assert entry.metadata["version"] == version
    assert entry.metadata["plugin_api_version"] == api_version
