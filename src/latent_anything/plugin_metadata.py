"""Metadata-only helpers for external Python entry points.

This module deliberately inspects entry-point declarations without importing
the referenced plugin object.  Loading untrusted third-party code belongs to
the discovery operation and is isolated there; callers can list provenance
and versions without executing plugin imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import EntryPoint

from latent_anything.plugin_groups import (
    CANONICAL_ENTRY_POINT_GROUPS,
    ENTRY_POINT_GROUP_ADAPTER,
    ENTRY_POINT_GROUP_ANALYSIS,
    ENTRY_POINT_GROUP_INTERVENTION,
    ENTRY_POINT_GROUP_PLANNER,
    ENTRY_POINT_GROUP_TRANSITION,
)
from latent_anything.registry_aliases import (
    KIND_ADAPTER,
    KIND_ANALYSIS,
    KIND_INTERVENTION,
    KIND_RUNTIME,
)

PLUGIN_API_VERSION = "1"
"""The supported external-plugin metadata contract version."""

PLUGIN_API_VERSION_ATTRIBUTE = "__latent_anything_plugin_api_version__"
"""Callable attribute used by external targets to declare compatibility."""

ENTRY_POINT_GROUP_TO_REGISTRY_KIND: dict[str, str] = {
    ENTRY_POINT_GROUP_ADAPTER: KIND_ADAPTER,
    ENTRY_POINT_GROUP_ANALYSIS: KIND_ANALYSIS,
    ENTRY_POINT_GROUP_INTERVENTION: KIND_INTERVENTION,
    # Transition and planner are proven capability families, but the
    # registry currently stores their built-ins under its runtime kind.
    ENTRY_POINT_GROUP_TRANSITION: KIND_RUNTIME,
    ENTRY_POINT_GROUP_PLANNER: KIND_RUNTIME,
}


@dataclass(frozen=True, slots=True)
class EntryPointMetadata:
    """Non-executing provenance metadata for one declared entry point."""

    name: str
    group: str
    value: str
    distribution: str | None
    version: str | None
    registry_kind: str

    @classmethod
    def from_entry_point(cls, entry_point: EntryPoint) -> EntryPointMetadata:
        """Extract metadata without calling ``entry_point.load()``.

        Raises
        ------
        ValueError
            If the declaration uses an unsupported group.  This is a
            declaration error, not an import/load failure.
        """

        try:
            registry_kind = ENTRY_POINT_GROUP_TO_REGISTRY_KIND[entry_point.group]
        except KeyError:
            supported = ", ".join(CANONICAL_ENTRY_POINT_GROUPS)
            raise ValueError(
                f"Unsupported latent-anything entry-point group {entry_point.group!r}; expected one of: {supported}"
            ) from None

        distribution = entry_point.dist
        return cls(
            name=entry_point.name,
            group=entry_point.group,
            value=entry_point.value,
            distribution=distribution.name if distribution is not None else None,
            version=distribution.version if distribution is not None else None,
            registry_kind=registry_kind,
        )


__all__ = [
    "ENTRY_POINT_GROUP_TO_REGISTRY_KIND",
    "PLUGIN_API_VERSION",
    "PLUGIN_API_VERSION_ATTRIBUTE",
    "EntryPointMetadata",
]
