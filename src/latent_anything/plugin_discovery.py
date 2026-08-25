"""Lazy, deterministic discovery of external latent-anything plugins.

Listing entry points is metadata-only.  Loading is an explicit operation so
base-package imports and registry listing never execute third-party code.
When loading, declarations are processed in canonical group/name/distribution/
declaration order.
An existing registry entry or an earlier declaration wins duplicate names;
the skipped declaration is reported and is never imported.  One broken plugin
does not prevent the remaining declarations from loading.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from importlib.metadata import EntryPoint

from latent_anything.plugin_groups import CANONICAL_ENTRY_POINT_GROUPS
from latent_anything.plugin_metadata import (
    PLUGIN_API_VERSION,
    PLUGIN_API_VERSION_ATTRIBUTE,
    EntryPointMetadata,
)
from latent_anything.registry import GLOBAL_REGISTRY, Registry


@dataclass(frozen=True, slots=True)
class DiscoveryIssue:
    """A duplicate or isolated plugin-load failure."""

    metadata: EntryPointMetadata
    reason: str
    exception_type: str | None = None


class PluginContractError(ValueError):
    """Raised when a loaded external target omits or mismatches API metadata."""


@dataclass(frozen=True, slots=True)
class DiscoveryReport:
    """Results of one explicit external-plugin load operation."""

    loaded: tuple[EntryPointMetadata, ...]
    issues: tuple[DiscoveryIssue, ...]

    @property
    def duplicates(self) -> tuple[DiscoveryIssue, ...]:
        """Return issues caused by deterministic duplicate-name handling."""

        return tuple(issue for issue in self.issues if issue.exception_type is None)

    @property
    def failures(self) -> tuple[DiscoveryIssue, ...]:
        """Return isolated entry-point load failures."""

        return tuple(issue for issue in self.issues if issue.exception_type is not None)


def _values_for_group(raw: object, group: str) -> object:
    """Select one group from either modern or legacy metadata APIs."""

    selector = getattr(raw, "select", None)
    if callable(selector):
        return selector(group=group)
    if isinstance(raw, Mapping):
        return raw.get(group, ())
    if isinstance(raw, Iterable):
        return tuple(point for point in raw if isinstance(point, EntryPoint) and point.group == group)
    return raw


def _collect_entry_points(provider: Callable[[], object] | None = None) -> tuple[EntryPoint, ...]:
    """Collect canonical declarations without loading their targets."""

    raw = importlib_metadata.entry_points() if provider is None else provider()
    points: list[EntryPoint] = []
    for group in CANONICAL_ENTRY_POINT_GROUPS:
        values = _values_for_group(raw, group)
        if isinstance(values, Iterable):
            points.extend(point for point in values if isinstance(point, EntryPoint))
    group_order = {group: index for index, group in enumerate(CANONICAL_ENTRY_POINT_GROUPS)}

    def sort_key(point: EntryPoint) -> tuple[object, ...]:
        """Return metadata-only, provider-order-independent declaration order.

        ``EntryPoint`` objects do not carry a public declaration identifier
        beyond their parsed fields.  Distribution name/version therefore
        establish the cross-distribution order for identical declarations;
        the remaining parsed declaration fields provide a deterministic final
        key without calling ``load`` or depending on object identity.
        """

        distribution = point.dist
        distribution_name = "" if distribution is None else str(distribution.name)
        distribution_version = "" if distribution is None else str(distribution.version)
        declaration_value = str(point.value)
        module, separator, attribute = declaration_value.partition(":")
        declaration_key = (
            str(point.group),
            str(point.name),
            declaration_value,
            module,
            separator,
            attribute,
        )
        return (
            group_order[point.group],
            str(point.name),
            distribution_name.casefold(),
            distribution_name,
            distribution_version,
            declaration_key,
        )

    return tuple(sorted(points, key=sort_key))


def list_entry_points(
    *,
    provider: Callable[[], object] | None = None,
) -> tuple[EntryPointMetadata, ...]:
    """List canonical plugin declarations without importing plugin code."""

    return tuple(EntryPointMetadata.from_entry_point(point) for point in _collect_entry_points(provider))


def load_entry_points(
    *,
    registry: Registry | None = None,
    provider: Callable[[], object] | None = None,
) -> DiscoveryReport:
    """Load canonical plugin declarations into a registry explicitly.

    Duplicate policy is deterministic and non-overriding: an existing
    registry name, or the first earlier declaration with that name, wins. A
    duplicate is recorded in ``DiscoveryReport.issues`` and its target is not
    loaded. Import and registration errors are recorded as isolated failures;
    other declarations continue processing.
    """

    target = GLOBAL_REGISTRY if registry is None else registry
    loaded: list[EntryPointMetadata] = []
    issues: list[DiscoveryIssue] = []
    claimed_names = {entry.name for entry in target.list()}

    for point in _collect_entry_points(provider):
        metadata = EntryPointMetadata.from_entry_point(point)
        if metadata.name in claimed_names:
            issues.append(
                DiscoveryIssue(
                    metadata=metadata,
                    reason=(
                        f"duplicate plugin name {metadata.name!r}; existing or earlier declaration wins, "
                        "so this entry point was not loaded"
                    ),
                )
            )
            continue
        claimed_names.add(metadata.name)
        try:
            factory = point.load()
            if not callable(factory):
                raise TypeError(f"entry point resolved to non-callable {type(factory).__name__}")
            api_version = getattr(factory, PLUGIN_API_VERSION_ATTRIBUTE, None)
            if api_version != PLUGIN_API_VERSION:
                raise PluginContractError(
                    f"plugin {metadata.name!r} declares API version {api_version!r}; "
                    f"latent-anything requires {PLUGIN_API_VERSION!r} via "
                    f"{PLUGIN_API_VERSION_ATTRIBUTE}"
                )
            target.register(
                metadata.registry_kind,
                metadata.name,
                factory,
                source="external",
                entry_point_group=metadata.group,
                entry_point_value=metadata.value,
                distribution=metadata.distribution,
                version=metadata.version,
                plugin_api_version=api_version,
            )
        except Exception as error:
            issues.append(
                DiscoveryIssue(
                    metadata=metadata,
                    reason=(f"failed to load/register plugin {metadata.name!r} from {metadata.value!r}: {error}"),
                    exception_type=type(error).__name__,
                )
            )
            continue
        loaded.append(metadata)

    return DiscoveryReport(loaded=tuple(loaded), issues=tuple(issues))


__all__ = [
    "DiscoveryIssue",
    "DiscoveryReport",
    "PluginContractError",
    "list_entry_points",
    "load_entry_points",
]
