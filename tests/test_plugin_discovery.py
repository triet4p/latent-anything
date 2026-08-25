"""Focused tests for lazy and isolated external-plugin discovery."""

from __future__ import annotations

from importlib.metadata import EntryPoint
from types import SimpleNamespace

from latent_anything.plugin_discovery import DiscoveryReport, list_entry_points, load_entry_points
from latent_anything.plugin_groups import ENTRY_POINT_GROUP_ADAPTER
from latent_anything.plugin_metadata import PLUGIN_API_VERSION_ATTRIBUTE
from latent_anything.registry import GLOBAL_REGISTRY, Registry
from tests.plugin_harness import assert_external_entry_provenance


class _FakeEntryPoint(EntryPoint):
    """EntryPoint test double whose target load is observable."""

    def __init__(
        self,
        name: str,
        target: object,
        *,
        value: str,
        distribution_name: str | None = None,
        distribution_version: str | None = None,
        add_api_marker: bool = True,
    ) -> None:
        super().__init__(name=name, value=value, group=ENTRY_POINT_GROUP_ADAPTER)
        vars(self).update(_target=target, load_calls=0)
        if distribution_name is not None:
            vars(self).update(dist=SimpleNamespace(name=distribution_name, version=distribution_version or ""))
        if callable(target) and add_api_marker:
            setattr(target, PLUGIN_API_VERSION_ATTRIBUTE, "1")

    @property
    def load_calls(self) -> int:
        """Return the number of target-load attempts."""

        return int(vars(self)["load_calls"])

    def load(self) -> object:
        calls = int(vars(self)["load_calls"]) + 1
        vars(self).update(load_calls=calls)
        target = vars(self)["_target"]
        if isinstance(target, BaseException):
            raise target
        return target


def _provider(points: list[EntryPoint]):
    return lambda: {ENTRY_POINT_GROUP_ADAPTER: points}


def test_listing_is_lazy_and_sorted() -> None:
    """Listing exposes metadata without calling plugin targets."""

    first = _FakeEntryPoint("zeta", lambda: "z", value="fixture:z")
    second = _FakeEntryPoint("alpha", lambda: "a", value="fixture:a")
    metadata = list_entry_points(provider=_provider([first, second]))

    assert [item.name for item in metadata] == ["alpha", "zeta"]
    assert first.load_calls == 0
    assert second.load_calls == 0


def test_loading_is_deterministic_and_records_provenance() -> None:
    """Explicit loading registers callable factories and metadata."""

    point = _FakeEntryPoint("hello", lambda: "hello", value="fixture:hello")
    registry = Registry("external-test")
    report = load_entry_points(registry=registry, provider=_provider([point]))

    assert [item.name for item in report.loaded] == ["hello"]
    assert report.issues == ()
    assert point.load_calls == 1
    entry = registry.lookup("hello")
    assert entry.factory() == "hello"
    assert_external_entry_provenance(
        registry,
        "hello",
        group=ENTRY_POINT_GROUP_ADAPTER,
        distribution=None,
        version=None,
        api_version="1",
    )
    assert entry.metadata["entry_point_value"] == "fixture:hello"


def test_duplicate_names_do_not_override_or_load() -> None:
    """Existing entries and earlier sorted declarations win duplicates."""

    earlier = _FakeEntryPoint("same", lambda: "earlier", value="fixture:earlier")
    later = _FakeEntryPoint("same", lambda: "later", value="fixture:later")
    registry = Registry("external-test")
    registry.register("adapter", "same", lambda: "existing")

    report = load_entry_points(registry=registry, provider=_provider([later, earlier]))

    assert report.loaded == ()
    assert len(report.duplicates) == 2
    assert not report.failures
    assert earlier.load_calls == 0
    assert later.load_calls == 0
    assert registry.lookup("same").factory() == "existing"


def test_sorted_first_declaration_wins_external_duplicate() -> None:
    """When no built-in exists, sorted declaration order chooses the winner."""

    later = _FakeEntryPoint("same", lambda: "later", value="fixture:z-later")
    earlier = _FakeEntryPoint("same", lambda: "earlier", value="fixture:a-earlier")
    registry = Registry("external-test")

    report = load_entry_points(registry=registry, provider=_provider([later, earlier]))

    assert [item.name for item in report.loaded] == ["same"]
    assert len(report.duplicates) == 1
    assert registry.lookup("same").factory() == "earlier"
    assert earlier.load_calls == 1
    assert later.load_calls == 0


def test_identical_declarations_are_ordered_by_distribution_metadata() -> None:
    """Reversing provider order cannot change an identical declaration winner."""

    def run(points: list[EntryPoint]) -> tuple[Registry, DiscoveryReport]:
        registry = Registry("external-test")
        report = load_entry_points(registry=registry, provider=_provider(points))
        return registry, report

    alpha = _FakeEntryPoint(
        "same",
        lambda: "alpha",
        value="fixture:same",
        distribution_name="alpha-plugin",
        distribution_version="1.0",
    )
    zulu = _FakeEntryPoint(
        "same",
        lambda: "zulu",
        value="fixture:same",
        distribution_name="zulu-plugin",
        distribution_version="1.0",
    )
    first_registry, first_report = run([zulu, alpha])

    alpha_reversed = _FakeEntryPoint(
        "same",
        lambda: "alpha",
        value="fixture:same",
        distribution_name="alpha-plugin",
        distribution_version="1.0",
    )
    zulu_reversed = _FakeEntryPoint(
        "same",
        lambda: "zulu",
        value="fixture:same",
        distribution_name="zulu-plugin",
        distribution_version="1.0",
    )
    second_registry, second_report = run([alpha_reversed, zulu_reversed])

    assert first_registry.lookup("same").factory() == "alpha"
    assert second_registry.lookup("same").factory() == "alpha"
    assert first_report.issues[0].metadata.distribution == "zulu-plugin"
    assert second_report.issues[0].metadata.distribution == "zulu-plugin"
    assert [issue.metadata.distribution for issue in first_report.issues] == [
        issue.metadata.distribution for issue in second_report.issues
    ]
    assert alpha.load_calls == 1
    assert zulu.load_calls == 0
    assert alpha_reversed.load_calls == 1
    assert zulu_reversed.load_calls == 0


def test_one_broken_plugin_does_not_block_another() -> None:
    """Import/load failures are actionable and isolated."""

    broken = _FakeEntryPoint("broken", RuntimeError("optional dependency missing"), value="fixture:broken")
    healthy = _FakeEntryPoint("healthy", lambda: "ok", value="fixture:healthy")
    registry = Registry("external-test")

    report = load_entry_points(registry=registry, provider=_provider([broken, healthy]))

    assert [item.name for item in report.loaded] == ["healthy"]
    assert len(report.failures) == 1
    assert report.failures[0].metadata.name == "broken"
    assert report.failures[0].exception_type == "RuntimeError"
    assert "optional dependency missing" in report.failures[0].reason
    assert registry.lookup("healthy").factory() == "ok"


def test_unsupported_plugin_api_version_is_isolated() -> None:
    """A mismatched callable contract fails clearly without registration."""

    incompatible = _FakeEntryPoint("incompatible", lambda: "bad", value="fixture:incompatible")
    target = vars(incompatible)["_target"]
    setattr(target, PLUGIN_API_VERSION_ATTRIBUTE, "999")
    registry = Registry("external-test")

    report = load_entry_points(registry=registry, provider=_provider([incompatible]))

    assert report.loaded == ()
    assert len(report.failures) == 1
    assert report.failures[0].exception_type == "PluginContractError"
    assert "requires '1'" in report.failures[0].reason
    assert "incompatible" not in registry


def test_missing_plugin_api_marker_is_isolated_without_disrupting_peers() -> None:
    """A missing marker is actionable while a compatible peer still loads."""

    missing = _FakeEntryPoint(
        "missing-marker",
        lambda: "missing",
        value="fixture:missing-marker",
        add_api_marker=False,
    )
    healthy = _FakeEntryPoint("healthy-marker", lambda: "healthy", value="fixture:healthy-marker")
    registry = Registry("external-test")

    report = load_entry_points(registry=registry, provider=_provider([missing, healthy]))

    assert [item.name for item in report.loaded] == ["healthy-marker"]
    assert len(report.failures) == 1
    failure = report.failures[0]
    assert failure.metadata.name == "missing-marker"
    assert failure.exception_type == "PluginContractError"
    assert "declares API version None" in failure.reason
    assert "requires '1'" in failure.reason
    assert "missing-marker" not in registry
    assert registry.lookup("healthy-marker").factory() == "healthy"


def test_importing_discovery_preserves_builtins() -> None:
    """The discovery module does not alter built-in registration/imports."""

    before = [(entry.kind, entry.name, entry.factory) for entry in GLOBAL_REGISTRY.list()]
    __import__("latent_anything.plugin_discovery")
    after = [(entry.kind, entry.name, entry.factory) for entry in GLOBAL_REGISTRY.list()]

    assert after == before
    assert GLOBAL_REGISTRY.lookup("vae").metadata["source"] == "built-in"
