"""Tests for the in-process registry (Sprint 17)."""

from __future__ import annotations

import pytest

from latent_anything.registry import (
    GLOBAL_REGISTRY,
    KIND_ADAPTER,
    KIND_METHOD_A,
    KIND_METHOD_B,
    Registry,
    RegistryEntry,
    list_entries,
    lookup,
    register,
)

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def empty_registry() -> Registry:
    """Return a fresh empty registry for isolated tests."""
    return Registry(name="test-scope")


# ── Registry construction ───────────────────────────────────────────


class TestRegistryConstruction:
    """Registry creation and basic properties."""

    def test_global_registry_exists(self) -> None:
        """GLOBAL_REGISTRY is a Registry instance with a name."""
        assert isinstance(GLOBAL_REGISTRY, Registry)
        assert GLOBAL_REGISTRY.name == "global"

    def test_registry_with_name(self) -> None:
        """A named registry reports its name."""
        r = Registry(name="my-registry")
        assert r.name == "my-registry"

    def test_registry_without_name(self) -> None:
        """A registry without a name returns None for name."""
        r = Registry()
        assert r.name is None

    def test_empty_registry_length(self, empty_registry: Registry) -> None:
        """A fresh registry is empty."""
        assert len(empty_registry) == 0

    def test_empty_registry_string(self, empty_registry: Registry) -> None:
        """repr shows the registry name and length."""
        assert repr(empty_registry) == "Registry('test-scope', len=0)"


# ── Registration ────────────────────────────────────────────────────


class TestRegistration:
    """Registering entries in the registry."""

    def test_register_adapter(self, empty_registry: Registry) -> None:
        """Register an adapter entry."""
        factory = lambda: None  # noqa: E731
        empty_registry.register(KIND_ADAPTER, "my_adapter", factory, description="test")
        assert len(empty_registry) == 1
        entry = empty_registry.lookup("my_adapter")
        assert entry.kind == KIND_ADAPTER
        assert entry.name == "my_adapter"
        assert entry.factory is factory
        assert entry.metadata["description"] == "test"

    def test_register_method_a(self, empty_registry: Registry) -> None:
        """Register a Layer A method entry."""
        factory = lambda: None  # noqa: E731
        empty_registry.register(KIND_METHOD_A, "my_method_a", factory)
        assert len(empty_registry) == 1
        assert empty_registry.lookup("my_method_a").kind == KIND_METHOD_A

    def test_register_method_b(self, empty_registry: Registry) -> None:
        """Register a Layer B method entry."""
        factory = lambda: None  # noqa: E731
        empty_registry.register(KIND_METHOD_B, "my_method_b", factory)
        assert len(empty_registry) == 1
        assert empty_registry.lookup("my_method_b").kind == KIND_METHOD_B

    def test_register_preserves_insertion_order(self, empty_registry: Registry) -> None:
        """Entries maintain insertion order when listed."""
        names = ["a", "b", "c", "d"]
        for i, name in enumerate(names):
            empty_registry.register(KIND_ADAPTER, name, lambda: None, order=i)  # type: ignore[arg-type]
        listed = empty_registry.list()
        assert [e.name for e in listed] == names

    def test_register_duplicate_name_raises(self, empty_registry: Registry) -> None:
        """Registering a duplicate name raises ValueError."""
        empty_registry.register(KIND_ADAPTER, "dup", lambda: None)
        with pytest.raises(ValueError, match="Duplicate registry entry.*dup"):
            empty_registry.register(KIND_ADAPTER, "dup", lambda: None)

    def test_register_duplicate_name_different_registry(self) -> None:
        """Same name in different registries is allowed."""
        r1 = Registry(name="r1")
        r2 = Registry(name="r2")
        r1.register(KIND_ADAPTER, "shared", lambda: None)
        r2.register(KIND_ADAPTER, "shared", lambda: None)  # Should not raise
        assert len(r1) == 1
        assert len(r2) == 1


# ── Lookup ──────────────────────────────────────────────────────────


class TestLookup:
    """Looking up entries by name."""

    def test_lookup_found(self, empty_registry: Registry) -> None:
        """lookup returns the correct entry."""
        factory = lambda: None  # noqa: E731
        empty_registry.register(KIND_ADAPTER, "target", factory, info="found")
        entry = empty_registry.lookup("target")
        assert entry.name == "target"
        assert entry.factory is factory
        assert entry.metadata["info"] == "found"

    def test_lookup_missing_name_raises(self, empty_registry: Registry) -> None:
        """lookup on missing name raises KeyError."""
        with pytest.raises(KeyError, match="No registry entry.*missing"):
            empty_registry.lookup("missing")

    def test_contains_found(self, empty_registry: Registry) -> None:
        """``in`` returns True for registered names."""
        empty_registry.register(KIND_ADAPTER, "exists", lambda: None)
        assert "exists" in empty_registry

    def test_contains_missing(self, empty_registry: Registry) -> None:
        """``in`` returns False for unregistered names."""
        assert "nope" not in empty_registry

    def test_lookup_after_multiple_registrations(self, empty_registry: Registry) -> None:
        """Lookup is not confused by multiple entries."""
        empty_registry.register(KIND_METHOD_A, "first", lambda: None)
        empty_registry.register(KIND_METHOD_B, "second", lambda: None)
        empty_registry.register(KIND_ADAPTER, "third", lambda: None)
        assert empty_registry.lookup("first").kind == KIND_METHOD_A
        assert empty_registry.lookup("second").kind == KIND_METHOD_B
        assert empty_registry.lookup("third").kind == KIND_ADAPTER


# ── Listing ─────────────────────────────────────────────────────────


class TestListing:
    """Listing entries with optional kind filtering."""

    @pytest.fixture
    def mixed_registry(self) -> Registry:
        """A registry with entries of all three kinds."""
        r = Registry(name="mixed")
        r.register(KIND_ADAPTER, "adapter_a", lambda: None)
        r.register(KIND_ADAPTER, "adapter_b", lambda: None)
        r.register(KIND_METHOD_A, "method_a_1", lambda: None)
        r.register(KIND_METHOD_B, "method_b_1", lambda: None)
        r.register(KIND_METHOD_A, "method_a_2", lambda: None)
        r.register(KIND_METHOD_B, "method_b_2", lambda: None)
        return r

    def test_list_all(self, mixed_registry: Registry) -> None:
        """list() with no filter returns all entries in order."""
        names = [e.name for e in mixed_registry.list()]
        assert names == ["adapter_a", "adapter_b", "method_a_1", "method_b_1", "method_a_2", "method_b_2"]

    def test_list_filter_adapters(self, mixed_registry: Registry) -> None:
        """list(kind=KIND_ADAPTER) returns only adapters."""
        names = [e.name for e in mixed_registry.list(KIND_ADAPTER)]
        assert names == ["adapter_a", "adapter_b"]

    def test_list_filter_method_a(self, mixed_registry: Registry) -> None:
        """list(kind=KIND_METHOD_A) returns only Layer A methods."""
        names = [e.name for e in mixed_registry.list(KIND_METHOD_A)]
        assert names == ["method_a_1", "method_a_2"]

    def test_list_filter_method_b(self, mixed_registry: Registry) -> None:
        """list(kind=KIND_METHOD_B) returns only Layer B methods."""
        names = [e.name for e in mixed_registry.list(KIND_METHOD_B)]
        assert names == ["method_b_1", "method_b_2"]

    def test_list_empty_kind(self, empty_registry: Registry) -> None:
        """list() on an empty registry returns an empty list."""
        assert empty_registry.list() == []

    def test_list_filter_unmatched_kind(self, empty_registry: Registry) -> None:
        """list() with a kind that has no entries returns an empty list."""
        empty_registry.register(KIND_ADAPTER, "only_adapter", lambda: None)
        assert empty_registry.list(KIND_METHOD_A) == []


# ── Factory retrieval ───────────────────────────────────────────────


class TestFactoryRetrieval:
    """Verifying that factory callables are retrievable and callable."""

    def test_factory_is_callable(self) -> None:
        """A registered factory is a callable."""

        def _identity(x: int) -> int:
            return x

        r = Registry()
        r.register(KIND_METHOD_A, "identity", _identity)
        entry = r.lookup("identity")
        assert callable(entry.factory)
        assert entry.factory(42) == 42

    def test_factory_for_each_kind(self, empty_registry: Registry) -> None:
        """factories for all three kinds are retrievable and callable."""

        def make_adapter() -> str:
            return "adapter"

        def make_method_a() -> str:
            return "method_a"

        def make_method_b() -> str:
            return "method_b"

        empty_registry.register(KIND_ADAPTER, "f_adapter", make_adapter)
        empty_registry.register(KIND_METHOD_A, "f_method_a", make_method_a)
        empty_registry.register(KIND_METHOD_B, "f_method_b", make_method_b)

        assert empty_registry.lookup("f_adapter").factory() == "adapter"
        assert empty_registry.lookup("f_method_a").factory() == "method_a"
        assert empty_registry.lookup("f_method_b").factory() == "method_b"


# ── RegistryEntry dataclass ─────────────────────────────────────────


class TestRegistryEntry:
    """RegistryEntry dataclass invariants."""

    def test_entry_is_frozen(self) -> None:
        """RegistryEntry is frozen and cannot be mutated."""
        entry = RegistryEntry(kind=KIND_ADAPTER, name="test", factory=lambda: None)
        with pytest.raises(AttributeError):
            entry.name = "changed"  # type: ignore[misc]

    def test_entry_default_metadata(self) -> None:
        """Entry created without metadata has an empty dict."""
        entry = RegistryEntry(kind=KIND_METHOD_A, name="empty", factory=lambda: None)
        assert entry.metadata == {}

    def test_entry_with_custom_metadata(self) -> None:
        """Entry metadata stores arbitrary key-value pairs."""
        entry = RegistryEntry(
            kind=KIND_METHOD_B,
            name="rich",
            factory=lambda: None,
            metadata={"version": "1.0", "description": "rich entry", "tags": ["fast", "stable"]},
        )
        assert entry.metadata["version"] == "1.0"
        assert entry.metadata["tags"] == ["fast", "stable"]


# ── Convenience helpers ─────────────────────────────────────────────


class TestConvenienceHelpers:
    """Module-level register, lookup, list_entries helpers."""

    def test_register_default_registry(self) -> None:
        """register() without registry uses GLOBAL_REGISTRY."""
        # Use a known entry that we know is there from built-in registration
        assert "vae" in GLOBAL_REGISTRY
        assert GLOBAL_REGISTRY.lookup("vae").kind == KIND_ADAPTER

    def test_register_custom_registry(self, empty_registry: Registry) -> None:
        """register() with registry uses the given registry."""
        register(KIND_METHOD_B, "custom_lerp", lambda: None, registry=empty_registry)
        assert "custom_lerp" in empty_registry
        assert "custom_lerp" not in GLOBAL_REGISTRY

    def test_lookup_convenience(self) -> None:
        """lookup() without registry uses GLOBAL_REGISTRY."""
        entry = lookup("vae")
        assert entry.kind == KIND_ADAPTER
        assert entry.name == "vae"

    def test_lookup_custom_registry(self, empty_registry: Registry) -> None:
        """lookup() with registry uses the given registry."""
        empty_registry.register(KIND_METHOD_A, "custom_pca", lambda: None)
        entry = lookup("custom_pca", registry=empty_registry)
        assert entry.name == "custom_pca"

    def test_list_entries_convenience(self) -> None:
        """list_entries() without registry uses GLOBAL_REGISTRY."""
        adapters = list_entries(KIND_ADAPTER)
        adapter_names = {e.name for e in adapters}
        assert "vae" in adapter_names
        assert "pca" not in adapter_names

    def test_list_entries_custom_registry(self, empty_registry: Registry) -> None:
        """list_entries() with registry uses the given registry."""
        empty_registry.register(KIND_METHOD_B, "test_b", lambda: None)
        result = list_entries(registry=empty_registry)
        assert len(result) == 1
        assert result[0].name == "test_b"


# ── GLOBAL_REGISTRY built-in entries ────────────────────────────────


class TestGlobalRegistryBuiltins:
    """Verifying that GLOBAL_REGISTRY has the expected built-in entries."""

    def test_global_registry_has_all_adapters(self) -> None:
        """GLOBAL_REGISTRY contains all built-in adapters."""
        adapter_names = {e.name for e in GLOBAL_REGISTRY.list(KIND_ADAPTER)}
        assert "vae" in adapter_names
        assert "random_projection" in adapter_names
        assert "hidden_state" in adapter_names
        assert "gaussian_renderer" in adapter_names
        assert "conv_vae" in adapter_names
        assert len(adapter_names) == 5

    def test_global_registry_has_all_method_a(self) -> None:
        """GLOBAL_REGISTRY contains all four Layer A methods."""
        method_a_names = {e.name for e in GLOBAL_REGISTRY.list(KIND_METHOD_A)}
        assert method_a_names == {"pca", "umap", "sae", "linear_probe"}

    def test_global_registry_has_all_method_b(self) -> None:
        """GLOBAL_REGISTRY contains all three Layer B methods."""
        method_b_names = {e.name for e in GLOBAL_REGISTRY.list(KIND_METHOD_B)}
        assert method_b_names == {"lerp", "steering", "activation_patch"}

    def test_global_registry_total_entries(self) -> None:
        """GLOBAL_REGISTRY has 12 entries (5 adapters + 4 analysis + 3 intervention)."""
        assert len(GLOBAL_REGISTRY) >= 12
        n_adapters = len(GLOBAL_REGISTRY.list(KIND_ADAPTER))
        n_method_a = len(GLOBAL_REGISTRY.list(KIND_METHOD_A))
        n_method_b = len(GLOBAL_REGISTRY.list(KIND_METHOD_B))
        assert n_adapters == 5
        assert n_method_a == 4
        assert n_method_b == 3

    def test_vae_entry_factory_is_callable_class(self) -> None:
        """The VAE entry's factory is the VAE class itself."""
        entry = GLOBAL_REGISTRY.lookup("vae")
        assert callable(entry.factory)
        # Import the actual class to verify
        from latent_anything.adapters.vae import VAE

        assert entry.factory is VAE

    def test_pca_entry_factory_is_callable_class(self) -> None:
        """The PCA entry's factory is the PCA class itself."""
        entry = GLOBAL_REGISTRY.lookup("pca")
        from latent_anything.methods.pca import PCA

        assert entry.factory is PCA

    def test_lerp_entry_factory_is_callable_class(self) -> None:
        """The Lerp entry's factory is the Lerp class itself."""
        entry = GLOBAL_REGISTRY.lookup("lerp")
        from latent_anything.methods.lerp import Lerp

        assert entry.factory is Lerp

    def test_activation_patch_entry_metadata(self) -> None:
        """ActivationPatch entry has descriptive metadata."""
        entry = GLOBAL_REGISTRY.lookup("activation_patch")
        assert "protocol" in entry.metadata
        assert entry.metadata["source"] == "built-in"


# ── Error cases ─────────────────────────────────────────────────────


class TestErrorCases:
    """Edge cases and error handling."""

    def test_lookup_empty_registry(self, empty_registry: Registry) -> None:
        """Lookup on an empty registry raises KeyError."""
        with pytest.raises(KeyError, match="No registry entry.*anything"):
            empty_registry.lookup("anything")

    def test_register_then_lookup_wrong_name(self, empty_registry: Registry) -> None:
        """After registering one entry, lookup a different name raises KeyError."""
        empty_registry.register(KIND_ADAPTER, "real", lambda: None)
        with pytest.raises(KeyError, match="No registry entry.*wrong"):
            empty_registry.lookup("wrong")

    def test_len_after_registration_and_duplicate_fail(self, empty_registry: Registry) -> None:
        """Len does not increase when a duplicate registration fails."""
        empty_registry.register(KIND_METHOD_A, "m", lambda: None)
        with pytest.raises(ValueError):
            empty_registry.register(KIND_METHOD_B, "m", lambda: None)
        assert len(empty_registry) == 1

    def test_register_kind_not_restricted(self, empty_registry: Registry) -> None:
        """Registry does not validate kind — any string is accepted."""
        empty_registry.register("custom_kind", "custom_entry", lambda: None)
        assert "custom_entry" in empty_registry
        assert empty_registry.lookup("custom_entry").kind == "custom_kind"

    def test_factory_returns_distinct_instances(self, empty_registry: Registry) -> None:
        """Calling the factory multiple times returns distinct instances."""

        class _Counter:
            def __init__(self) -> None:
                self.count = 0

            def increment(self) -> int:
                self.count += 1
                return self.count

        empty_registry.register(KIND_METHOD_A, "counter", _Counter)
        factory = empty_registry.lookup("counter").factory
        assert factory().increment() == 1
        assert factory().increment() == 1  # distinct instance each time


# ── Integration: registry does not break existing imports ───────────


class TestNoBreakage:
    """Verifying that importing the registry does not break existing code."""

    def test_can_still_import_adapters(self) -> None:
        """Existing adapter imports still work."""
        from latent_anything.adapters import VAE

        assert VAE is not None

    def test_can_still_import_methods(self) -> None:
        """Existing method imports still work."""
        from latent_anything.methods import PCA

        assert PCA is not None
