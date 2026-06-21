"""Tests for registry-backed config instantiation (Sprint 18)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from latent_anything.adapters import VAE
from latent_anything.adapters.random_projection import RandomProjection
from latent_anything.config import ObjectSpec, build_from_config, build_from_dict
from latent_anything.methods.activation_patch import ActivationPatch
from latent_anything.methods.lerp import Lerp
from latent_anything.methods.pca import PCA
from latent_anything.methods.steering import SteeringVector
from latent_anything.registry import KIND_ADAPTER, KIND_METHOD_A, Registry

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def empty_registry() -> Registry:
    """Return a fresh empty registry for isolation."""
    return Registry(name="test-scope")


# ── ObjectSpec construction ─────────────────────────────────────────


class TestObjectSpec:
    """ObjectSpec model invariants."""

    def test_minimal_spec(self) -> None:
        """A spec with only kind and name has empty params."""
        spec = ObjectSpec(kind="adapter", name="vae")
        assert spec.kind == "adapter"
        assert spec.name == "vae"
        assert spec.params == {}

    def test_spec_with_params(self) -> None:
        """A spec with params stores them."""
        spec = ObjectSpec(kind="method_a", name="pca", params={"n_components": 3})
        assert spec.params == {"n_components": 3}

    def test_empty_name_rejected(self) -> None:
        """An empty name raises pydantic ValidationError."""
        with pytest.raises(ValidationError):
            ObjectSpec(kind="adapter", name="")

    def test_from_dict_flat(self) -> None:
        """from_dict constructs a spec from a plain dict."""
        spec = ObjectSpec.from_dict({"kind": "method_a", "name": "pca", "params": {"n_components": 3}})
        assert spec.kind == "method_a"
        assert spec.name == "pca"
        assert spec.params == {"n_components": 3}

    def test_from_dict_with_nested_spec(self) -> None:
        """from_dict coarses nested ObjectSpec-compatible dicts."""
        data = {
            "kind": "method_b",
            "name": "activation_patch",
            "params": {
                "adapter": {"kind": "adapter", "name": "vae", "params": {"input_dim": 8, "latent_dim": 3}},
            },
        }
        spec = ObjectSpec.from_dict(data)
        assert spec.kind == "method_b"
        assert isinstance(spec.params["adapter"], dict)
        assert spec.params["adapter"]["name"] == "vae"


# ── build_from_config — Adapters ────────────────────────────────────


class TestBuildAdapter:
    """Building adapters from config specs."""

    def test_build_vae(self) -> None:
        """Build a VAE with input_dim and latent_dim."""
        spec = ObjectSpec(kind="adapter", name="vae", params={"input_dim": 8, "latent_dim": 3})
        vae = build_from_config(spec)
        assert isinstance(vae, VAE)
        assert vae.input_dim == 8
        assert vae.latent_dim == 3

    def test_build_vae_with_optional_params(self) -> None:
        """Build a VAE with optional params."""
        spec = ObjectSpec(
            kind="adapter",
            name="vae",
            params={"input_dim": 16, "latent_dim": 4, "learning_rate": 0.01, "beta": 0.5},
        )
        vae = build_from_config(spec)
        assert vae.input_dim == 16
        assert vae.latent_dim == 4
        assert vae.learning_rate == 0.01
        assert vae.beta == 0.5

    def test_build_random_projection(self) -> None:
        """Build a RandomProjection."""
        spec = ObjectSpec(kind="adapter", name="random_projection", params={"input_dim": 10, "latent_dim": 3})
        rp = build_from_config(spec)
        assert isinstance(rp, RandomProjection)
        assert rp.input_dim == 10
        assert rp.latent_dim == 3

    def test_build_random_projection_with_seed(self) -> None:
        """Build a RandomProjection with reproducible seed."""
        spec = ObjectSpec(
            kind="adapter",
            name="random_projection",
            params={"input_dim": 10, "latent_dim": 3, "random_state": 42},
        )
        rp1 = build_from_config(spec)

        spec2 = ObjectSpec(
            kind="adapter",
            name="random_projection",
            params={"input_dim": 10, "latent_dim": 3, "random_state": 42},
        )
        rp2 = build_from_config(spec2)

        import numpy as np

        assert np.allclose(rp1.projection_matrix_, rp2.projection_matrix_)


# ── build_from_config — Layer A Methods ─────────────────────────────


class TestBuildMethodA:
    """Building Layer A methods from config specs."""

    def test_build_pca(self) -> None:
        """Build PCA with n_components."""
        spec = ObjectSpec(kind="method_a", name="pca", params={"n_components": 5})
        pca = build_from_config(spec)
        assert isinstance(pca, PCA)
        assert pca.n_components == 5

    def test_build_pca_default(self) -> None:
        """Build PCA with no params (None n_components)."""
        spec = ObjectSpec(kind="method_a", name="pca")
        pca = build_from_config(spec)
        assert isinstance(pca, PCA)
        assert pca.n_components is None

    def test_build_umap(self) -> None:
        """Build UMAP with params."""
        spec = ObjectSpec(kind="method_a", name="umap", params={"n_neighbors": 10, "n_components": 3})
        umap = build_from_config(spec)
        from latent_anything.methods.umap import UMAP

        assert isinstance(umap, UMAP)
        assert umap.n_neighbors == 10
        assert umap.n_components == 3

    def test_build_sae(self) -> None:
        """Build SAE with n_components."""
        spec = ObjectSpec(kind="method_a", name="sae", params={"n_components": 8})
        sae = build_from_config(spec)
        from latent_anything.methods.sae import SAE

        assert isinstance(sae, SAE)
        assert sae.n_components == 8


# ── build_from_config — Layer B Methods ─────────────────────────────


class TestBuildMethodB:
    """Building Layer B methods from config specs."""

    def test_build_lerp(self) -> None:
        """Build a Lerp (no params needed)."""
        spec = ObjectSpec(kind="method_b", name="lerp")
        lerp = build_from_config(spec)
        assert isinstance(lerp, Lerp)
        assert lerp.space is None

    def test_build_steering(self) -> None:
        """Build a SteeringVector (no params needed)."""
        spec = ObjectSpec(kind="method_b", name="steering")
        sv = build_from_config(spec)
        assert isinstance(sv, SteeringVector)
        assert sv.space is None

    def test_build_activation_patch_with_nested_vae(self) -> None:
        """Build ActivationPatch with a nested VAE adapter spec."""
        spec = ObjectSpec(
            kind="method_b",
            name="activation_patch",
            params={
                "adapter": ObjectSpec(kind="adapter", name="vae", params={"input_dim": 8, "latent_dim": 3}),
            },
        )
        patch = build_from_config(spec)
        assert isinstance(patch, ActivationPatch)
        assert patch.is_fitted is False  # freshly built, not fitted

    def test_build_activation_patch_from_dict(self) -> None:
        """Build ActivationPatch with nested VAE from a plain dict."""
        data = {
            "kind": "method_b",
            "name": "activation_patch",
            "params": {
                "adapter": {"kind": "adapter", "name": "vae", "params": {"input_dim": 8, "latent_dim": 3}},
            },
        }
        patch = build_from_dict(data)
        assert isinstance(patch, ActivationPatch)
        # The adapter inside is a VAE
        assert isinstance(patch._adapter, VAE)  # type: ignore[attr-defined]

    def test_build_activation_patch_with_random_projection(self) -> None:
        """Build ActivationPatch with a RandomProjection adapter."""
        spec = ObjectSpec(
            kind="method_b",
            name="activation_patch",
            params={
                "adapter": ObjectSpec(
                    kind="adapter",
                    name="random_projection",
                    params={"input_dim": 10, "latent_dim": 3, "random_state": 42},
                ),
            },
        )
        patch = build_from_config(spec)
        assert isinstance(patch, ActivationPatch)
        assert isinstance(patch._adapter, RandomProjection)  # type: ignore[attr-defined]


# ── Error cases ─────────────────────────────────────────────────────


class TestBuildErrors:
    """Clear validation errors for invalid specs."""

    def test_unknown_name_raises_keyerror(self) -> None:
        """Unknown name should raise KeyError with descriptive message."""
        spec = ObjectSpec(kind="adapter", name="nonexistent_thing")
        with pytest.raises(KeyError) as exc:
            build_from_config(spec)
        msg = str(exc.value)
        assert "nonexistent_thing" in msg
        assert "Available names" in msg

    def test_unknown_name_in_custom_registry(self, empty_registry: Registry) -> None:
        """Unknown name in custom registry should include registry name."""
        spec = ObjectSpec(kind="adapter", name="missing")
        with pytest.raises(KeyError) as exc:
            build_from_config(spec, registry=empty_registry)
        msg = str(exc.value)
        assert "missing" in msg
        assert "test-scope" in msg

    def test_kind_mismatch_raises_valueerror(self) -> None:
        """Kind mismatch should raise ValueError with descriptive message."""
        # vae is kind=adapter, but we request kind=method_a
        spec = ObjectSpec(kind="method_a", name="vae", params={"input_dim": 8, "latent_dim": 3})
        with pytest.raises(ValueError) as exc:
            build_from_config(spec)
        msg = str(exc.value)
        assert "kind mismatch" in msg.lower() or "kind" in msg
        assert "vae" in msg

    def test_missing_required_param_raises_typeerror(self) -> None:
        """Missing required constructor param should raise TypeError."""
        # VAE requires input_dim and latent_dim
        spec = ObjectSpec(kind="adapter", name="vae", params={})
        with pytest.raises(TypeError) as exc:
            build_from_config(spec)
        msg = str(exc.value)
        assert "vae" in msg
        assert "missing" in msg.lower() or "required" in msg.lower() or "argument" in msg.lower()

    def test_wrong_param_type_raises_typeerror(self) -> None:
        """Wrong parameter type should raise TypeError."""
        spec = ObjectSpec(kind="adapter", name="vae", params={"input_dim": "not_an_int", "latent_dim": 3})
        with pytest.raises(TypeError) as exc:
            build_from_config(spec)
        msg = str(exc.value)
        assert "vae" in msg

    def test_unknown_nested_adapter_raises_keyerror(self) -> None:
        """Unknown adapter name inside nested spec should raise KeyError."""
        spec = ObjectSpec(
            kind="method_b",
            name="activation_patch",
            params={
                "adapter": ObjectSpec(kind="adapter", name="unknown_adapter", params={"input_dim": 8}),
            },
        )
        with pytest.raises(KeyError) as exc:
            build_from_config(spec)
        assert "unknown_adapter" in str(exc.value)

    def test_custom_registry_with_no_entries(self, empty_registry: Registry) -> None:
        """build_from_config on empty registry raises KeyError."""
        spec = ObjectSpec(kind="adapter", name="anything")
        with pytest.raises(KeyError):
            build_from_config(spec, registry=empty_registry)


# ── build_from_config — custom registry ─────────────────────────────


class TestCustomRegistry:
    """Config instantiation with non-global registries."""

    def test_build_from_custom_registry(self, empty_registry: Registry) -> None:
        """Build from a custom (non-global) registry."""

        # Register a simple factory
        def _make_pca(n_components: int = 2) -> PCA:
            return PCA(n_components=n_components)

        empty_registry.register(KIND_METHOD_A, "custom_pca", _make_pca, description="custom PCA factory")

        spec = ObjectSpec(kind="method_a", name="custom_pca", params={"n_components": 4})
        pca = build_from_config(spec, registry=empty_registry)
        assert isinstance(pca, PCA)
        assert pca.n_components == 4

    def test_custom_registry_not_in_global(self, empty_registry: Registry) -> None:
        """Entries in a custom registry are not visible to GLOBAL_REGISTRY."""
        empty_registry.register(KIND_ADAPTER, "secret", lambda: "secret-value")
        spec = ObjectSpec(kind="adapter", name="secret")
        result = build_from_config(spec, registry=empty_registry)
        assert result == "secret-value"

        # Not in global
        with pytest.raises(KeyError):
            build_from_config(spec)


# ── build_from_dict convenience ─────────────────────────────────────


class TestBuildFromDict:
    """The build_from_dict convenience wrapper."""

    def test_build_from_dict_pca(self) -> None:
        """Build PCA from a plain dict."""
        pca = build_from_dict({"kind": "method_a", "name": "pca", "params": {"n_components": 3}})
        assert isinstance(pca, PCA)
        assert pca.n_components == 3

    def test_build_from_dict_lerp(self) -> None:
        """Build Lerp from a plain dict with no params."""
        lerp = build_from_dict({"kind": "method_b", "name": "lerp"})
        assert isinstance(lerp, Lerp)

    def test_build_from_dict_activation_patch(self) -> None:
        """Build ActivationPatch from a plain dict with nested adapter."""
        data = {
            "kind": "method_b",
            "name": "activation_patch",
            "params": {
                "adapter": {"kind": "adapter", "name": "vae", "params": {"input_dim": 8, "latent_dim": 3}},
            },
        }
        patch = build_from_dict(data)
        assert isinstance(patch, ActivationPatch)


# ── Build each required class ───────────────────────────────────────


class TestBuildAllRequired:
    """Verify all six required classes can be built from config."""

    def test_build_vae_from_config(self) -> None:
        """1. VAE"""
        obj = build_from_config(ObjectSpec(kind="adapter", name="vae", params={"input_dim": 8, "latent_dim": 3}))
        assert isinstance(obj, VAE)

    def test_build_random_projection_from_config(self) -> None:
        """2. RandomProjection"""
        obj = build_from_config(
            ObjectSpec(kind="adapter", name="random_projection", params={"input_dim": 10, "latent_dim": 3})
        )
        assert isinstance(obj, RandomProjection)

    def test_build_pca_from_config(self) -> None:
        """3. PCA"""
        obj = build_from_config(ObjectSpec(kind="method_a", name="pca", params={"n_components": 3}))
        assert isinstance(obj, PCA)

    def test_build_lerp_from_config(self) -> None:
        """4. Lerp"""
        obj = build_from_config(ObjectSpec(kind="method_b", name="lerp"))
        assert isinstance(obj, Lerp)

    def test_build_steering_from_config(self) -> None:
        """5. SteeringVector"""
        obj = build_from_config(ObjectSpec(kind="method_b", name="steering"))
        assert isinstance(obj, SteeringVector)

    def test_build_activation_patch_from_config(self) -> None:
        """6. ActivationPatch"""
        obj = build_from_config(
            ObjectSpec(
                kind="method_b",
                name="activation_patch",
                params={
                    "adapter": ObjectSpec(kind="adapter", name="vae", params={"input_dim": 8, "latent_dim": 3}),
                },
            )
        )
        assert isinstance(obj, ActivationPatch)
