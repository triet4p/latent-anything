"""Parity tests: registry constructor vs direct import constructor.

Verifies that every built-in adapter and method produces the **same type**
of instance whether constructed via ``GLOBAL_REGISTRY.lookup().factory(...)``
or via direct import. This proves "registry-first" does not change behavior.
"""

from __future__ import annotations

from latent_anything.adapters import VAE, ConvVAE, GaussianRendererAdapter, HiddenStateAdapter, RandomProjection
from latent_anything.methods import PCA, SAE, UMAP, ActivationPatch, Lerp, SteeringVector
from latent_anything.registry import GLOBAL_REGISTRY

# ── Adapter parity ──────────────────────────────────────────────────


class TestAdapterParity:
    """Registry-constructed adapters have same type as direct imports."""

    def test_vae_parity(self) -> None:
        """Registry-constructed VAE is instance of directly imported VAE."""
        entry = GLOBAL_REGISTRY.lookup("vae")
        params = {"input_dim": 8, "latent_dim": 3}
        via_registry = entry.factory(**params)
        via_direct = VAE(**params)
        assert type(via_registry) is type(via_direct)
        assert via_registry.input_dim == via_direct.input_dim
        assert via_registry.latent_dim == via_direct.latent_dim

    def test_random_projection_parity(self) -> None:
        """Registry-constructed RandomProjection has same type."""
        entry = GLOBAL_REGISTRY.lookup("random_projection")
        params = {"input_dim": 10, "latent_dim": 4}
        via_registry = entry.factory(**params)
        via_direct = RandomProjection(**params)
        assert type(via_registry) is type(via_direct)

    def test_hidden_state_parity(self) -> None:
        """Registry-constructed HiddenStateAdapter has same type."""
        entry = GLOBAL_REGISTRY.lookup("hidden_state")
        params = {"input_dim": 16, "hidden_dim": 8}
        via_registry = entry.factory(**params)
        via_direct = HiddenStateAdapter(**params)
        assert type(via_registry) is type(via_direct)

    def test_gaussian_renderer_parity(self) -> None:
        """Registry-constructed GaussianRendererAdapter has same type."""
        entry = GLOBAL_REGISTRY.lookup("gaussian_renderer")
        params = {"n_gaussians": 10, "img_height": 28, "img_width": 28}
        via_registry = entry.factory(**params)
        via_direct = GaussianRendererAdapter(**params)
        assert type(via_registry) is type(via_direct)

    def test_conv_vae_parity(self) -> None:
        """Registry-constructed ConvVAE has the direct-import type."""
        via_registry = GLOBAL_REGISTRY.lookup("conv_vae").factory(latent_dim=3)
        via_direct = ConvVAE(latent_dim=3)
        assert type(via_registry) is type(via_direct)


# ── Layer A method parity ───────────────────────────────────────────


class TestMethodAParity:
    """Registry-constructed Layer A methods have same type as direct imports."""

    def test_pca_parity(self) -> None:
        """Registry-constructed PCA has same type."""
        entry = GLOBAL_REGISTRY.lookup("pca")
        params = {"n_components": 3}
        via_registry = entry.factory(**params)
        via_direct = PCA(**params)
        assert type(via_registry) is type(via_direct)
        assert via_registry.n_components == via_direct.n_components

    def test_pca_default_parity(self) -> None:
        """Registry-constructed PCA with no args has same type."""
        entry = GLOBAL_REGISTRY.lookup("pca")
        via_registry = entry.factory()
        via_direct = PCA()
        assert type(via_registry) is type(via_direct)
        assert via_registry.n_components == via_direct.n_components

    def test_umap_parity(self) -> None:
        """Registry-constructed UMAP has same type."""
        entry = GLOBAL_REGISTRY.lookup("umap")
        via_registry = entry.factory(n_neighbors=10, n_components=3)
        via_direct = UMAP(n_neighbors=10, n_components=3)
        assert type(via_registry) is type(via_direct)

    def test_umap_default_parity(self) -> None:
        """Registry-constructed UMAP with defaults has same type."""
        entry = GLOBAL_REGISTRY.lookup("umap")
        via_registry = entry.factory()
        via_direct = UMAP()
        assert type(via_registry) is type(via_direct)

    def test_sae_parity(self) -> None:
        """Registry-constructed SAE has same type."""
        entry = GLOBAL_REGISTRY.lookup("sae")
        params = {"n_components": 8}
        via_registry = entry.factory(**params)
        via_direct = SAE(**params)
        assert type(via_registry) is type(via_direct)


# ── Layer B method parity ───────────────────────────────────────────


class TestMethodBParity:
    """Registry-constructed Layer B methods have same type as direct imports."""

    def test_lerp_parity(self) -> None:
        """Registry-constructed Lerp has same type."""
        entry = GLOBAL_REGISTRY.lookup("lerp")
        via_registry = entry.factory()
        via_direct = Lerp()
        assert type(via_registry) is type(via_direct)
        assert via_registry.space == via_direct.space

    def test_steering_parity(self) -> None:
        """Registry-constructed SteeringVector has same type."""
        entry = GLOBAL_REGISTRY.lookup("steering")
        via_registry = entry.factory()
        via_direct = SteeringVector()
        assert type(via_registry) is type(via_direct)

    def test_activation_patch_parity(self) -> None:
        """Registry-constructed ActivationPatch (with VAE) has same type."""
        entry = GLOBAL_REGISTRY.lookup("activation_patch")
        vae_entry = GLOBAL_REGISTRY.lookup("vae")
        adapter = vae_entry.factory(input_dim=8, latent_dim=3)
        via_registry = entry.factory(adapter=adapter)
        via_direct = ActivationPatch(adapter=adapter)
        assert type(via_registry) is type(via_direct)


# ── Factory identity parity ─────────────────────────────────────────


class TestFactoryIdentity:
    """The registry factory IS the class for built-in entries.

    Uses the public-API imports already at the module top-level rather
    than re-importing from private module paths.
    """

    def test_vae_factory_is_vae(self) -> None:
        """registry.lookup('vae').factory is VAE."""
        assert GLOBAL_REGISTRY.lookup("vae").factory is VAE

    def test_pca_factory_is_pca(self) -> None:
        """registry.lookup('pca').factory is PCA."""
        assert GLOBAL_REGISTRY.lookup("pca").factory is PCA

    def test_lerp_factory_is_lerp(self) -> None:
        """registry.lookup('lerp').factory is Lerp."""
        assert GLOBAL_REGISTRY.lookup("lerp").factory is Lerp

    def test_umap_factory_is_umap(self) -> None:
        """registry.lookup('umap').factory is UMAP."""
        assert GLOBAL_REGISTRY.lookup("umap").factory is UMAP

    def test_sae_factory_is_sae(self) -> None:
        """registry.lookup('sae').factory is SAE."""
        assert GLOBAL_REGISTRY.lookup("sae").factory is SAE

    def test_steering_factory_is_steering(self) -> None:
        """registry.lookup('steering').factory is SteeringVector."""
        assert GLOBAL_REGISTRY.lookup("steering").factory is SteeringVector

    def test_activation_patch_factory_is_class(self) -> None:
        """registry.lookup('activation_patch').factory is ActivationPatch."""
        assert GLOBAL_REGISTRY.lookup("activation_patch").factory is ActivationPatch

    def test_random_projection_factory_is_class(self) -> None:
        """registry.lookup('random_projection').factory is RandomProjection."""
        assert GLOBAL_REGISTRY.lookup("random_projection").factory is RandomProjection

    def test_hidden_state_factory_is_class(self) -> None:
        """registry.lookup('hidden_state').factory is HiddenStateAdapter."""
        assert GLOBAL_REGISTRY.lookup("hidden_state").factory is HiddenStateAdapter

    def test_gaussian_renderer_factory_is_class(self) -> None:
        """registry.lookup('gaussian_renderer').factory is GaussianRendererAdapter."""
        assert GLOBAL_REGISTRY.lookup("gaussian_renderer").factory is GaussianRendererAdapter
