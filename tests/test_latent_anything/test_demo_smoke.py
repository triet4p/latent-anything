"""Smoke tests: core helpers from end-to-end demo scripts still pass.

Verifies that the key imports and API surfaces used by every
``scripts/end_to_end_*.py`` demo still work after the registry
refactoring. This is a lightweight alternative to running the full
scripts (which may have heavy dependencies like matplotlib).
"""

from __future__ import annotations


class TestRegistryDemoSmoke:
    """Coverage for ``end_to_end_registry_demo.py``."""

    def test_global_registry_kind_constants_available(self) -> None:
        """GLOBAL_REGISTRY and kind constants are importable."""
        from latent_anything.registry import GLOBAL_REGISTRY, KIND_ADAPTER, KIND_METHOD_A, KIND_METHOD_B

        assert GLOBAL_REGISTRY.name == "global"
        assert KIND_ADAPTER == "adapter"
        assert KIND_METHOD_A == "method_a"
        assert KIND_METHOD_B == "method_b"

    def test_registry_listing_grouping_works(self) -> None:
        """Listing by kind gives the expected counts (core of the demo)."""
        from latent_anything.registry import GLOBAL_REGISTRY, KIND_ADAPTER, KIND_METHOD_A, KIND_METHOD_B

        adapters = GLOBAL_REGISTRY.list(KIND_ADAPTER)
        methods_a = GLOBAL_REGISTRY.list(KIND_METHOD_A)
        methods_b = GLOBAL_REGISTRY.list(KIND_METHOD_B)
        assert len(adapters) == 4
        assert len(methods_a) == 3
        assert len(methods_b) == 3
        assert len(GLOBAL_REGISTRY) >= 10


class TestConfigDemoSmoke:
    """Coverage for ``end_to_end_config_demo.py``."""

    def test_object_spec_importable(self) -> None:
        """ObjectSpec and build_from_config are importable."""
        from latent_anything.config import ObjectSpec, build_from_config

        spec = ObjectSpec(kind="method_a", name="pca", params={"n_components": 2})
        pca = build_from_config(spec)
        assert pca is not None

    def test_build_vae_from_config(self) -> None:
        """Build VAE from config (demo step 1)."""
        from latent_anything.adapters import VAE
        from latent_anything.config import ObjectSpec, build_from_config

        spec = ObjectSpec(
            kind="adapter",
            name="vae",
            params={
                "input_dim": 8,
                "latent_dim": 3,
                "learning_rate": 0.005,
                "n_epochs": 300,
                "beta": 1.0,
                "random_state": 42,
            },
        )
        vae = build_from_config(spec)
        assert isinstance(vae, VAE)

    def test_build_activation_patch_with_nested_adapter(self) -> None:
        """Build ActivationPatch with nested VAE (demo step 4)."""
        from latent_anything.config import ObjectSpec, build_from_config

        spec = ObjectSpec(
            kind="method_b",
            name="activation_patch",
            params={
                "adapter": ObjectSpec(kind="adapter", name="vae", params={"input_dim": 8, "latent_dim": 3}),
            },
        )
        patch = build_from_config(spec)
        assert patch is not None


class TestVaeDemoSmoke:
    """Coverage for ``end_to_end_vae_demo.py``."""

    def test_core_imports_work(self) -> None:
        """The core imports used by the VAE demo are available."""
        from latent_anything import Trajectory
        from latent_anything.adapters import VAE
        from latent_anything.methods import PCA, UMAP

        assert Trajectory is not None
        assert VAE is not None
        assert PCA is not None
        assert UMAP is not None


class TestOtherDemoSmokes:
    """Minimal import verification for remaining demo scripts."""

    def test_lerp_demo_imports(self) -> None:
        """Lerp demo imports work."""
        from latent_anything import Trajectory
        from latent_anything.methods import Lerp

        assert Lerp is not None
        assert Trajectory is not None

    def test_steering_demo_imports(self) -> None:
        """Steering demo imports work."""
        from latent_anything.methods import SteeringVector

        assert SteeringVector is not None

    def test_sae_demo_imports(self) -> None:
        """SAE demo imports work."""
        from latent_anything.methods import SAE

        assert SAE is not None

    def test_gaussian_renderer_demo_imports(self) -> None:
        """Gaussian renderer demo imports work."""
        from latent_anything.adapters import GaussianRendererAdapter

        assert GaussianRendererAdapter is not None

    def test_hidden_state_demo_imports(self) -> None:
        """Hidden state demo imports work."""
        from latent_anything.adapters import HiddenStateAdapter

        assert HiddenStateAdapter is not None

    def test_random_projection_demo_imports(self) -> None:
        """Random projection demo imports work."""
        from latent_anything.adapters import RandomProjection

        assert RandomProjection is not None

    def test_activation_patch_demo_imports(self) -> None:
        """Activation patch demo imports work."""
        from latent_anything.methods import ActivationPatch

        assert ActivationPatch is not None

    def test_pca_demo_imports(self) -> None:
        """PCA demo imports work."""
        from latent_anything.methods import PCA

        assert PCA is not None

    def test_umap_demo_imports(self) -> None:
        """UMAP demo imports work."""
        from latent_anything.methods import UMAP

        assert UMAP is not None

    def test_batch_executor_demo_imports(self) -> None:
        """Batch executor demo imports work."""
        from latent_anything.runtime import BatchExecutor

        assert BatchExecutor is not None

    def test_cache_demo_imports(self) -> None:
        """Cache demo imports work."""
        from latent_anything.runtime import InMemoryCache

        assert InMemoryCache is not None

    def test_async_runtime_demo_imports(self) -> None:
        """Async runtime demo imports work."""
        from latent_anything.runtime import RuntimeProfiler

        assert RuntimeProfiler is not None
