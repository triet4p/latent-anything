"""Tests for AnalysisPipeline (Sprint 20 — Pipeline #1)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from pydantic import ValidationError

from latent_anything.adapters import VAE, HiddenStateAdapter
from latent_anything.config import ObjectSpec
from latent_anything.methods import PCA
from latent_anything.pipeline import (
    AnalysisPipeline,
    PipelineResult,
    PipelineSpec,
    build_pipeline_from_config,
)

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def small_vae() -> VAE:
    """Return a quickly-trained VAE for pipeline tests."""
    return VAE(input_dim=8, latent_dim=3, n_epochs=10, random_state=42)


@pytest.fixture
def simple_pca() -> PCA:
    """Return a PCA with n_components=2."""
    return PCA(n_components=2)


@pytest.fixture
def synthetic_data() -> np.ndarray:
    """Return synthetic [0,1]-scaled data (50 samples, 8 features)."""
    rng = np.random.default_rng(42)
    return rng.uniform(0, 1, size=(50, 8))


@pytest.fixture
def synthetic_hidden_data() -> np.ndarray:
    """Return synthetic normal data (30 samples, 10 features)."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(30, 10))


# ── Construction ────────────────────────────────────────────────────


class TestAnalysisPipelineConstruction:
    """AnalysisPipeline construction invariants."""

    def test_construct_with_vae_and_pca(self, small_vae: VAE, simple_pca: PCA) -> None:
        """Construct a pipeline with VAE adapter and PCA method."""
        pipeline = AnalysisPipeline(adapter=small_vae, method=simple_pca)
        assert pipeline.adapter is small_vae
        assert pipeline.method is simple_pca
        assert pipeline.latent_space.dim == 3

    def test_construct_with_hidden_state_and_pca(self, simple_pca: PCA) -> None:
        """Construct a pipeline with HiddenStateAdapter and PCA."""
        hsa = HiddenStateAdapter(input_dim=10, hidden_dim=4, random_state=42)
        pipeline = AnalysisPipeline(adapter=hsa, method=simple_pca)
        assert pipeline.adapter is hsa
        assert pipeline.latent_space.dim == 4

    def test_latent_space_property(self, small_vae: VAE, simple_pca: PCA) -> None:
        """latent_space property returns a LatentSpace with matching attributes."""
        pipeline = AnalysisPipeline(adapter=small_vae, method=simple_pca)
        ls = pipeline.latent_space
        assert ls.dim == small_vae.latent_dim
        assert ls.geometry == "euclidean"

    def test_adapter_and_method_accessible(self, small_vae: VAE, simple_pca: PCA) -> None:
        """adapter and method attributes are accessible."""
        pipeline = AnalysisPipeline(adapter=small_vae, method=simple_pca)
        assert pipeline.adapter is small_vae
        assert pipeline.method is simple_pca


# ── Run ─────────────────────────────────────────────────────────────


class TestAnalysisPipelineRun:
    """AnalysisPipeline.run() behavior."""

    def test_run_returns_typed_result(self, small_vae: VAE, simple_pca: PCA, synthetic_data: np.ndarray) -> None:
        """run() returns a PipelineResult with correct shapes."""
        small_vae.fit(synthetic_data)
        pipeline = AnalysisPipeline(adapter=small_vae, method=simple_pca)
        result = pipeline.run(synthetic_data)

        assert isinstance(result, PipelineResult)
        assert result.latents.shape == (50, 3)
        assert result.transformed.shape == (50, 2)
        assert result.latent_space.dim == 3

    def test_run_with_hidden_state_pca(self, simple_pca: PCA, synthetic_hidden_data: np.ndarray) -> None:
        """Run pipeline with HiddenStateAdapter + PCA."""
        hsa = HiddenStateAdapter(input_dim=10, hidden_dim=4, random_state=42)
        pipeline = AnalysisPipeline(adapter=hsa, method=simple_pca)
        result = pipeline.run(synthetic_hidden_data)

        assert isinstance(result, PipelineResult)
        assert result.latents.shape == (30, 4)
        assert result.transformed.shape == (30, 2)

    def test_run_deterministic(self, simple_pca: PCA, synthetic_data: np.ndarray) -> None:
        """Multiple runs produce consistent results (deterministic)."""
        vae = VAE(input_dim=8, latent_dim=3, n_epochs=10, random_state=42)
        vae.fit(synthetic_data)
        pipeline = AnalysisPipeline(adapter=vae, method=simple_pca)

        result1 = pipeline.run(synthetic_data)
        result2 = pipeline.run(synthetic_data)

        np.testing.assert_array_almost_equal(result1.transformed, result2.transformed)

    def test_run_with_fitted_vae(self, synthetic_data: np.ndarray) -> None:
        """VAE must be fitted before pipeline.run() — test the full flow."""
        vae = VAE(input_dim=8, latent_dim=3, n_epochs=5, random_state=42)
        vae.fit(synthetic_data)
        pca = PCA(n_components=2)
        pipeline = AnalysisPipeline(adapter=vae, method=pca)

        result = pipeline.run(synthetic_data)
        assert result.latents.shape == (50, 3)
        assert result.transformed.shape == (50, 2)

    def test_input_not_mutated(self, small_vae: VAE, simple_pca: PCA, synthetic_data: np.ndarray) -> None:
        """run() does not mutate the input array."""
        small_vae.fit(synthetic_data)
        pipeline = AnalysisPipeline(adapter=small_vae, method=simple_pca)
        original = synthetic_data.copy()
        pipeline.run(synthetic_data)
        np.testing.assert_array_equal(synthetic_data, original)

    def test_transformed_components_match(self, small_vae: VAE, synthetic_data: np.ndarray) -> None:
        """Transformed result matches manual encode→fit_transform."""
        small_vae.fit(synthetic_data)
        pca = PCA(n_components=2)
        pipeline = AnalysisPipeline(adapter=small_vae, method=pca)

        # Manual
        latents_manual = small_vae.encode(synthetic_data)
        pca_manual = PCA(n_components=2)
        transformed_manual = pca_manual.fit_transform(latents_manual)

        # Pipeline
        result = pipeline.run(synthetic_data)

        np.testing.assert_array_almost_equal(result.latents, latents_manual)
        np.testing.assert_array_almost_equal(result.transformed, transformed_manual)


# ── PipelineResult ──────────────────────────────────────────────────


class TestPipelineResult:
    """PipelineResult invariants."""

    def test_is_frozen_dataclass(self) -> None:
        """PipelineResult is a frozen dataclass with required fields."""
        assert dataclasses.is_dataclass(PipelineResult)
        frozen = PipelineResult.__dataclass_fields__["latents"]
        assert frozen  # exists
        assert PipelineResult.__dataclass_fields__["transformed"]
        assert PipelineResult.__dataclass_fields__["latent_space"]
        # Check frozen is not writable — frozen=True means no assignment
        # We test this by checking the frozen metadata
        assert PipelineResult.__dataclass_fields__["latents"].default is dataclasses.MISSING

    def test_result_immutable(self) -> None:
        """PipelineResult fields cannot be reassigned."""
        result = PipelineResult(
            latents=np.array([1.0]),
            transformed=np.array([2.0]),
            latent_space=__import__("latent_anything").LatentSpace(dim=3),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.latents = np.array([3.0])  # type: ignore[misc]


# ── Config-backed construction ──────────────────────────────────────


class TestPipelineSpec:
    """PipelineSpec model invariants."""

    def test_minimal_spec(self) -> None:
        """PipelineSpec with adapter and method specs."""
        spec = PipelineSpec(
            adapter=ObjectSpec(kind="adapter", name="vae", params={"input_dim": 8, "latent_dim": 3}),
            method=ObjectSpec(kind="method_a", name="pca", params={"n_components": 2}),
        )
        assert spec.adapter.name == "vae"
        assert spec.method.name == "pca"
        assert spec.adapter.kind == "adapter"
        assert spec.method.kind == "method_a"

    def test_spec_with_dict_auto_coercion(self) -> None:
        """PipelineSpec auto-coerces plain dicts to ObjectSpec."""
        spec = PipelineSpec(
            adapter={"kind": "adapter", "name": "vae", "params": {"input_dim": 8, "latent_dim": 3}},  # pyright: ignore[reportArgumentType]
            method={"kind": "method_a", "name": "pca", "params": {"n_components": 2}},  # pyright: ignore[reportArgumentType]
        )
        assert isinstance(spec.adapter, ObjectSpec)
        assert isinstance(spec.method, ObjectSpec)
        assert spec.adapter.name == "vae"
        assert spec.method.name == "pca"

    def test_spec_empty_adapter_name_rejected(self) -> None:
        """Empty adapter name raises pydantic ValidationError."""
        with pytest.raises(ValidationError):
            PipelineSpec(
                adapter=ObjectSpec(kind="adapter", name=""),
                method=ObjectSpec(kind="method_a", name="pca"),
            )

    def test_spec_empty_method_name_rejected(self) -> None:
        """Empty method name raises pydantic ValidationError."""
        with pytest.raises(ValidationError):
            PipelineSpec(
                adapter=ObjectSpec(kind="adapter", name="vae"),
                method=ObjectSpec(kind="method_a", name=""),
            )


class TestBuildPipelineFromConfig:
    """build_pipeline_from_config() behavior."""

    def test_build_with_vae_pca(self) -> None:
        """Build pipeline from PipelineSpec with VAE + PCA and run it."""
        spec = PipelineSpec(
            adapter=ObjectSpec(kind="adapter", name="vae", params={"input_dim": 8, "latent_dim": 3}),
            method=ObjectSpec(kind="method_a", name="pca", params={"n_components": 2}),
        )
        pipeline = build_pipeline_from_config(spec)
        assert isinstance(pipeline, AnalysisPipeline)
        assert pipeline.latent_space.dim == 3

        # Fit VAE, then run
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 1, size=(50, 8))
        pipeline.adapter.fit(data)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        result = pipeline.run(data)
        assert result.transformed.shape == (50, 2)

    def test_build_with_hidden_state_pca(self) -> None:
        """Build pipeline from PipelineSpec with HiddenStateAdapter + PCA."""
        spec = PipelineSpec(
            adapter=ObjectSpec(
                kind="adapter",
                name="hidden_state",
                params={"input_dim": 10, "hidden_dim": 4, "random_state": 42},
            ),
            method=ObjectSpec(kind="method_a", name="pca", params={"n_components": 2}),
        )
        pipeline = build_pipeline_from_config(spec)
        assert isinstance(pipeline, AnalysisPipeline)
        assert pipeline.latent_space.dim == 4

        rng = np.random.default_rng(42)
        data = rng.normal(size=(30, 10))
        result = pipeline.run(data)
        assert result.transformed.shape == (30, 2)

    def test_build_with_dict_like_spec(self) -> None:
        """Build pipeline using dict-like PipelineSpec (auto-coercion)."""
        spec = PipelineSpec(
            adapter={"kind": "adapter", "name": "vae", "params": {"input_dim": 8, "latent_dim": 3}},  # pyright: ignore[reportArgumentType]
            method={"kind": "method_a", "name": "pca", "params": {"n_components": 2}},  # pyright: ignore[reportArgumentType]
        )
        pipeline = build_pipeline_from_config(spec)
        assert isinstance(pipeline, AnalysisPipeline)

        rng = np.random.default_rng(42)
        data = rng.uniform(0, 1, size=(50, 8))
        pipeline.adapter.fit(data)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        result = pipeline.run(data)
        assert result.transformed.shape == (50, 2)

    def test_unknown_adapter_raises_keyerror(self) -> None:
        """Unknown adapter name raises KeyError."""
        spec = PipelineSpec(
            adapter=ObjectSpec(kind="adapter", name="nonexistent_adapter"),
            method=ObjectSpec(kind="method_a", name="pca"),
        )
        with pytest.raises(KeyError):
            build_pipeline_from_config(spec)

    def test_kind_mismatch_raises_valueerror(self) -> None:
        """Wrong kind for method (method_a expecting a method_b entry) raises ValueError."""
        spec = PipelineSpec(
            adapter=ObjectSpec(kind="adapter", name="vae", params={"input_dim": 8, "latent_dim": 3}),
            method=ObjectSpec(kind="method_a", name="lerp"),
        )
        with pytest.raises(ValueError):
            build_pipeline_from_config(spec)
