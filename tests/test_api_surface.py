"""Deliberate snapshot of the public beta naming surface."""

from __future__ import annotations

import latent_anything
from latent_anything import adapters, methods
from latent_anything.registry import KIND_ADAPTER, KIND_METHOD_A, KIND_METHOD_B


def test_top_level_public_beta_export_snapshot() -> None:
    assert latent_anything.__all__ == [
        "AnalysisPipeline",
        "BatchExecutor",
        "CacheKey",
        "CacheStats",
        "ConceptDataset",
        "ConceptDirectionResult",
        "ControlBaselines",
        "CrossSeedReport",
        "GLOBAL_REGISTRY",
        "InMemoryCache",
        "LatentSpace",
        "LatentValue",
        "LinearProbe",
        "LinearProbeConfig",
        "LinearProbeResult",
        "MLPProbe",
        "MLPProbeConfig",
        "MLPProbeResult",
        "ManipulationPipeline",
        "ManipulationPipelineSpec",
        "Method",
        "ObjectSpec",
        "PipelineResult",
        "PipelineSpec",
        "ProbeComparison",
        "TCAV",
        "TCAVConfig",
        "TCAVResult",
        "TCAVScore",
        "TransformerLogitTarget",
        "ProfileEvent",
        "Registry",
        "RegistryEntry",
        "RuntimeProfile",
        "RuntimeProfiler",
        "Trajectory",
        "build_from_config",
        "build_from_dict",
        "build_manipulation_pipeline_from_config",
        "build_pipeline_from_config",
        "compare_probes",
        "compute_tcav",
        "cross_seed_evaluation",
        "evaluate_layers",
        "intervention_agreement",
        "learn_linear_separator_direction",
        "learn_mean_diff_direction",
        "list_entries",
        "lookup_entry",
        "nonlinear_memorization_test",
        "register_entry",
    ]


def test_method_and_adapter_protocol_snapshot() -> None:
    assert methods.__all__ == ["ActivationPatch", "BMethod", "Method", "PCA", "SAE", "UMAP", "Lerp", "SteeringVector"]
    assert adapters.__all__ == [
        "ConvVAE",
        "DecodableAdapter",
        "FlatBatchDecodableAdapter",
        "GaussianRendererAdapter",
        "HiddenStateAdapter",
        "ModelAdapter",
        "RandomProjection",
        "VAE",
    ]


def test_registry_kind_snapshot() -> None:
    assert (KIND_ADAPTER, KIND_METHOD_A, KIND_METHOD_B) == ("adapter", "analysis", "intervention")
