# Release Surface Inventory: 0.1.0-beta.1

**Sprint:** 26
**Task:** 1
**Status:** Complete

## Public Package Exports

Top-level exports from `latent_anything.__all__`:

- `AnalysisPipeline`
- `BatchExecutor`
- `CacheKey`
- `CacheStats`
- `GLOBAL_REGISTRY`
- `InMemoryCache`
- `LatentSpace`
- `ManipulationPipeline`
- `ManipulationPipelineSpec`
- `Method`
- `ObjectSpec`
- `PipelineResult`
- `PipelineSpec`
- `ProfileEvent`
- `Registry`
- `RegistryEntry`
- `RuntimeProfile`
- `RuntimeProfiler`
- `Trajectory`
- `build_from_config`
- `build_from_dict`
- `build_manipulation_pipeline_from_config`
- `build_pipeline_from_config`
- `list_entries`
- `lookup_entry`
- `register_entry`

Package version before release cut: `0.1.0`.

Final package metadata version after the beta cut: `0.1.0b1` (PEP 440-compatible form of the `v0.1.0-beta.1` release tag).

## Built-In Registry Entries

| Kind | Name | Protocol metadata |
| --- | --- | --- |
| `adapter` | `gaussian_renderer` | `ModelAdapter, DecodableAdapter` |
| `adapter` | `hidden_state` | `ModelAdapter` |
| `adapter` | `random_projection` | `ModelAdapter, DecodableAdapter, FlatBatchDecodableAdapter` |
| `adapter` | `vae` | `ModelAdapter, DecodableAdapter, FlatBatchDecodableAdapter` |
| `method_a` | `pca` | `Method` |
| `method_a` | `sae` | `Method` |
| `method_a` | `umap` | `Method` |
| `method_b` | `activation_patch` | `BMethod` |
| `method_b` | `lerp` | `BMethod` |
| `method_b` | `steering` | `BMethod` |

## Pipelines And Runtime Helpers

- `AnalysisPipeline`: adapter encode plus Layer A fit/transform, with optional encode cache and profiling.
- `ManipulationPipeline`: Layer B manipulation for data-space and trajectory stories, with async wrappers and profiling.
- `BatchExecutor`: deterministic first-axis batching for arrays, adapters, and Layer A transforms.
- `InMemoryCache`: memory-only numpy array cache with stable cache keys and stats.
- `RuntimeProfiler`, `RuntimeProfile`, `ProfileEvent`: stage timing hooks for cache, encode, method, and decode stages.

## Demo And Support Scripts

Release-facing scripts in `scripts/`:

- `end_to_end_activation_patch_demo.py`
- `end_to_end_async_runtime_demo.py`
- `end_to_end_batch_executor_demo.py`
- `end_to_end_cache_demo.py`
- `end_to_end_config_demo.py`
- `end_to_end_gaussian_renderer_demo.py`
- `end_to_end_gaussian_set_demo.py`
- `end_to_end_hidden_state_demo.py`
- `end_to_end_lerp_demo.py`
- `end_to_end_manipulation_demo.py`
- `end_to_end_pca_demo.py`
- `end_to_end_pipeline_demo.py`
- `end_to_end_random_projection_demo.py`
- `end_to_end_registry_demo.py`
- `end_to_end_sae_demo.py`
- `end_to_end_showcase_demo.py`
- `end_to_end_spherical_demo.py`
- `end_to_end_steering_demo.py`
- `end_to_end_umap_demo.py`
- `end_to_end_vae_demo.py`
- `showcase_config.py`

## Existing Demo Artifacts

Tracked generated artifacts available before Sprint 26:

- `artifacts/showcase_demo_plot.png`
- `artifacts/showcase_demo_summary.txt`
- `artifacts/showcase_config_snapshot.txt`
- `artifacts/hidden_state_demo_plot.png`
- `artifacts/gaussian_set_demo_plot.png`
- `artifacts/gaussian_renderer_demo.png`
- `artifacts/cache_demo_summary.txt`
- `artifacts/batch_executor_demo_summary.txt`
- `artifacts/async_runtime_demo_summary.txt`

## README And Changelog State

- `README.md` has a minimal install path and version smoke snippet, but it does not yet describe the beta scope or link release-ready demos.
- `CHANGELOG.md` still has `[Unreleased]` only and must be cut to `[0.1.0-beta.1] - 2026-07-10` only after the release gate is clean.

## Release Scope Conclusion

The current surface is credible for a core-framework beta covering latent-space primitives, concrete adapters, Layer A/B methods, registry/config, pipeline composition, and first runtime helpers. It is not a full implementation of every theory layer and must not claim probing/TCAV, planning, rollout, discrete latent adapters, streaming runtime, or interactive visualization as shipped features.
