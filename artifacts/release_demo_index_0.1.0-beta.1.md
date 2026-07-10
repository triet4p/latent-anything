# Release Demo Index: 0.1.0-beta.1

This index points to the release-facing demos and tracked artifacts for the `0.1.0-beta.1` release.

## Best Starting Points

- `scripts/end_to_end_showcase_demo.py` - End-to-end VAE plus PCA plus ActivationPatch plus Lerp story.
- `artifacts/showcase_demo_plot.png` - Tracked visual output from the showcase.
- `artifacts/showcase_demo_summary.txt` - Tracked text summary from the showcase.
- `artifacts/showcase_config_snapshot.txt` - Tracked config snapshot for reproducibility.

## Layer A Demos

- `scripts/end_to_end_pca_demo.py` - PCA dimensionality reduction.
- `scripts/end_to_end_umap_demo.py` - UMAP nonlinear dimensionality reduction.
- `scripts/end_to_end_sae_demo.py` - Sparse autoencoder projection.
- `scripts/end_to_end_hidden_state_demo.py` - Hidden-state adapter plus PCA/UMAP visualization.
- `artifacts/hidden_state_demo_plot.png` - Tracked hidden-state visualization.

## Layer B Demos

- `scripts/end_to_end_lerp_demo.py` - Geometry-aware interpolation.
- `scripts/end_to_end_steering_demo.py` - Contrastive steering vectors.
- `scripts/end_to_end_activation_patch_demo.py` - Adapter-mediated activation patching.
- `scripts/end_to_end_manipulation_demo.py` - ManipulationPipeline data-space and trajectory stories.

## Adapter And Geometry Demos

- `scripts/end_to_end_vae_demo.py` - Learned explicit latent adapter.
- `scripts/end_to_end_random_projection_demo.py` - Fixed random projection adapter.
- `scripts/end_to_end_spherical_demo.py` - Unit-norm spherical geometry.
- `scripts/end_to_end_gaussian_set_demo.py` - Structured Gaussian-set latent geometry.
- `scripts/end_to_end_gaussian_renderer_demo.py` - Deterministic Gaussian renderer adapter.
- `artifacts/gaussian_set_demo_plot.png` - Tracked Gaussian-set visualization.
- `artifacts/gaussian_renderer_demo.png` - Tracked Gaussian renderer visualization.

## Pipeline, Registry, Config, And Runtime Demos

- `scripts/end_to_end_registry_demo.py` - Built-in registry listing and lookup.
- `scripts/end_to_end_config_demo.py` - Registry-backed config instantiation.
- `scripts/end_to_end_pipeline_demo.py` - AnalysisPipeline composition.
- `scripts/end_to_end_cache_demo.py` - AnalysisPipeline encode-cache behavior.
- `artifacts/cache_demo_summary.txt` - Tracked cache demo summary.
- `scripts/end_to_end_batch_executor_demo.py` - Deterministic batch execution.
- `artifacts/batch_executor_demo_summary.txt` - Tracked batch executor summary.
- `scripts/end_to_end_async_runtime_demo.py` - Concurrent async runtime/profiling demo.
- `artifacts/async_runtime_demo_summary.txt` - Tracked async runtime summary.

## Known Demo Limits

- Probe/TCAV demos are not shipped in this beta.
- Interactive visualization is not shipped in this beta.
- Several primitive demos write artifacts when run locally, but only the artifacts listed above are tracked as release reference outputs.
