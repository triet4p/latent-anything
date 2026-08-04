# Changelog

## [Unreleased]

### Added

- Added a pinned ACT LeRobot adapter that captures first-action decoder queries through official preprocessing/action-selection/postprocessing, preserves action-queue semantics, and provides observational projection, probing, control, and trajectory evidence with an opt-in public checkpoint smoke. (#sprint-58)
- Added a typed LeRobot v3 dataset bridge with schema/normalization/task descriptors, lazy episode readers, bounded streaming samples, alignment provenance, explicit captured-latent conversion, an offline fixture, and a pinned public metadata inspection artifact. (#sprint-57)
- Added the optional `latent_anything[lerobot]` boundary with a pinned LeRobot 0.6.x compatibility window, lazy raw upstream seams, bridge-owned context/evaluation types, CPU smoke tests, and a dedicated compatibility CI lane. (#sprint-56)
- Added constrained 3D Gaussian manipulation (localized SE(3) rigid edits, bounded opacity/color changes, removal, and opacity-weighted merge) with held-out multi-view geometric/rendering metrics and a deterministic naive-arithmetic failure control. (#sprint-55)
- Added a real 3D Gaussian splat renderer facade backed by optional `gsplat>=1.4,<2.0`, with typed camera transforms, 3D Gaussian latent metadata, deterministic CPU fixtures, strict parameter validation, and an opt-in GPU/public-scene lane. (#sprint-54)

- Added geometry-aware trajectory smoothing and robust latent-velocity change-point segmentation with immutable source metadata, confidence/provenance results, tolerance-aware boundary metrics, and typed visualization overlays. (#sprint-53)

- Added geometry-aware dynamic time warping for unequal-length latent trajectories, with typed alignment results and interactive renderer overlays.

- Added matrix-backed SO(3)/SE(3) pose geometry with group operations, valid interpolation, frame/unit metadata, trajectory slicing, serialization, and LeRobot-compatible state metadata.

- Added density-penalized geodesic path interpolation (`DensityGeodesic`, registered under `intervention` as `density_geodesic`): a non-Euclidean path method that treats the latent space as a Riemannian manifold whose metric is the inverse of a learned density, so the geodesic bends toward high-density on-manifold regions instead of cutting across low-density gaps. Config-driven via `GeodesicConfig`/`build_from_config`. (#sprint-50)
- Added `GeodesicPath` result with the full optimized path, density-penalized and Euclidean lengths, per-point log-density diagnostics, optional decoded images with a reconstruction diagnostic, and a `PathOptimizationStatus` (converged, iterations, initial/final energy, message). (#sprint-50)
- Added a deterministic, bounded path optimizer (lerp initialization, `max_iter`/`n_points` bounds, backtracking line search, gradient-norm/relative-energy convergence) in `geometry.py`, with `density_exponent = 0` provably recovering the lerp path. (#sprint-50)
- Added per-point `log_density` and analytic `log_density_gradient` oracles (and a `state_digest` for stable cache keys) to `GaussianMixtureDensity`, giving the geodesic its pullback-type density oracle. (#sprint-50)
- Added cache (`InMemoryCache`) and profiling (`RuntimeProfiler`) integration to `DensityGeodesic.optimize`, since path optimization is expensive; identical endpoint/config/oracle-state calls are served from cache. (#sprint-50)
- Added a reproducible geodesic benchmark (`scripts/geodesic_benchmark.py`) comparing lerp, slerp (where applicable), and the density geodesic on a curved ring manifold and real ConvVAE digits latents, with an `artifacts/geodesic_benchmark.json` artifact. (#sprint-50)
- Added guidance (`docs/GEODESIC_INTERPOLATION.md`) documenting when the density geodesic is justified and when simpler lerp/slerp/metric interpolation is preferable. (#sprint-50)
- Added orthonormal subspace projection and concept removal (`OrthonormalSubspace`, `SubspaceProjection`): an immutable fitted orthonormal basis bound to one coordinate-system identity with an explicit origin (PCA / probe / concept / explicit), orthogonal projection `P z`, residual `(I - P) z`, subspace coverage, and concept transfer, each returning new immutable `LatentValue` outputs with operation/provenance metadata. (#sprint-49)
- Added `LatentValue` latent arithmetic (`add`, `subtract`, `add_scaled`, `scale`, `+`, `-`) that is only allowed for values proven to share a coordinate system — same geometry, point shape, stored shape, and a matching declared `coordinate_identity`; arithmetic across unrelated coordinate systems or with an undeclared identity raises `ValueError` instead of returning a plausible-looking array. (#sprint-49)
- Added a canonical coordinate-system identity (`LatentValue.identity`, `coordinate_identity`) built from `source_representation_identity`, `source_model`, and revision metadata, and an explicit compatibility check (`assert_arithmetic_compatible`). (#sprint-49)
- Registered `subspace_projection` under the canonical `intervention` registry kind with config-driven construction (`SubspaceProjectionConfig`, `build_from_config`). (#sprint-49)
- Added 64 analytic/property tests for idempotence, orthogonality, reconstruction, coverage, subspace serialization, basis families, non-interchangeability, and invalid cross-space arithmetic rejection. (#sprint-49)
- Added a reproducible concept-removal benchmark (`scripts/concept_removal_benchmark.py`) measuring target suppression, off-target preservation, decode degradation, and a random-subspace control on real ConvVAE digits latents, with an `artifacts/concept_removal_benchmark.json` artifact. (#sprint-49)
- Added a projection-basis comparison benchmark (`scripts/projection_basis_comparison.py`) showing that PCA, probe-coefficient, and concept-direction bases are not interchangeable (pairwise principal-angle alignment, different removal effects, origin recorded per basis) with an `artifacts/projection_basis_comparison.json` artifact. (#sprint-49)
- Added a latent-arithmetic benchmark (`scripts/latent_arithmetic_benchmark.py`) demonstrating monotone in-system steering and cross-identity arithmetic rejection, with an `artifacts/latent_arithmetic_benchmark.json` artifact. (#sprint-49)

- Added an anisotropic Gaussian geometry (`LatentSpace(geometry="anisotropic")`) with a fitted, positive-definite covariance metric: covariance validation, diagonal-loading regularization, Mahalanobis distance, whitening/inverse transforms, and declared metric interpolation. (#sprint-48)
- Added the stateful covariance contract (`CovarianceState`, `CovarianceConfig`, `fit_covariance_state`) that binds a fitted metric to its source representation identity and provenance, with JSON and `.npz` serialization. (#sprint-48)
- Routed all anisotropic algorithms through the focused `geometry.py` module while keeping `LatentSpace` as the small public facade, matching the Sprint-30 geometry extraction. (#sprint-48)
- Documented the interpolation semantics decision: under a constant covariance the metric geodesic coincides with the affine coordinate lerp, but interpolation is routed through the declared metric and requires a fitted covariance instead of silently defaulting to Euclidean. (#sprint-48)
- Added 55 analytic/property tests for affine invariance, singular-covariance handling, distance symmetry, whitening round-trips, serialization, and the `LatentSpace` anisotropic dispatch. (#sprint-48)
- Added a reproducible Euclidean-vs-Mahalanobis benchmark (`scripts/anisotropy_benchmark.py`) on a controlled anisotropic dataset and real ConvVAE digits latents, with an `artifacts/anisotropy_benchmark.json` artifact. (#sprint-48)

- Added an optional interactive visualization package (`latent_anything.visualization`, install with `uv sync --extra viz`) that renders typed analysis results without embedding plotting logic into analysis methods. (#sprint-47)
- Added typed renderer inputs (`ProjectionView`, `TrajectoryView`, `MetricSummary`) and builders that convert probe, K-means, density, trajectory, and feature-atlas results into views (`projection_from_probe`, `projection_from_kmeans`, `projection_from_density`, `projection_from_trajectory`, `projection_from_atlas`). (#sprint-47)
- Added a Plotly-based 2D/3D projection explorer (`projection_explorer`) with category coloring, continuous color scaling, hover metadata, trajectory overlays, and box/lasso selection. (#sprint-47)
- Added a notebook widget path (`ProjectionExplorer`) that renders an interactive ipywidgets container with a metadata-inspection panel in Jupyter and degrades cleanly to self-contained HTML or PNG/SVG export outside a notebook. (#sprint-47)
- Added declared responsiveness targets (`DEFAULT_POINT_LIMIT_2D = 50_000`, `DEFAULT_POINT_LIMIT_3D = 20_000`) with deterministic, category-stratified downsampling that never thins trajectory overlays. (#sprint-47)
- Added schema/snapshot tests for renderer inputs and figures, import-isolation tests proving the base package never imports plotly/kaleido/ipywidgets, and a manual browser visual QA checklist (`docs/visual-qa-checklist.md`). (#sprint-47)
- Added an interactive real-model walkthrough (`scripts/interactive_viz_walkthrough.py`) that renders digits ConvVAE K-means, probe, density, trajectory, and SAE feature-atlas charts with quantitative metrics and a 60k-point responsiveness check. (#sprint-47)

- Added sparse-autoencoder feature evaluation (`SAEFeatureEvaluation`) covering held-out reconstruction MSE, L0/L1 activity, dead-feature detection, activation frequency, decoder/encoder norms, train/validation separation, and portable `.npz` checkpoint serialization of fitted SAE state. (#sprint-46)
- Added cross-seed feature stability (`cross_seed_sae_stability`) that matches features by decoder-direction cosine similarity across seeds instead of comparing arbitrary feature indices. (#sprint-46)
- Added feature example/counterexample ranking (`rank_feature_examples`) and a probe/concept/causal-steering cross-check (`cross_check_feature`) against a scalar transformer logit target. (#sprint-46)
- Added a portable, queryable feature-atlas JSON artifact (`build_feature_atlas`, `save_feature_atlas`, `load_feature_atlas`) independent of any visualization frontend. (#sprint-46)
- Added offline regression-threshold tests and a marked pinned-GPT-2 full-model evaluation test for the SAE feature pipeline. (#sprint-46)
- Added activation-space Integrated Gradients for a selected transformer residual-layer/token activation and scalar next-token logit, with bounded baselines, typed NumPy results, completeness/convergence diagnostics, sensitivity reporting, direct/config construction, and marked real-checkpoint evidence. (#sprint-45)
- Added representation-bound Gaussian-mixture density estimation with held-out calibration, calibrated OOD scores, responsibilities, provenance, AUROC/AUPRC diagnostics, Mahalanobis baseline comparison, and cross-seed uncertainty reports. (#sprint-44)

### Fixed

- Fixed anisotropic covariance states to defensively own immutable arrays and provenance, enforce invariants consistently across construction and loading, reject blank representation identities, and validate both endpoints before geometry operations.

- Normalized the SAE L1 sparsity penalty per element so it is comparable to the reconstruction loss; an unnormalized `sum(|latent|)` collapsed every feature to dead even at small `l1_coef`. (#sprint-46)
- Made marked model-download integration tests opt-in via `LATENT_ANYTHING_RUN_NETWORK=1`, so the default CI quality gate remains offline and reproducible.

### Added

- Added K-means clustering module (`KMeans`) for latent structure discovery with explicit geometry compatibility checks, typed cluster results (`KMeansResult`), silhouette diagnostics, and a nearest-versus-second-nearest distance-margin confidence proxy. (#sprint-43)
- Added bootstrap/seed stability analysis (`cluster_stability_analysis`) with Hungarian label alignment and adjusted Rand index for quantifying cluster robustness. (#sprint-43)
- Added external validation (`compare_with_labels`) using adjusted Rand index, mutual information, homogeneity, completeness, and V-measure against known ground-truth labels. (#sprint-43)
- Added geometry compatibility checks (`check_clustering_geometry`) that reject clustering on unsupported structured (`gaussian_set`) and discrete (`discrete_code`) latent spaces while allowing `euclidean` and `unit_norm` geometries. (#sprint-43)
- Added `KMeansConfig` / `KMeans` class supporting both direct use and config-driven construction via `ObjectSpec`/`build_from_config`. (#sprint-43)
- Registered `kmeans` under the `"analysis"` registry kind for config-driven construction. (#sprint-43)
- Added 50 tests covering config validation, result serialization, fit-predict lifecycle, input validation, standardization, silhouette/confidence diagnostics, geometry checks, stability analysis, external validation, degenerate/edge-case inputs, registry construction, and 2 marked real-integration tests for VAE and transformer representations. (#sprint-43)

- Added concept/reference dataset handling (`ConceptDataset`) with deterministic sampling, train/test split, per-class stratification, and full provenance (source, representation-space, model-version). (#sprint-42)
- Added concept-direction learning via both mean difference (`learn_mean_diff_direction`) and regularised linear separator (`learn_linear_separator_direction`), with direction stability (bootstrap cosine similarity) and held-out separability reporting. (#sprint-42)
- Added scalar model target specification (`TransformerLogitTarget`) and internal gradient computation that differentiates a specific token logit w.r.t. activations at a declared layer in decoder-only transformers. (#sprint-42)
- Added typed TCAV result types (`TCAVScore`, `TCAVResult`) with per-example directional sensitivities, aggregate TCAV score, uncertainty (multi-seed CI95), and full provenance. (#sprint-42)
- Added repeated random-concept baselines with binomial significance testing and Bonferroni correction for the declared family of comparisons. (#sprint-42)
- Added intervention cross-check (`intervention_agreement`) that compares observational TCAV sensitivity with bounded matched-norm interventions along the learned direction. (#sprint-42)
- Added `TCAVConfig` / `TCAV` class supporting both direct use and config-driven construction via `ObjectSpec`/`build_from_config`. (#sprint-42)
- Registered `tcav` under the `"analysis"` registry kind for config-driven construction. (#sprint-42)
- Added 58 tests covering ConceptDataset validation, direction learning, gradient computation (synthetic model), full `compute_tcav` pipeline, intervention cross-check, registry construction, edge cases, and 2 marked real-integration tests for VAE and transformer representations. (#sprint-42)

- Added a revision-pinned decoder-only transformer integration (`TransformerLMIntegration`) with direct logit lens, typed input/hidden-state/lens-result values with NumPy-facing public payloads and full model/tokenizer provenance. (#sprint-39)
- Added native `output_hidden_states=True` as the canonical observation path for transformer hidden states, with verified embedding/residual/final hidden-state indexing and shapes. (#sprint-39)
- Added a direct logit lens implementation with explicit final-normalization (LayerNorm) and output-head (LM head) assumptions; learned/tuned translators deferred. (#sprint-39)
- Added validation of native hidden states and final logits against direct backend execution, including padded-token masking and final-layer parity checks. (#sprint-39)
- Added hook-based activation intervention support via `ActivationCaptureSession` for one bounded activation intervention at a specified layer, with hook cleanup verification. (#sprint-39)
- Added token rank/probability trajectory measurement across layers, with stability tracking under predeclared prompt perturbations. (#sprint-39)
- Added comprehensive test suite: 38 offline tests with fake backend + 11 marked real-checkpoint tests, plus a reproducible artifact demo script. (#sprint-39)
- Added a `transformers` optional install extra with pinned GPT-2 (`gpt2` model at revision `e7da7f2`, 124M parameters) for the transformer integration. (#sprint-39)

- Added a label-aware `LinearProbe` class with leakage-guarded train/val/test splitting, training-only feature standardization, regularization, and class-balance controls. (#sprint-40)
- Added `LinearProbeConfig` (pydantic v2) and `LinearProbeResult` (frozen dataclass) with labels, predictions, probabilities, coefficients, split metadata, and provenance. (#sprint-40)
- Added `compute_controls()` for majority-class, shuffled-label, and raw-input baselines evaluated on the same train/test split as the probe. (#sprint-40)
- Added `cross_seed_evaluation()` with 95 % confidence intervals and `evaluate_layers()` for probing across multiple representation layers. (#sprint-40)
- Added `control_baselines`, `cross_seed_report`, and `evaluate_layers` public exports. (#sprint-40)
- Registered `LinearProbe` under the `"analysis"` registry kind for config-driven construction. (#sprint-40)
- Reconciled the Sprint 36 centroid-based `probe_accuracy` helper: renamed to `_centroid_probe_accuracy` (internal fast path for `evaluate_explanation`); public `probe_accuracy` now delegates to `LinearProbe`. (#sprint-40)
- Added 42 offline tests covering unit, leakage, degenerate-class, config-construction, and 3 marked real-integration tests for the VAE and transformer integrations. (#sprint-40)

- Added a bounded nonlinear MLP probe (`MLPProbe`) with configurable hidden-layer architecture, deterministic PyTorch initialisation, early stopping on validation accuracy, and NumPy-facing typed results (`MLPProbeConfig`, `MLPProbeResult`). (#sprint-41)
- Added architecture reporting (layer sizes, parameter count, activation, n_hidden_layers) in probe results. (#sprint-41)
- Added `nonlinear_memorization_test()` with predeclared selectivity threshold (default 2× chance) to detect label memorization. (#sprint-41)
- Added `compare_probes()` that classifies representation access as linear-only, nonlinear-only, both, unsupported, or memorization-prone under explicit accuracy and gap thresholds. (#sprint-41)
- Reused Sprint 40 `_stratified_split` and training-only standardization, keeping nonlinear-model-specific state (architecture, n_params, early stopping) out of the linear result type. (#sprint-41)
- Registered `MLPProbe` under the `"analysis"` registry kind for config-driven construction. (#sprint-41)
- Added 33 offline tests covering capacity, overfit, degenerate-label, determinism, config round-trip, memorization testing, comparison, and 2 marked real-integration tests. (#sprint-41)- Evidence-ledger validation now inventories all theory capabilities, verifies local evidence links in CI, and reports the D2/D3 stable-coverage denominator without downloading optional models. (#sprint-27)
- `LatentValue` carries immutable flat batches and structured latent states with explicit `LatentSpace` association, safe NumPy conversion, and beta `Trajectory` compatibility. (#sprint-29)
- `LatentSpace` now supports categorical `discrete_code` geometry with codebook validation, normalized Hamming distance, and an explicit no-continuous-interpolation policy. (#sprint-30)
- Registry and config construction now use canonical `analysis` and `intervention` kinds; legacy beta kinds remain aliases with migration diagnostics until `0.9.0`. (#sprint-31)
- Added internal PyTorch activation capture sessions with safe hook lifecycle, NumPy outputs, provenance metadata, and intervention callbacks. (#sprint-32)
- Added optional `transformers`, `3d`, and `lerobot` installation extras alongside the Diffusers integration. (#sprint-33)
- Added a CPU-reproducible convolutional VAE integration evaluated on sklearn digits image data. (#sprint-34)
- Added a lazy, revision-pinned Diffusers `AutoencoderKL` integration with explicit network acquisition gating and reproducible interpolation diagnostics. (#sprint-35)
- Added a control-aware VAE explanation benchmark with held-out probes, negative controls, seed intervals, and decoded steering effects. (#sprint-36)
- Added a revision-pinned conditional text-to-image diffusion integration (`DiffusersConditionalPipeline`) that records scheduler latent states via native `callback_on_step_end` and selected denoiser activations via `ActivationCaptureSession`, with typed NumPy-faced request/result objects and separate `LatentSpace` descriptors for VAE bottlenecks, scheduler states, and denoiser activations. (#sprint-37)
- Added a combined `diffusers-full` optional install extra that includes both `diffusers` and `transformers` dependencies for the conditional diffusion pipeline. (#sprint-37)
- Added `SchedulerIntervention` data type and intervention support to `DiffusersConditionalPipeline.generate()` — additive edits on scheduler latent states during denoising via `callback_on_step_end`, with `random_direction()` and `matched_norm_direction()` helpers. (#sprint-38)
- Added deterministic offline smoke tests for `SchedulerIntervention` validation, direction helpers, intervention passthrough in `generate()`, and a gated real-checkpoint benchmark that verifies the intervention changes intermediate latents. (#sprint-38)
- Added a comprehensive experiment script (`diffusers_conditional_intervention_experiment.py`) that compares no-edit/prompt-only/random-direction/matched-norm controls across multiple seeds with target change, SSIM, latent norm drift, and trajectory-cosine metrics, plus a timestep-by-strength sweep. (#sprint-38)

### Changed
- `09_gaussian_rasterization.ipynb`: improve Exp 4 and Exp 5 visualizations — denser overlap scenes, cumulative contribution breakdown, rendered image quality comparison, percent-savings heatmaps.

### Fixed

- Evidence-ledger schema-v2 validation now enforces typed `role`/`path` records and the required evidence roles for every D1, D2, and D3 claim.
- `LatentValue.metadata` now returns defensive immutable snapshots, including NumPy metadata that callers attempt to make writable again.
- `end_to_end_showcase_demo.py`: add missing `pydantic` to inline script dependencies so `uv run --script` resolves it correctly; add `sys.stdout.reconfigure(encoding='utf-8')` for Windows Unicode support.

## [0.1.0-beta.1] - 2026-07-10

### Release Summary

`0.1.0-beta.1` is the first core-framework beta for Latent Anything. It ships concrete latent-space primitives, first adapter modes, Layer A inspection methods, Layer B manipulation methods, registry/config construction, analysis/manipulation pipelines, and first Layer C runtime helpers.

This is a pre-1.0 beta, not an API-stability promise and not a claim that every theory layer is implemented.

### Demo And Artifact Links

- [Release demo index](artifacts/release_demo_index_0.1.0-beta.1.md)
- [Showcase summary](artifacts/showcase_demo_summary.txt)
- [Showcase plot](artifacts/showcase_demo_plot.png)
- [Gaussian renderer demo](artifacts/gaussian_renderer_demo.png)
- [Gaussian-set demo plot](artifacts/gaussian_set_demo_plot.png)
- [Cache demo summary](artifacts/cache_demo_summary.txt)
- [Batch executor demo summary](artifacts/batch_executor_demo_summary.txt)
- [Async runtime demo summary](artifacts/async_runtime_demo_summary.txt)

### Install And Test Notes

- Package version metadata is `0.1.0b1` for PEP 440 compatibility.
- Recommended release tag: `v0.1.0-beta.1`.
- Local install path: `uv sync --locked`.
- Release gate: `uv sync --locked`, `uv run ruff check src tests scripts`, `uv run ruff format --check src tests scripts`, `uv run pyright`, and `uv run pytest`.

### Known Limitations

- Package is not published to PyPI in this sprint.
- External plugin discovery via Python entry points is not shipped yet.
- Streaming runtime is not shipped yet.
- Interactive Plotly/notebook widget visualization is not shipped yet.
- Probing/TCAV, clustering, feature attribution, trajectory similarity, rollout, planning, and discrete latent adapters remain future work.
- Large modules such as `pipeline.py` and `latent_space.py` are beta-acceptable but recorded as post-beta refactor candidates once new concrete execution or geometry stories justify extraction.

### Theory Coverage Caveats

- The release is theory-informed, not theory-complete.
- Shipped code covers core primitives, selected representation learning adapters, selected geometry-aware operations, first manipulation methods, concrete pipelines, and runtime helpers.
- Theory layers for planning, latent prediction, discrete latents, full probing/TCAV, and large-scale world-model integrations are documented but not implemented in this beta.

### Added

- **Async runtime wrappers** — Added `AnalysisPipeline.run_async()`, `ManipulationPipeline.run_data_async()` / `run_trajectory_async()`, and `BatchExecutor.encode_async()` / `decode_async()` / `transform_async()` as thread-backed `asyncio` wrappers over the existing concrete runtime paths. Sync APIs remain available for current scripts and notebooks. (#sprint-24)
- **Runtime profiling hooks** — Added `RuntimeProfiler`, `RuntimeProfile`, and `ProfileEvent` for stage-level timing across `cache`, `encode`, `method`, and `decode` operations. `AnalysisPipeline`, `ManipulationPipeline`, and `BatchExecutor` accept optional profiling hooks without changing their return types. (#sprint-24)
- **Async runtime demo script** — `scripts/end_to_end_async_runtime_demo.py` runs one `AnalysisPipeline` job and one `ManipulationPipeline` job concurrently and writes `artifacts/async_runtime_demo_summary.txt`. Local snapshot: concurrent wall time 98.607 ms with per-stage breakdowns for both jobs. (#sprint-24)

- **`InMemoryCache` — cache backend #1** — `src/latent_anything/runtime/cache.py` with memory-only `get`, `set`, `clear`, and stats. Stores defensive copies of numpy arrays so callers cannot mutate cached values by accident. (#sprint-23)
- **Stable `CacheKey` structure** — Records namespace, operation, component name, component config hash, behavior-affecting component state hash, input data hash, and framework version when available. Data/config/state hashes use SHA-256 without introducing a pickle or disk format. (#sprint-23, #sprint-25)
- **AnalysisPipeline cache integration** — `AnalysisPipeline(adapter, method, cache=InMemoryCache())` caches adapter `encode` latents for repeated identical runs while always fitting stateful Layer A methods on the current pipeline instance. (#sprint-23, #sprint-25)
- **Cache demo script** — `scripts/end_to_end_cache_demo.py` demonstrates adapter-encode reuse through `AnalysisPipeline` and writes `artifacts/cache_demo_summary.txt`. (#sprint-23, #sprint-25)
- **New public exports** — `InMemoryCache`, `CacheKey`, and `CacheStats` exported from top-level `latent_anything` package and `latent_anything.runtime`. (#sprint-23)
- **`BatchExecutor` — Runtime #1** — `src/latent_anything/runtime/batch_executor.py` with deterministic first-axis numpy chunking, eager/synchronous execution, and output concatenation in original order. Supports generic `map_array()` plus adapter `encode()`, adapter `decode()`, and Layer A method `transform()` helpers. (#sprint-22)
- **Batch executor tests** — 23 tests covering construction, exact divisibility, remainder batches, batch size 1, batch size larger than data, invalid batch sizes, adapter `encode`/`decode`, PCA `transform`, output order/shape preservation, dtype preservation, and operation-output validation. (#sprint-22)
- **Batch executor demo script** — `scripts/end_to_end_batch_executor_demo.py` prints direct-vs-batched timing snapshots on synthetic data and writes `artifacts/batch_executor_demo_summary.txt`. Local snapshot: `RandomProjection.encode` direct 1.934 ms vs batched 3.302 ms; `PCA.transform` direct 1.507 ms vs batched 4.475 ms. (#sprint-22)
- **New public export** — `BatchExecutor` exported from top-level `latent_anything` package and `latent_anything.runtime`. (#sprint-22)
- **`ManipulationPipeline` — Pipeline #2** — `src/latent_anything/pipeline.py` with a concrete manipulation pipeline for Layer B methods. Supports two stories: (1) adapter-mediated data-space output via `run_data()` — encode → BMethod → decode → metric-ready `np.ndarray` (used with `ActivationPatch`); (2) latent-only trajectory output via `run_trajectory()` — BMethod `apply_trajectory` returning a new `Trajectory` (used with `SteeringVector`, `Lerp`). Deliberately avoids a generic `run()` because `__call__` signatures differ across B-Methods. (#sprint-21)
- **`_PipelineBase` — shared pipeline sketch** — Minimal base recording the common surface (adapter + method + optional `latent_space`) between `AnalysisPipeline` and `ManipulationPipeline`. Sketch only — freeze waits for Pipeline #3 (Rule of Three). (#sprint-21)
- **`ManipulationPipelineSpec` + `build_manipulation_pipeline_from_config`** — Config-backed construction path using Sprint 18 config machinery. Supports optional adapter spec for data-space stories. (#sprint-21)
- **28 manipulation pipeline tests** — covering construction (6), data-space story (3), trajectory story (4), fit delegation (3), convenience methods (3), spec model invariants (4), and config-backed construction errors (5). (#sprint-21)
- **Manipulation pipeline demo script** — `scripts/end_to_end_manipulation_demo.py` reproduces the Sprint 13 showcase path through Pipeline #2: ActivationPatch (data-space) and SteeringVector (trajectory) stories with matplotlib visualisation of before/after metrics and trajectory spread. (#sprint-21)
- **New public exports** — `ManipulationPipeline`, `ManipulationPipelineSpec`, `build_manipulation_pipeline_from_config` exported from top-level `latent_anything` package. (#sprint-21)

### Changed

- **`ActivationPatch` and `ManipulationPipeline` staged path** — `ActivationPatch` now exposes `apply_latent()`, allowing `ManipulationPipeline.run_data()` and `run_data_async()` to profile `encode → method → decode` as separate runtime stages instead of one opaque method call. (#sprint-24)
- **New public runtime exports** — `RuntimeProfiler`, `RuntimeProfile`, and `ProfileEvent` are exported from both `latent_anything.runtime` and the top-level `latent_anything` package. (#sprint-24)
- Test suite: 594 total tests (589 existing + 4 runtime async/profiling + 1 demo smoke). (#sprint-24)

- Test suite: 589 total tests (575 existing + 13 cache + 1 demo smoke). (#sprint-23)
- Test suite: 575 total tests (551 existing + 23 batch executor + 1 demo smoke). (#sprint-22)
- **`AnalysisPipeline`** now inherits from `_PipelineBase` — no behavioral change, purely structural. (#sprint-21)
- **`__init__.py`** — Added manipulation pipeline exports to both the import block and `__all__`. (#sprint-21)
- Test suite: 551 total tests (523 existing + 28 manipulation pipeline). (#sprint-21)

- **Separated built-in registry module** — `src/latent_anything/_plugin_builtins.py` as the single stable import location where all built-in adapters and methods are registered into `GLOBAL_REGISTRY`. This decouples `registry.py` from concrete class dependencies, making it pure infrastructure. (#sprint-19)
- **Internal plugin extraction contract** — Documented in `_plugin_builtins.py` docstring: registration-only (no re-exports), deterministic order, no circular imports, one-to-one with built-in classes, and entry-point readiness for future external plugins. (#sprint-19)
- **Parity test suite** — 22 new tests in `test_parity.py` covering registry constructor vs direct import constructor for all 10 built-in classes (4 adapters + 3 method_a + 3 method_b), plus factory identity checks proving `registry.lookup("name").factory` is the class itself. (#sprint-19)
- **Demo smoke test suite** — 15 new tests in `test_demo_smoke.py` verifying that every `scripts/end_to_end_*.py` demo's core imports and helpers still work after the registry refactoring. (#sprint-19)

- **`registry.py`** — Refactored to infrastructure-only: removed all adapter/method class imports and the registration block at module bottom. `Registry` class, kind constants, convenience helpers, and `GLOBAL_REGISTRY` singleton remain unchanged. (#sprint-19)
- **`__init__.py`** — Added `from latent_anything import _plugin_builtins` to trigger built-in registration on package import, before any registry-dependent modules (like `config.py`). (#sprint-19)
- Test suite: 502 total tests (465 existing + 22 parity + 15 demo smoke). (#sprint-19)

- **Registry-backed config instantiation** — `src/latent_anything/config.py` with pydantic v2 `ObjectSpec` model (`kind`, `name`, `params`), `build_from_config(spec)` that resolves registry entries and instantiates them, and `build_from_dict(data)` convenience wrapper. Config instantiation instance #1 — registry-local, deliberately narrow, no Pipeline/workflow language. (#sprint-18)
- **Nested spec resolution** — `build_from_config` recursively resolves nested `ObjectSpec` values inside `params`, enabling Layer B methods with adapter dependencies to be built from config (e.g. `ActivationPatch(adapter=VAE(...))`). Nested specs work as both `ObjectSpec` instances and plain dicts. (#sprint-18)
- **Clear validation errors** — `build_from_config` raises `KeyError` with sorted available names for unknown entries, `ValueError` with kind mismatch details, and `TypeError` with the failing params for instantiation failures. (#sprint-18)
- **Config-driven demo script** — `scripts/end_to_end_config_demo.py` builds the showcase object stack (VAE, PCA, Lerp, ActivationPatch with nested VAE) entirely from pydantic config specs without manual constructor calls. No Pipeline abstraction introduced. (#sprint-18)
- **New public exports** — `ObjectSpec`, `build_from_config`, `build_from_dict` exported from the top-level `latent_anything` package. (#sprint-18)
- **pydantic v2 dependency** — `pydantic>=2.0,<3.0` added to project dependencies for config model validation. (#sprint-18)
- Test suite: 36 config tests covering ObjectSpec construction (5), adapter building (4), Layer A method building (4), Layer B method building (4), error cases (7), custom registry (2), build_from_dict (3), and all six required classes (6). 465 total tests. (#sprint-18)

- **In-process registry** — `src/latent_anything/registry.py` with `Registry` class (`OrderedDict`-backed), frozen `RegistryEntry` dataclass, kind constants (`KIND_ADAPTER`, `KIND_METHOD_A`, `KIND_METHOD_B`), module-level convenience helpers (`register`, `lookup`, `list_entries`), and a `GLOBAL_REGISTRY` singleton. Designed as registry instance #1 with no Python entry points yet. (#sprint-17)
- **Built-in class registration** — All 10 built-in classes registered in `GLOBAL_REGISTRY`: VAE, RandomProjection, HiddenStateAdapter, GaussianRendererAdapter (adapters); PCA, UMAP, SAE (Layer A methods); Lerp, SteeringVector, ActivationPatch (Layer B methods). Registration uses class references as factories with metadata (protocol, description, source). (#sprint-17)
- **Deterministic ordering** — `Registry.list()` and `Registry.list(kind=...)` return entries in insertion order, guaranteed by `OrderedDict` backing. (#sprint-17)
- **Duplicate and missing-name guards** — `Registry.register()` raises `ValueError` on duplicate names (with descriptive message including registry name). `Registry.lookup()` raises `KeyError` on missing names. (#sprint-17)
- **Standalone registry demo** — `scripts/end_to_end_registry_demo.py` lists all registered entries grouped by kind, demonstrates lookup, duplicate-guard, and missing-name guard. (#sprint-17)
- Test suite: 48 registry tests covering construction (5), registration (6), lookup (5), listing (6), factory retrieval (2), RegistryEntry invariants (3), convenience helpers (6), GLOBAL_REGISTRY builtins (8), error cases (5), and no-breakage verification (2). (#sprint-17)

- **`GaussianRendererAdapter`** — ModelAdapter #4 (mode iii: deterministic renderer), closing the last evidence gap for the 3-mode ModelAdapter ADR. A concrete adapter that treats a fixed-size set of 2D Gaussian parameters as a latent representation and renders them into an RGB image via a deterministic numpy-only Gaussian splat renderer. `latent_space` returns a `gaussian_set` `LatentSpace` with position(2) + scale(2) + opacity(1) + color(3) = 8 columns. Conforms to both `ModelAdapter` and `DecodableAdapter` Protocols. (#sprint-16)
- **Deterministic 2D Gaussian splat decode** — `decode(latent) -> (H, W, 3)` renders Gaussian parameters with additive alpha composition: each Gaussian contributes `weight = opacity * exp(-((x-px)²/(2*sx²) + (y-py)²/(2*sy²)))` and colour is accumulated across all Gaussians with no normalisation (avoiding the single-Gaussian flat-colour artifact). Pure numpy, no CUDA or ``gsplat`` dependency. (#sprint-16)
- **Heuristic grid-based encode** — `encode(image) -> (n_gaussians, 8)` provides a simple approximate inverse: places Gaussians on a regular grid, samples colour from pixel centres, with small random jitter for visual variety. Documented as latent-source-first — the adapter is designed for create-then-decode workflows, not encode-then-decode roundtrips. (#sprint-16)
- End-to-end demo script `scripts/end_to_end_gaussian_renderer_demo.py` — synthetic Gaussian-set latent → decode to RGB → interpolation sequence via `LatentSpace` `gaussian_set` geometry → heuristic encode → roundtrip → 2×3 matplotlib figure with state A and B, midpoint, 3-step interpolation, and info panel. (#sprint-16)
- Test suite: 52 GaussianRendererAdapter tests covering construction (5), latent_space property (9), decode shape/range (4), determinism (2), opacity/colour constraints (3), decode input validation (9), encode output shape/constraints (9), roundtrip (3), no-mutation (2), and Protocol conformance (6). 381 total tests. (#sprint-16)

- **`gaussian_set` geometry** — Geometry case #3 for `LatentSpace`, the first structured set-like latent shape. Fixed-size Gaussian parameter sets with position, scale, opacity, and color channels. Enables `LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=100)` with `shape=(100, 10)` while preserving flat-geometry ergonomics for existing callers. (#sprint-15)
- **Gaussian parameter schema** — `gaussian_set_param_layout` metadata key documenting column layout (position, scale, opacity, color start indices and lengths). (#sprint-15)
- **`LatentSpace` Gaussian-set validation** — `validate_point()` checks shape `(n_gaussians, param_dim)` plus numeric constraints: scale components > 0 (log-normal storage), opacity in [0,1], color channels in [0,1]. (#sprint-15)
- **Permutation-aware distance for Gaussian sets** — `distance()` sorts both sets by position (lexicographic) then computes Frobenius norm. Deterministic, O(n log n), avoids optimal-assignment complexity. Invariant to Gaussian reordering. (#sprint-15)
- **Interpolation for Gaussian-set states** — `interpolate()` sorts by position for correspondence, then interpolates position (lerp), scale (log-space lerp for positivity), opacity (lerp + clamp [0,1]), and color (lerp + clamp [0,1]). (#sprint-15)
- **`n_gaussians` and `param_dim` properties** — `LatentSpace.n_gaussians` returns `int | None` (set count for `gaussian_set`, `None` for flat geometries). `LatentSpace.param_dim` returns total parameter dimensionality per Gaussian. (#sprint-15)
- End-to-end demo script `scripts/end_to_end_gaussian_set_demo.py` — synthetic Gaussian-set points → validates construction/param-layout → permutation-aware distance (invariant under shuffle) → interpolation at 5 steps → 2×3 matplotlib figure showing position, scale, opacity, color, distance profile, and info panel. (#sprint-15)
- Test suite: 34 new `LatentSpace` tests covering Gaussian-set construction (8), validation (9), distance (4), interpolation (8), property-based invariants (2), normalize (1), and backward compatibility with flat geometries (2). 82 total LatentSpace tests. (#sprint-15)

- **Adapter Protocols frozen** — `adapters/protocols.py` with structural `typing.Protocol` split: base `ModelAdapter` (shape-generic: `encode` + `latent_space`), `DecodableAdapter` (shape-generic `+decode`), and `FlatBatchDecodableAdapter` (narrow flat batch-matrix `encode`/`decode` surface for methods such as `ActivationPatch`). The split reflects the core evidence from three adapters: `decode` is NOT universal, and decodable does not always mean flat-batch. Frozen at ModelAdapter #3 (Sprint 14), then refined after Sprint 16's structured deterministic renderer. (#sprint-14, #sprint-16)
- **`HiddenStateAdapter`** — ModelAdapter #3 (mode ii: no-explicit-latent), demonstrating the pattern where hidden-state activations *are* the latent representation. Fixed random 2-layer ReLU MLP with He initialisation. `encode(data)` returns `(n_samples, hidden_dim)` hidden activations. No `decode` method — there is no decoder. `latent_space` returns Euclidean `LatentSpace` with `exposure_mode="hidden_state"` metadata. Pure numpy — no torch or heavy transformer dependency. (#sprint-14)
- **`ActivationPatch` runtime guard** — now requires `FlatBatchDecodableAdapter` with `isinstance` check at construction. `HiddenStateAdapter` (no `decode`) and `GaussianRendererAdapter` (structured decodable, not flat-batch) are cleanly rejected with `TypeError`. (#sprint-14, #sprint-16)
- End-to-end demo script `scripts/end_to_end_hidden_state_demo.py` — synthetic 8D clusters → HiddenStateAdapter encode → PCA/UMAP 2D visualization → no decode story. (#sprint-14)
- Test suite: 30 HiddenStateAdapter tests covering construction, latent_space property, encode shape/determinism/nonlinearity/validation, ModelAdapter conformance, DecodableAdapter and FlatBatchDecodableAdapter non-conformance, ActivationPatch rejection, and reproducibility. (#sprint-14)

- **First composition showcase** — Sprint 13 end-to-end story: generate synthetic 8D cluster data → train VAE → encode to 3D latent → PCA Layer A introspection (source/target/failure regions visible) → baseline metrics → ActivationPatch Layer B edit (encode → patch → decode) → post-edit metrics (68.2% distance improvement toward target) → Lerp trajectory panel → composite 2×2 figure. Proves existing primitives compose without new abstractions. (#sprint-13)
- Showcase config `scripts/showcase_config.py` — lightweight local dict (not framework-wide config system) with seed 42, data generation params, VAE params, split config, and output paths. Guarantees reproducibility for the Sprint 13 narrative. (#sprint-13)
- Showcase script `scripts/end_to_end_showcase_demo.py` — orchestration entry point implementing the full compose narrative. Reuses existing patterns from `end_to_end_vae_demo.py`, `end_to_end_activation_patch_demo.py`, and `end_to_end_lerp_demo.py` rather than duplicating. (#sprint-13)
- Test suite: 18 showcase tests covering config parsing, data generation shapes/range/labels, split correctness, baseline metric semantics (finite, positive), post-edit improvement verification, input non-mutation, PCA projection shapes, and trajectory panel lengths. (#sprint-13)

- `ActivationPatch` — B-Method #3 (Layer B / Manipulation), model-mediated activation patching that works through a ModelAdapter (encode → patch → decode). The first B-Method whose output is in **data space** (not latent space), with a required adapter (VAE/RandomProjection). Supports `fit(source_data, target_data)` to learn a latent delta, `__call__(input_data)` for data-space patching, and `apply_trajectory(trajectory)` returning `np.ndarray` (not `Trajectory`). Pure numpy public surface — torch used internally via adapter. (#sprint-12)
- `BMethod` Protocol — frozen structural `typing.Protocol` defining `space` / `is_fitted` / `apply_trajectory`, promoted to public surface. Deliberately excludes `__call__` (signatures differ across instances). Frozen at B-Method #3 (ActivationPatch, Sprint 12), validated by 3 distinct philosophies: stateless latent→latent (Lerp), stateful latent→latent (SteeringVector), model-mediated data→data (ActivationPatch). (#sprint-12)
- End-to-end demo script `scripts/end_to_end_activation_patch_demo.py` — two scenarios: (A) VAE latent arithmetic with blob cluster data → patch reconstruction morphs from cluster A toward cluster B, (B) trajectory patching with Lerp trajectory → decode each point → visualize data-space morphing sequence. 2×2 matplotlib grid with PCA latent space arrow and trajectory comparison. (#sprint-12)
- Test suite: 29 ActivationPatch + BMethod Protocol tests covering construction with VAE/RandomProjection, structured decoder rejection, space delegation, is_fitted lifecycle, fit delta computation, `__call__` semantic correctness (moves toward target), input non-mutation, error cases (before-fit, empty, mismatched dim), apply_trajectory returning ndarray with correct shape, and Protocol conformance checks (Lerp/SteeringVector/ActivationPatch all pass; non-conforming objects fail). (#sprint-12)
- `SteeringVector` — B-Method #2 (Layer B / Manipulation), stateful steering method that learns a unit direction from contrast pairs (`fit(positives, negatives)`) and applies it to latent vectors (`__call__(latent, strength)`). Supports `direction` property, `apply_trajectory(trajectory, strength)` for trajectory-level steering, and optional `LatentSpace` for geometry-aware post-steer normalization (e.g. project back to sphere). Pure numpy — no torch leakage. (#sprint-11)
- End-to-end demo script `scripts/end_to_end_steering_demo.py` — two scenarios: (A) Euclidean steering with synthetic 8D contrast clusters → PCA to 2D → steering path at multiple strengths, (B) spherical steering with unit-norm 3D contrast data → geometry-aware normalization → points stay on sphere. 1×2 matplotlib visualization with arrows and strength annotations. (#sprint-11)
- Test suite: 32 SteeringVector tests covering construction with/without LatentSpace, fit direction learning, direction property (before/after fit), `__call__` edge cases (zero/negative/wrong-dim/strength scaling), input non-mutation, apply_trajectory shape/semantics, spherical normalization at multiple strengths, and no-torch-leakage verification. (#sprint-11)

- **`ModelAdapter` 3-mode ADR gained mode (ii) evidence** — modes (i) and (ii) confirmed by running code (VAE, RandomProjection, HiddenStateAdapter). The Protocol split was frozen at instance #3, while full ADR validation remained pending until Sprint 16's deterministic renderer mode (iii). (#sprint-14)
- **`_ModelAdapterBase` removed** — superseded by frozen `ModelAdapter`/`DecodableAdapter` Protocols. The UNSTABLE internal sketch (`_base.py`) is replaced by the public Protocol surface. (#sprint-14)
- **`ActivationPatch`** — adapter parameter now has a runtime `isinstance` guard requiring `FlatBatchDecodableAdapter`. `HiddenStateAdapter` (no `decode`) and structured decoders like `GaussianRendererAdapter` are cleanly rejected at construction with `TypeError`. (#sprint-14, #sprint-16)
- **`adapters/__init__.py`** — exports `ModelAdapter`, `DecodableAdapter`, `FlatBatchDecodableAdapter`, and `HiddenStateAdapter`. Docstring updated to reflect frozen Protocol design. (#sprint-14, #sprint-16)
- **`VAE` and `RandomProjection` docstrings** — updated to reference `ModelAdapter` and `DecodableAdapter` Protocols instead of the removed `_ModelAdapterBase`. (#sprint-14)

- **`BMethod` Protocol frozen** — `_b_base.py` removed, superseded by frozen `BMethod` Protocol in `methods/b_protocols.py`. Lerp and SteeringVector migrated to note Protocol conformance. This is the third Rule of Three freeze in the project (after `Method` at Sprint 6 and geometry dispatch patterns at Sprint 9). The separation of A (`Method`) and B (`BMethod`) Protocols confirms that A/B/C layers have fundamentally different method shapes — the aspirational unified interface is disproven by code. (#sprint-12)
- `Lerp` — added `is_fitted` property (always `True` — stateless methods are always ready) and generic `apply_trajectory(**kwargs)` method supporting `other`+`t` (delegates to `between`) or `n_steps` (delegates to `blend_sequence`). Docstring updated to note `BMethod` Protocol conformance. (#sprint-12)
- `SteeringVector` — docstring updated to note `BMethod` Protocol conformance. No code changes (already conforms structurally). (#sprint-12)

- `Lerp` — B-Method #1 (Layer B / Manipulation), stateless interpolation method wrapping `LatentSpace.interpolate()` for geometry-aware dispatch. Supports `__call__(a, b, t)` for single-point interpolation, `between(traj_a, traj_b, t)` for pointwise trajectory interpolation, and `blend_sequence(trajectory, n_steps)` for trajectory densification. Pure numpy — no torch leakage. (#sprint-10)
- End-to-end demo script `scripts/end_to_end_lerp_demo.py` — two scenarios: (A) Euclidean lerp with two random 8D vectors → PCA to 2D, (B) spherical slerp on unit-norm vectors → PCA projection → lerp-vs-slerp path comparison → trajectory blending with `blend_sequence`. 1×2 matplotlib visualization with t-value annotations and sphere reference outline. (#sprint-10)
- Test suite: 28 Lerp tests covering construction with/without LatentSpace, correct interpolation (t=0→a, t=1→b, t=0.5→midpoint), geometry dispatch (slerp stays on sphere), trajectory between (shape, endpoints, error cases), and blend_sequence (densification, endpoint preservation, edge cases). (#sprint-10)
- **Giai đoạn 2 begins** — first Layer B method. Both validated ADRs (geometry-keyed `LatentSpace`, geometry-dispatch) exercised from the consumer side. `ModelAdapter` 3-mode ADR remains pending. (#sprint-10)

- Spherical geometry (`unit_norm`) as geometry case #2 for `LatentSpace` — validates the geometry-keyed and geometry-dispatch ADRs with real code. `geometry` moved from class-level constant to instance-level parameter, validated at construction. (#sprint-9)
- `LatentSpace.distance(a, b) -> float` — dispatches on `self.geometry`: Euclidean (`||a-b||`) or angular (`arccos`). (#sprint-9)
- `LatentSpace.interpolate(a, b, t) -> np.ndarray` — dispatches on `self.geometry`: lerp for Euclidean, proper slerp for spherical (geodesic on unit sphere) with edge-case handling for `sin(ω) ≈ 0`. (#sprint-9)
- `LatentSpace.normalize(point) -> np.ndarray` — euclidean returns copy; unit_norm projects to unit sphere. Zero vector raises for spherical. (#sprint-9)
- End-to-end demo script `scripts/end_to_end_spherical_demo.py` — synthetic unit-norm data → LatentSpace with unit_norm geometry → demonstrates validate_point, angular distance, slerp-vs-lerp, and normalization. 1×3 matplotlib visualization (3D scatter + lerp path + slerp path). (#sprint-9)
- Test suite: 35 new LatentSpace tests covering geometry construction, validate_point for unit_norm, distance (6 cases including known angles), interpolate/slerp (midpoint, endpoints, unit-norm invariance, edge cases), and normalize. (#sprint-9)
- **ADR milestone**: `LatentSpace` geometry-keyed ADR and geometry-dispatch ADR both moved from `pending` → `validated`. This is the last sprint of Giai đoạn 1 — core primitives confirmed with two geometry cases. (#sprint-9)

- RandomProjection adapter — ModelAdapter #2 (fixed-weight/stateless, pretrained pattern), pure numpy with Johnson-Lindenstrauss-style random Gaussian projection matrix, `encode`/`decode`/`latent_space`, no `fit` method. (#sprint-8)
- Internal `_ModelAdapterBase` ABC sketching the shared ModelAdapter shape (`encode`/`decode`/`latent_space`), marked UNSTABLE and not part of the public surface. `fit` deliberately excluded (VAE-specific). (#sprint-8)
- End-to-end demo script `scripts/end_to_end_random_projection_demo.py` — synthetic cluster data → RandomProjection encode → LatentSpace → Trajectory → PCA/UMAP → 1×3 matplotlib visualization (PCA original, PCA latent, UMAP latent) demonstrating JL distance preservation. (#sprint-8)
- Test suite: 24 RandomProjection tests covering construction, projection matrix normalisation, encode/decode shapes and error cases, roundtrip, reproducibility (same/different seeds), and approximate Johnson-Lindenstrauss distance preservation. (#sprint-8)
- VAE (Variational Autoencoder) adapter — ModelAdapter #1 (mode i: explicit learned latent), torch-based with `encode`/`decode`/`latent_space` and all-numpy public surface. Trains from scratch on [0,1]-scaled data with MSE + KL divergence loss. (#sprint-7)
- `adapters` sub-package (`src/latent_anything/adapters/`) — namespace for model adapter implementations, sibling to `methods/`. Not exported from top-level `__init__.py`. (#sprint-7)
- End-to-end demo script `scripts/end_to_end_vae_demo.py` — synthetic cluster data → VAE training → encode → LatentSpace → Trajectory → PCA + UMAP → 2×2 matplotlib visualization (original, reconstruction, PCA latent, UMAP latent). First pipeline exercising ALL Layer A primitives together. (#sprint-7)
- Test suite: 23 VAE tests covering construction, `hidden_dim` heuristic, `latent_space` property, fit/loss, encode/decode shape invariants, error cases, roundtrip reconstruction, and `random_state` reproducibility. (#sprint-7)
- SAE (Sparse Autoencoder) dimensionality reduction method — Method #3 with fundamentally different philosophy (gradient-descent training, L1 sparsity, encoder/decoder architecture), torch-based with all-numpy public surface. (#sprint-6)
- `Method` Protocol — frozen structural `typing.Protocol` defining `fit(data: np.ndarray) -> None` / `transform(data: np.ndarray) -> np.ndarray`, promoted to public surface as third core primitive. (#sprint-6)
- End-to-end demo script `scripts/end_to_end_sae_demo.py` — synthetic latent → Trajectory → SAE training → sparse feature projection → side-by-side PCA vs UMAP vs SAE 2D visualization with matplotlib. (#sprint-6)
- Test suite: 16 SAE tests covering construction, fit, transform, sparsity verification, `random_state` reproducibility, error cases, and Protocol conformance smoke tests. (#sprint-6)
- UMAP dimensionality reduction method (`fit`/`transform`/`fit_transform`) wrapping `umap-learn` with numpy public surface, stochastic and stateful. (#sprint-5)
- Internal `_MethodBase` base class sketching the shared shape for stateful methods (`fit`/`transform`/`fit_transform`), marked UNSTABLE and not part of the public API. (#sprint-5)
- PCA migrated to `_MethodBase` inheritance, removing the now-redundant `fit_transform` override. (#sprint-5)
- End-to-end demo script `scripts/end_to_end_umap_demo.py` — synthetic latent → Trajectory → UMAP fit → 2D projection → side-by-side matplotlib visualization with PCA comparison. (#sprint-5)
- Test suite: 12 UMAP tests covering construction, fit/transform shape invariants, fit_transform roundtrip, `random_state` reproducibility, and input validation errors. (#sprint-5)

- `_MethodBase` docstring updated from "UNSTABLE — will be replaced" to "Internal convenience base backed by the frozen `Method` Protocol" following Rule of Three (Method #3 freeze trigger). (#sprint-6)
- PCA and UMAP docstrings updated to note conformance to the frozen `Method` Protocol. (#sprint-6)
- `LatentSpace` concrete class for Euclidean flat vector spaces with `dim`, `geometry`, `source_model`, metadata, and `validate_point()`. (#sprint-4)
- `Trajectory` immutable sequence class holding 2D numpy latent arrays with `len`, indexing/slice (returns new `Trajectory`), and `to_numpy()`. (#sprint-4)
- PCA dimensionality reduction method (`fit`/`transform`/`fit_transform`) wrapping scikit-learn with numpy public surface. (#sprint-4)
- End-to-end demo script `scripts/end_to_end_pca_demo.py` — synthetic latent → Trajectory → PCA → 2D matplotlib visualization. (#sprint-4)
- Test suite: 41 tests covering LatentSpace (11), Trajectory with hypothesis property-based tests (16), PCA fit/transform/roundtrip (12), and package smoke (2). (#sprint-4)
- Package scaffold: `src/latent_anything/` (src-layout) with root-level `pyproject.toml` and PEP 621 metadata.
- Development tooling: `ruff` (lint + format), `pyright` (strict type-checking), `pytest`, and `hypothesis` configured in `pyproject.toml`.
- Smoke test suite: `tests/test_latent_anything/test_package.py` verifying import and version.
- CI workflow (`.github/workflows/ci.yml`) running `ruff check`, `ruff format --check`, `pyright`, and `pytest` on Python 3.12, 3.13, and 3.14.
- README updated with package installation via `uv`, Quick Start example, and current project structure.
- Python version range adjusted: `requires-python` set to `>=3.12,<3.15`, local development targets Python 3.13.
- Deploy workflow restricted: `deploy-latent-anything-theory.yml` now triggers only on `theory-v*` tag pushes or manual `workflow_dispatch`.

### Fixed

- Made release-note extraction fail when a matching changelog section contains only a version heading and no release body content.
- Prevented `AnalysisPipeline` cache entries from crossing between adapters that share hyperparameters but have different learned or randomly initialized state.
- Ensured cached analysis runs always fit the current stateful Layer A method instead of returning a transformed array while leaving the method unfitted.
- Narrowed the frozen `BMethod.apply_trajectory` invariant to the shared trajectory argument while preserving method-specific optional arguments on concrete implementations.
- Restored strict Pyright coverage for every Python file changed in Sprints 17–25.
- Expanded the full test suite to 596 passing tests with cache identity and state-consistency regressions.
