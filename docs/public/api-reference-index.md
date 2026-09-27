# Choose a method and find a released API

This index is pinned to the published `latent-anything==1.0.0` contract. The canonical root surface is the **211-name** `canonical_stable_surface` in the [frozen API snapshot](https://github.com/triet4p/latent-anything/blob/v1.0.0/artifacts/api_freeze_snapshot_1.0.0.json) (snapshot SHA-256 `46ef4edd2bcb5d6235a02633caba10408b6e2000d5614928bef0794b20b005f4`). Each name in the coverage map appears once and means `from latent_anything import Name`. The route is the public guide or authoritative specialist reference for that name. No signatures are copied here: use the [release-tagged API reference](https://github.com/triet4p/latent-anything/blob/v1.0.0/docs/API_REFERENCE.md) and snapshot for exact names, fields, defaults, and compatibility details.

Classification is about the guide route, not a quality or support tier:

- **Worked-example API** — exercised in a runnable public guide example.
- **Explained supporting API** — explained as an input, result, configuration, helper, or related contract, but not given a separate runnable example.
- **Specialist/reference API** — route first to the specialist reference or frozen contract; the general guide does not present it as a default path.

## Choose a route

| If your task is… | Start here | Keep in mind |
| --- | --- | --- |
| Represent an encoder output, define axes/identity, or adapt an existing encoder/decoder | [Latent primitives and adapters](latent-primitives.md) or [decoder-free hidden states](hidden-state-path.md) | Shape equality alone does not establish matching model, layer, checkpoint, or preprocessing identity. |
| Project, probe, cluster, score density, evaluate sparse features, attribute a target, or analyze ordered trajectories | [Inspect and explain](inspect-explain.md) | A plot, probe score, density score, or attribution is not by itself a causal or general-model claim. |
| Edit a latent, fit a contrast direction, remove a subspace, or compare an adapter-mediated intervention | [Intervene and compare](intervene-compare.md) | A latent edit is not a model-forward activation hook; keep identity, null, and off-target controls. |
| Train a small VAE or connect the optional Diffusers VAE seam | [VAE paths](vae-paths.md) | Built-in VAEs are small examples; optional dependencies and model evidence are separate boundaries. |
| Select geometry-aware interpolation or structured pose types | [Geometry and pose routes](#geometry-and-pose) | Prefer the simplest geometry-correct path; do not treat an optimizer or geometry label as validation. |
| Use a Gaussian renderer or another structured 3D adapter | [3D and structured-renderer route](#3d-and-structured-renderer-exports) | These are `latent_anything.adapters` submodule exports and optional integrations, not flat-batch decoder guarantees. |
| Predict transitions, evaluate a world-model rollout, or choose a planner | [World models and transitions](#world-models-and-transitions) and [planning](#planning-and-reward-evaluation) | Evidence is model- and task-specific; the included specialist lanes do not validate arbitrary world models or hardware. |
| Work with discrete codes or tokenized latent rollouts | [Discrete latents](#discrete-latents-and-token-rollouts) | Code IDs are categorical; do not apply continuous latent arithmetic unless the API explicitly defines it. |
| Record local experiments or connect MLflow/W&B | [Experiment tracking](#experiment-tracking) | Provider adapters are optional and deliberately narrower than hosted-service or general tracking support. |
| Find a CLI command, async counterpart, exception, or exact config/result schema | [Release-wide contracts](#release-wide-contracts) and the [frozen API reference](https://github.com/triet4p/latent-anything/blob/v1.0.0/docs/API_REFERENCE.md) | CLI metadata, config/result fields, sync/async declarations, and exception bases are snapshot-derived; signatures are intentionally not duplicated here. |

## Geometry and pose

Use the [density-penalized geodesic guide](https://github.com/triet4p/latent-anything/blob/main/docs/GEODESIC_INTERPOLATION.md) when a fitted density oracle and curved, low-density-crossing paths justify a bounded path optimizer. If the latent is flat or unit-norm, prefer the simpler geometry-appropriate interpolation described by the [intervention guide](intervene-compare.md) rather than optimizing a path unnecessarily. `SE3`, `SO3`, and the pose value/configuration types are structured reference APIs; their exact conventions belong to the [frozen API reference](https://github.com/triet4p/latent-anything/blob/v1.0.0/docs/API_REFERENCE.md) and snapshot, not to an inferred flat-vector convention.

## 3D and structured-renderer exports

`GaussianRendererAdapter`, `Gaussian3DRendererAdapter`, and `GaussianCamera` are available from the `latent_anything.adapters` submodule, not as additional names in the 211-name root map. Consult the release reference and [optional-integration boundaries](https://github.com/triet4p/latent-anything/blob/main/docs/OPTIONAL_INTEGRATIONS.md) before installing the `3d` extra. A structured renderer may have decode semantics without satisfying `FlatBatchDecodableAdapter`; do not route it into `ActivationPatch` solely because it can render.

The released guide does **not** claim arbitrary 3DGS checkpoints, Open3D/trimesh adapters, general model support, or CUDA validation. Optional dependency availability is not a model, checkpoint, or accelerator evidence claim.

## World models and transitions

Choose the exact representation family before composing rollout code:

- [JEPA/LeWM world model](https://github.com/triet4p/latent-anything/blob/main/docs/JEPA_WORLD_MODEL.md) describes a decoder-free mean-transition adapter and its bounded synthetic CPU evidence.
- [RSSM-style transition](https://github.com/triet4p/latent-anything/blob/main/docs/RSSM_TRANSITION.md) documents recurrent state, masked variable-length sequences, and the retained rollout-drift limitations.
- [Tokenized world model](https://github.com/triet4p/latent-anything/blob/main/docs/TOKENIZED_WORLD_MODEL.md) covers categorical codes, seeded sampling, codebook checks, and its synthetic evidence limits.
- The [pipeline reference](https://github.com/triet4p/latent-anything/blob/main/docs/PIPELINES.md) describes the shared predictive-mean transition and rollout composition boundaries.

These are distinct reference implementations, not interchangeable model adapters or evidence for an arbitrary checkpoint. The root entries below identify released types; the exact current method signatures and serialized fields remain in the tagged API reference/snapshot.

## Planning and reward evaluation

Use the released CEM or MPPI planner family when the action bounds, horizon, population, objective, and evaluation loop match its contract. The [pipeline specialist reference](https://github.com/triet4p/latent-anything/blob/main/docs/PIPELINES.md) explains planner composition and rollout ownership; the frozen API reference/snapshot carries exact configuration and result schemas. `RewardValueEvaluator` and the related scoring/value APIs are a separate evaluation route—do not equate predicted return, reward/value estimates, and measured task success.

The `lerobot`, `lerobot-diffusion`, and `lerobot-smolvla` extras are install profiles, not claims that every policy is supported. The SmolVLA lane is **BLOCKED**, non-gating, and must not be presented as validated.

## Discrete latents and token rollouts

For categorical image-code representations, use the [VQ-VAE discrete-latent reference](https://github.com/triet4p/latent-anything/blob/main/docs/VQ_VAE_INTEGRATION.md). `VQVAE` is a submodule adapter export (`latent_anything.adapters.VQVAE`), not a root-map entry. Its integer codes are discrete; code replacement is not continuous interpolation. For action-conditioned token dynamics, use the [tokenized world-model reference](https://github.com/triet4p/latent-anything/blob/main/docs/TOKENIZED_WORLD_MODEL.md), which distinguishes greedy/mean transition behavior from seeded sampling and documents synthetic CPU limits.

## Experiment tracking

`FileSystemRunRecorder` is the core local run-record route shown in the compose/persist example. Provider adapters are separate optional integration modules; see the [experiment-tracking contract](https://github.com/triet4p/latent-anything/blob/main/docs/OPTIONAL_INTEGRATIONS.md#experiment-tracking-sprint-76) for the `tracking-mlflow`, `tracking-wandb`, and combined `tracking` profiles. MLflow is limited to a local file tracking URI in that guide; W&B is limited to offline or disabled mode. Provider adapter class names are not extra canonical root exports, and installing an SDK does not imply hosted-service, authentication, dashboard, or general tracking validation.

## Release-wide contracts

The frozen snapshot records **28 configuration schemas** in section G and **89 public dataclass/result schemas** in section H. Their top-level names are routed in the canonical map below; consult those snapshot sections for field types/defaults rather than duplicating signatures or field tables in this index.

Section I lists five CLI commands: `capture-points` (alias `list-capture-points`), `compare-runs`, `inspect-dataset`, `inspect-policy`, and `replay-run` (alias `replay-run-config`). See the release reference's [CLI contract](https://github.com/triet4p/latent-anything/blob/v1.0.0/docs/API_REFERENCE.md#cli-contract); these commands are not additional root API names.

The 12 optional profiles, in frozen declaration order, are `docs`, `diffusers`, `transformers`, `diffusers-full`, `3d`, `lerobot`, `lerobot-diffusion`, `lerobot-smolvla`, `viz`, `tracking-mlflow`, `tracking-wandb`, and `tracking`. The [optional integrations guide](https://github.com/triet4p/latent-anything/blob/main/docs/OPTIONAL_INTEGRATIONS.md) describes their boundaries. The base package does not import provider SDKs by default.

The snapshot records nine synchronous/asynchronous pairs. Section K uses `ManipulationPipeline` as the class spelling for the public `InterventionPipeline` alias: `AnalysisPipeline.run` / `AnalysisPipeline.run_async`; `ManipulationPipeline.run_data` / `ManipulationPipeline.run_data_async` and `ManipulationPipeline.run_trajectory` / `ManipulationPipeline.run_trajectory_async`; `RolloutPipeline.run` / `RolloutPipeline.run_async` and `RolloutPipeline.stream` / `RolloutPipeline.stream_async`; and `BatchExecutor.decode` / `BatchExecutor.decode_async`, `BatchExecutor.encode` / `BatchExecutor.encode_async`, `BatchExecutor.map_array` / `BatchExecutor.map_array_async`, and `BatchExecutor.transform` / `BatchExecutor.transform_async`. `stream_async` is an async generator; the other listed async counterparts are coroutines. See [runtime and async boundaries](compose-persist-extend.md#batching-cache-profiling-async-and-streaming-boundaries); exact signatures remain in the snapshot.

Section L records eight custom exceptions: `ArtifactStoreError`, `DiagnosticRequestError`, `RecorderContractError`, `PluginContractError`, `PortableNodeError`, `PortableResultError`, `DuplicateRunError`, and `DiskCacheError`. `RecorderContractError` and `PluginContractError` are module-scoped entries rather than canonical root exports; the root-export map includes only names in `A_public_surface.canonical_stable_surface`.

## Submodule method, pipeline, adapter, and runtime re-exports

The following are the snapshot's explicit submodule re-export routes from section C. These are alternate import paths, **not** extra entries in the 211-name root count; a name may also appear in the root map or more than one submodule. The list is deliberately names-only.

| Released submodule | Re-exported names | Reader route |
| --- | --- | --- |
| `latent_anything.methods` | `ActivationPatch`, `BMethod`, `Method`, `PCA`, `SAE`, `UMAP`, `Lerp`, `SteeringVector`, `AnalysisMethod`, `Intervention` | [Inspect/explain](inspect-explain.md) and [intervene/compare](intervene-compare.md) |
| `latent_anything.pipeline` | `AnalysisPipeline`, `ManipulationPipeline`, `PipelineContract`, `PipelineResult`, `PipelineSpec`, `RolloutPipeline`, `RolloutPipelineSpec`, `RewardValueEvaluationSpec`, `RolloutResult`, `ManipulationPipelineSpec`, `CEMPlannerSpec`, `MPPIPlannerSpec`, `build_cem_planner_from_config`, `build_mppi_planner_from_config`, `build_manipulation_pipeline_from_config`, `build_pipeline_from_config`, `build_reward_value_evaluator_from_config`, `build_rollout_pipeline_from_config`, `InterventionPipeline` | [Compose and persist](compose-persist-extend.md) and [planning](#planning-and-reward-evaluation) |
| `latent_anything.adapters` | `ConvVAE`, `DecodableAdapter`, `FlatBatchDecodableAdapter`, `GaussianRendererAdapter`, `Gaussian3DRendererAdapter`, `GaussianCamera`, `HiddenStateAdapter`, `JEPAWorldModelAdapter`, `JEPAWorldModelConfig`, `JEPALatentHealth`, `JEPAEvaluationReport`, `JEPAPrediction`, `JEPAPredictionMetrics`, `JEPARolloutMetrics`, `ModelAdapter`, `RandomProjection`, `VAE`, `VQVAE`, `TokenPrediction`, `TokenPredictionMetrics`, `TokenRolloutMetrics`, `TokenizedEvaluationReport`, `TokenizedWorldModel`, `TokenizedWorldModelConfig` | [Adapters](latent-primitives.md#adapter-capabilities), [VAE](vae-paths.md), [3D](#3d-and-structured-renderer-exports), [world models](#world-models-and-transitions), and [discrete latents](#discrete-latents-and-token-rollouts) |
| `latent_anything.runtime` | `BatchExecutor`, `CacheKey`, `CacheStats`, `DiskCacheError`, `DiskCacheStats`, `InMemoryCache`, `ProfileEvent`, `RuntimeProfile`, `RuntimeProfiler`, `SQLiteDiskCache`, `hash_array`, `hash_component_config`, `hash_component_state`, `make_cache_key`, `make_disk_cache_key` | [Runtime boundaries](compose-persist-extend.md#batching-cache-profiling-async-and-streaming-boundaries) |
| `latent_anything.registry` | `GLOBAL_REGISTRY`, `KIND_ADAPTER`, `KIND_ANALYSIS`, `KIND_INTERVENTION`, `KIND_METHOD_A`, `KIND_METHOD_B`, `KIND_PLANNER`, `KIND_RUNTIME`, `Registry`, `RegistryEntry`, `list_entries`, `lookup`, `register` | [Registry and plugin configuration](compose-persist-extend.md#registry-configuration-and-plugin-discovery) |
| `latent_anything.methods.protocols` | `AnalysisMethod`, `Method` | [Adapter and method protocols](latent-primitives.md#adapter-capabilities) |
| `latent_anything.methods.b_protocols` | `Intervention`, `BMethod` | [Intervention guide](intervene-compare.md) |
| `latent_anything.manipulation_pipeline` | `InterventionPipeline`, `ManipulationPipeline` | [Pipeline comparison](compose-persist-extend.md#choose-by-input-and-lifecycle) |

## Canonical root-export coverage map

Each name in this table is classified exactly once. All entries use the canonical root import path `latent_anything`; the named route is specific to the topic and classification. Chunks split long name groups for readability and do not change their classification or route.

| Classification | Guide or specialist route | Canonical stable root exports |
| --- | --- | --- |
| Worked-example API | [Latent primitives example](latent-primitives.md#shapes-and-coordinate-meaning) | `LatentSpace`, `LatentValue`, `Trajectory` |
| Worked-example API | [Inspect/explain CPU example](inspect-explain.md#runnable-cpu-example-projection-probe-and-clusters-on-known-data) | `KMeans`, `KMeansConfig`, `KMeansResult`, `LinearProbe`, `LinearProbeConfig`, `LinearProbeResult`, `compare_with_labels` |
| Worked-example API | [Intervention CPU example](intervene-compare.md#runnable-cpu-example-compare-identity-changed-and-off-target-results) | `SubspaceProjection` |
| Worked-example API | [Compose/config CPU example](compose-persist-extend.md#prerequisites-and-a-bounded-cpu-example) | `AnalysisPipeline`, `InterventionPipeline`, `PipelineResult`, `PipelineSpec`, `build_pipeline_from_config` |
| Worked-example API | [Registry configuration](compose-persist-extend.md#registry-configuration-and-plugin-discovery) | `ObjectSpec` |
| Worked-example API | [Runtime profiling](compose-persist-extend.md#batching-cache-profiling-async-and-streaming-boundaries) | `RuntimeProfiler` |
| Worked-example API | [Run records and typed envelopes](compose-persist-extend.md#run-records-typed-envelopes-and-what-round-tripping-proves) | `encode_result_envelope`, `decode_result_envelope`, `FileSystemRunRecorder`, `RunRecord` |
| Explained supporting API | [Latent identity and adapter protocols](latent-primitives.md#adapter-capabilities) | `AnalysisMethod`, `coordinate_identity`, `assert_arithmetic_compatible` |
| Explained supporting API | [Inspect/explain methods](inspect-explain.md#choose-by-question) | `ClusterStabilityReport`, `ConceptDataset`, `ConceptDirectionResult`, `ControlBaselines`, `CovarianceConfig`, `CovarianceState`, `CrossSeedReport`, `FeatureAtlas`, `FeatureAtlasEntry`, `FeatureCrossCheck`, `FeatureRanking`, `IntegratedGradients`, `IntegratedGradientsConfig`, `IntegratedGradientsResult`, `MLPProbe` |
| Explained supporting API | [Inspect/explain methods](inspect-explain.md#choose-by-question) | `MLPProbeConfig`, `MLPProbeResult`, `ProbeComparison`, `TCAV`, `TCAVConfig`, `TCAVResult`, `TCAVScore`, `TransformerLogitTarget`, `SAEConfig`, `SAEEvaluationResult`, `SAEFeatureEvaluation`, `SAEFeatureMetrics`, `SAEStabilityResult`, `SensitivityReport`, `BoundaryMetrics` |
| Explained supporting API | [Inspect/explain methods](inspect-explain.md#choose-by-question) | `ChangePointResult`, `Segment`, `SegmentationConfig`, `SmoothedTrajectory`, `SmoothingConfig`, `build_feature_atlas`, `GMMConfig`, `GaussianMixtureDensity`, `DensityResult`, `DensityMetrics`, `DensityEvaluationReport`, `DensityStabilityReport`, `DTWConfig`, `DTWCostSummary`, `DTWResult` |
| Explained supporting API | [Inspect/explain methods](inspect-explain.md#choose-by-question) | `density_cross_seed_evaluation`, `mahalanobis_baseline`, `fit_covariance_state`, `compute_dtw`, `indexwise_distance`, `detect_change_points`, `evaluate_boundaries`, `smooth_trajectory`, `smoothing_distortion`, `check_clustering_geometry`, `cluster_stability_analysis`, `compare_probes`, `compute_integrated_gradients`, `compute_tcav`, `cross_check_feature` |
| Explained supporting API | [Inspect/explain methods](inspect-explain.md#choose-by-question) | `cross_seed_evaluation`, `cross_seed_sae_stability`, `evaluate_layers`, `evaluate_sae_features`, `evaluate_sensitivity`, `intervention_agreement`, `learn_linear_separator_direction`, `learn_mean_diff_direction`, `load_feature_atlas`, `nonlinear_memorization_test`, `rank_feature_examples`, `save_feature_atlas` |
| Explained supporting API | [Intervention methods](intervene-compare.md#choose-the-operation) | `OrthonormalSubspace`, `SubspaceProjectionConfig` |
| Explained supporting API | [Registry and config helpers](compose-persist-extend.md#registry-configuration-and-plugin-discovery) | `GLOBAL_REGISTRY`, `Registry`, `RegistryEntry`, `build_from_config`, `build_from_dict`, `list_entries`, `lookup_entry`, `register_entry` |
| Explained supporting API | [Pipeline lifecycles and specs](compose-persist-extend.md#choose-by-input-and-lifecycle) | `PipelineContract`, `ManipulationPipelineSpec`, `CEMPlannerSpec`, `MPPIPlannerSpec`, `RolloutPipeline`, `RolloutPipelineSpec`, `RolloutResult`, `RewardValueEvaluationSpec`, `build_manipulation_pipeline_from_config`, `build_cem_planner_from_config`, `build_mppi_planner_from_config`, `build_reward_value_evaluator_from_config`, `build_rollout_pipeline_from_config` |
| Explained supporting API | [Runtime, batching, and caches](compose-persist-extend.md#batching-cache-profiling-async-and-streaming-boundaries) | `BatchExecutor`, `CacheKey`, `CacheStats`, `DiskCacheError`, `DiskCacheStats`, `InMemoryCache`, `ProfileEvent`, `RuntimeProfile`, `SQLiteDiskCache`, `make_disk_cache_key` |
| Explained supporting API | [Portable results and run records](compose-persist-extend.md#run-records-typed-envelopes-and-what-round-tripping-proves) | `ArtifactStore`, `ArtifactStoreError`, `StoredArtifact`, `PortableLimits`, `PortableNodeError`, `encode_portable`, `decode_portable`, `PortableEnvelope`, `PortableResultError`, `ArtifactRef`, `DuplicateRunError`, `RunComparisonReport`, `build_comparison_report`, `compute_run_identity`, `migrate_run_record` |
| Specialist/reference API | [Density geodesics](https://github.com/triet4p/latent-anything/blob/main/docs/GEODESIC_INTERPOLATION.md) | `DensityGeodesic`, `GeodesicConfig`, `GeodesicPath`, `PathOptimizationStatus` |
| Specialist/reference API | [Pose and structured 3D reference](#3d-and-structured-renderer-exports) | `PoseConfig`, `PoseMetadata`, `PoseTrajectory`, `SE3`, `SO3` |
| Specialist/reference API | [World-model transition routes](#world-models-and-transitions) | `JEPAWorldModelAdapter`, `JEPAWorldModelConfig`, `JEPALatentHealth`, `JEPAEvaluationReport`, `JEPAPrediction`, `JEPAPredictionMetrics`, `JEPARolloutMetrics`, `DeterministicLatentTransition`, `GaussianPrediction`, `OneStepMetrics`, `RolloutMetrics`, `StochasticGaussianLatentTransition`, `StochasticOneStepMetrics`, `StochasticRollout`, `StochasticRolloutMetrics` |
| Specialist/reference API | [World-model transition routes](#world-models-and-transitions) | `LatentTransition`, `TokenPrediction`, `TokenPredictionMetrics`, `TokenRolloutMetrics`, `TokenizedEvaluationReport`, `TokenizedWorldModel`, `TokenizedWorldModelConfig`, `RSSMLatentTransition`, `RSSMTransitionConfig`, `RSSMPrediction`, `RSSMRollout`, `RSSMOneStepMetrics`, `RSSMRolloutMetrics` |
| Specialist/reference API | [Planning and reward evaluation](#planning-and-reward-evaluation) | `CEMConfig`, `CEMIteration`, `CEMPlanResult`, `CEMPlanner`, `MPPIConfig`, `MPPIIteration`, `MPPIPlanResult`, `MPPIPlanner`, `MPPIRecedingHorizonResult`, `HoldoutEvaluation`, `LinearRewardScorer`, `MonteCarloValueEstimator`, `RewardValueDiagnostics`, `RewardValueEvaluationResult`, `RewardValueEvaluator` |
| Specialist/reference API | [Planning and reward evaluation](#planning-and-reward-evaluation) | `TrajectoryScoreComparison`, `ValueCalibration`, `compare_real_imagined_scores`, `compute_discounted_returns`, `compute_mppi_weights` |
| Specialist/reference API | [Diagnostics and evidence limits](#diagnostics-and-evidence-boundaries) | `CaptureSelection`, `ComparisonRequest`, `ControlSelection`, `DiagnosticRequest`, `DiagnosticRequestError`, `DiagnosticResult`, `DiagnosticSelection`, `InterventionRequest`, `OutputSelection` |

The map contains 22 worked-example APIs, 123 explained supporting APIs, and 66 specialist/reference APIs: **211 unique canonical root exports total**. The coverage is frozen to the `1.0.0` snapshot, not inferred from the current source tree.

## Diagnostics and evidence boundaries

The public diagnostic API types are request/selection/result contracts; they are not a claim that every model can be diagnosed. `DiagnosticWorkflow` is **not** a stable public import. Accepted diagnostic evidence is limited to the named ordinary-DL encoder and transformer cases; SmolVLA is **BLOCKED** and non-gating, while arbitrary-model and GPU/CUDA diagnostic claims are excluded. Read the [AI engineer guide](https://github.com/triet4p/latent-anything/blob/main/docs/AI_ENGINEER_GUIDE.md), the [release's explicit non-API inventory](https://github.com/triet4p/latent-anything/blob/v1.0.0/docs/API_REFERENCE.md#explicitly-non-api-inventory), and the [support/version policy](https://github.com/triet4p/latent-anything/blob/main/docs/SUPPORT_POLICY.md) for the actual evidence and compatibility scope.