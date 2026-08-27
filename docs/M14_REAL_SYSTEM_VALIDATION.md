# M14 — Ma trận kiểm chứng hệ thống thực và hợp đồng phát hành

Tài liệu này là hợp đồng lập kế hoạch cho Sprints 78–80. M14 chưa hoàn tất và
không được hiểu là cam kết rằng các lane dưới đây đã có bằng chứng D2/D3. Mỗi
dòng là một đơn vị kiểm chứng độc lập; trạng thái chỉ được nâng từ `planned`
sang `verified` khi có artifact bất biến, revision, lệnh tái lập và người chịu
trách nhiệm. Các tên model, revision, lệnh và tên artifact giữ bằng tiếng Anh
để có thể sao chép nguyên văn.

The exhaustive theory-gap execution plan for Sprint 79 is maintained in
[EVIDENCE_GAP_PLAN](EVIDENCE_GAP_PLAN.md), with one machine-readable row for
each current D0/D1 implementation-applicable or benchmark-only item in
[`task_78.38_gap_map.json`](../artifacts/task_78.38_gap_map.json). This plan
does not change evidence levels or supersede the validator.

The user-facing migration and frozen API references are
[MIGRATION](MIGRATION.md) and [API_REFERENCE](API_REFERENCE.md); their counts
and symbol contracts are projections of the checked-in API snapshot, not a
second source of truth.

## Quy ước và điều kiện dừng

- `D0` = tài liệu; `D1` = code + focused tests; `D2` = benchmark dữ liệu không
  tầm thường; `D3` = D2 trên model pretrained/trained thực. Bằng chứng đồ họa
  hoặc notebook một mình không đạt D2.
- Mọi real-model lane phải pin `model@revision`, dataset/backend, license/access,
  device, versions, seed, network policy, resource peak và artifact hash.
- Không tải model khi lập kế hoạch. Không dùng mock để thay real model ở lane
  tích hợp. Pure algorithm phải dùng real dataset/system substitute (ghi rõ).
- Remote CUDA chỉ được chạy qua `.agents/skills/remote-cuda-test/SKILL.md`:
  commit/push trước, disposable clone đúng SHA, Bash/WSL/Git Bash, isolated
  caches, kiểm tra NVIDIA/CUDA/compiler, trap cleanup, không sửa checkout server.
- GitHub Actions cần tài khoản external có quyền; nếu thiếu thì là blocker,
  không đổi workflow để giả PASS. Sprint 80 dừng trước tag/publish nếu còn
  blocker, waiver chưa được owner ký, hoặc bằng chứng dưới ngưỡng 95% core /
  90% overall.

## 24 lane bắt buộc

| ID | Lane / public surface | Source → current tests/evidence | Real target / backend | Environment / command | Deterministic acceptance + artifact | Resources / network / cleanup | Status / blocker / owner |
|---|---|---|---|---|---|---|---|
| L01 | LatentSpace, LatentValue, PipelineContract, Result | `latent_space.py`, `latent_value.py`, `pipeline.py`, `adapters/conv_vae.py` → core tests and `scripts/m14_l01_core.py`, D2 | sklearn digits split; existing ConvVAE + AnalysisPipeline/PCA; NumPy/PyTorch CPU | local; `uv run python scripts/m14_l01_core.py` and `uv run pytest tests/test_m14_l01_core.py -q` | held-out reconstruction vs zero baseline, finite shapes/dtypes, stable schema/digest, no input mutation; `artifacts/m14/l01-core.json` | resource peak not measured; M14 estimate only; offline; output artifact/run record retained | verified D2; no blocker; API owner |
| L02 | geometry: covariance, projection, SO3/SE3, DTW, smoothing | `geometry.py`, `projection.py`, `pose.py`, `dtw.py`, `temporal.py`, `scripts/m14_l02_plan.py`, `scripts/m14_l02_data.py`, `scripts/m14_l02_metrics.py`, `scripts/m14_l02_envelope.py`, `scripts/m14_l02_geometry.py` → focused tests, D1/D2 synthetic; design contract [`l02-geometry.plan.json`](../artifacts/m14/l02-geometry.plan.json) | real sklearn digits held-out substitute with existing ConvVAE; model-induced latent sequences (not recorded physical trajectories); NumPy/SciPy CPU | local; side-effect-free validation: `uv run python -m scripts.m14_l02_geometry --check`; real runner: `uv run python -m scripts.m14_l02_geometry`; focused tests: `uv run pytest tests/test_m14_l02_geometry.py -q` | six independent manifold/T03-SLERP/T04-LERP/Riemannian/T04-SLERP/DTW verdicts with train-only density, 128 independent pair-path DTW trials, chance/shuffled/raw-pixel strong controls, endpoint/norm/angle tolerances; accepted artifact remains `artifacts/m14/l02-geometry.json` | offline; peak RSS not measured; retain run record and remove temporary plots/cache | 79.4C invocation fix; partial verified D2 for 4/6 records; manifold and DTW records failed honestly; no stable Fréchet or physical-trajectory claim; geometry owner |
| L03 | analysis: KMeans, probes, feature ranking | `clustering.py`, `probes.py`, `mlp_probe.py` → analysis tests, D2; concrete `TransformerLMIntegration` real forward path | GPT-2 hidden states on `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8`; sklearn digits labels | remote CUDA; `uv run pytest tests/test_clustering.py tests/test_probes.py tests/test_mlp_probe.py tests/test_m14_l03_analysis.py -q`; `uv run python -m scripts.m14_l03_analysis --run-real` | three independent accepted D2 records with seeded train/holdout metrics, baseline and memorization controls; `artifacts/m14/l03-analysis.json` self-digest `60bda13a4bbf68bbb6c9308cc813913fa653c37fba368fe1e4ea7a1f898ce06b`; run record `0bcaf14ef465f2ef5c5c909237d1f573596a77fa2ca51d042db74248cf4ca03a` | RTX 4060 Ti, CUDA 12.8, disposable clone/cache cleanup; report capture audit retained, raw transcript superseded and deleted | verified D2 forward-only; separate Sprint 79 transformer-hook verification passed 8/8 on exact SHA strict CUDA, resolving the structured hook/output cleanup blocker; no separate GPT-2 ModelAdapter or L11 claim; analysis owner |
| L04 | IntegratedGradients, TCAV, sensitivity/intervention controls | `integrated_gradients.py`, `tcav.py` → focused tests, D1 | GPT-2 `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8`; real prompt/label fixture | local CPU; `uv run pytest tests/test_integrated_gradients.py tests/test_tcav.py -q` | completeness/sensitivity, selectivity, seeded CI and causal intervention; `artifacts/m14/l04-explanations.json` | ~2–4 GB, network on first run; retain manifest, delete weights if disposable | planned; D3 absent until run; explanation owner |
| L05 | GaussianMixtureDensity, OOD, covariance/geodesic density | `density.py`, `geodesic.py` → density/geodesic tests, D1/D2 | real GPT-2 states + held-out prompts; sklearn GMM backend | local CPU; `uv run pytest tests/test_density.py tests/test_latent_anything/test_geodesic.py -q` | calibration, held-out AUROC threshold predeclared, path feasibility; `artifacts/m14/l05-density.json` | <3 GB; no secrets; remove cache | planned; threshold not yet evidenced; geometry owner |
| L06 | SAE, FeatureAtlas, cross-seed stability | `sae_evaluation.py`, `_sae_atlas.py` → SAE tests, D1/D2 | GPT-2 `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8`; real text corpus fixture | local CPU; `uv run pytest tests/test_sae_evaluation.py tests/test_sae_evaluation_network.py -m network -q` | dead-feature, reconstruction, cross-seed stability and atlas hash; `artifacts/m14/l06-sae.json` | ~4 GB; network only model; purge temporary corpus | planned; D3 gap; introspection owner |
| L07 | SubspaceProjection, steering, activation patch, LERP | `methods/activation_patch.py`, `methods/steering.py`, `projection.py`, `methods/lerp.py` → intervention tests, D1 | GPT-2 hidden states and VAE latents; PyTorch CPU | local; `uv run pytest tests/test_latent_anything/test_activation_patch.py tests/test_latent_anything/test_steering.py tests/test_latent_anything/test_projection.py tests/test_latent_anything/test_lerp.py -q` | paired control effect, reversibility/shape safety, no mutation; `artifacts/m14/l07-interventions.json` | <4 GB; offline after models; delete outputs | planned; cross-adapter evidence required; manipulation owner |
| L08 | ConvVAE adapter and manipulation pipeline | `adapters/conv_vae.py`, `manipulation_pipeline.py`, `pipeline.py` → ConvVAE held-out tests, D2 | sklearn digits 80/20; compact ConvVAE, PyTorch CPU | local; `uv run pytest tests/test_latent_anything/test_conv_vae.py tests/test_latent_anything/test_conv_vae_evidence.py -q` | ≥10% over all-zero baseline, non-degenerate latent, composition; `artifacts/m14/l08-convvae.json` | <1.5 GB, offline; no model download | planned; bounded local D2 only; generative owner |
| L09 | DiffusersAutoencoderKLAdapter | `integrations/diffusers_vae.py` → `test_diffusers_vae.py`, D2 | `stabilityai/sd-vae-ft-mse@31f26fdeee1355a5c34592e401dd41e45d25a493`, safetensors SHA `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`, MIT | local CPU offline after cache; `uv run pytest tests/test_diffusers_vae.py tests/test_diffusers_vae_network.py -m network -q` | direct/adapter parity, mean/posterior seeded semantics, shape/dtype/finiteness; `artifacts/m14/l09-diffusers-vae.json` | ~1.5 GB RSS, 334 MB weights; no network in validation; cache manifest only | planned; license/hash portability gate; generative owner |
| L10 | conditional diffusion capture/intervention | `integrations/diffusers_conditional.py` → conditional tests, D1 | `runwayml/stable-diffusion-v1-5@39593d56b552c3a24aeb192dd11d2a1429c3102b`; Diffusers scheduler/denoiser | remote CUDA recommended; `uv run pytest tests/test_diffusers_conditional_network.py -m network -q` | paired scheduler controls, timestep trace, intervention effect and artifact hash; `artifacts/m14/l10-diffusion.json` | high VRAM/download; license card/access must be recorded; cleanup disposable cache | planned; network/VRAM/access blocker until provisioned; diffusion owner |
| L11 | TransformerLogitTarget, hidden-state adapter | `integrations/transformer_lm.py` → transformer tests, D2 candidate | `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8`, MIT; Transformers | local CPU; `uv run pytest tests/test_transformer_lm_network.py -m network -q` | vocab 50257, 13 layers, hidden 768, logits/hidden parity; `artifacts/m14/l11-gpt2.json` | ~2–4 GB; public download; remove disposable cache | planned; capture versions/license; transformer owner |
| L12 | JEPAWorldModelAdapter and health metrics | `adapters/jepa.py` → JEPA tests, D2 structural | `facebook/ijepa_vith14_1k@be440b1cac639542ae553e71a9c7afd925ab5fac`, Transformers | local CPU or remote CUDA; `uv run pytest tests/test_latent_anything/test_jepa_checkpoint.py -m network -q` | finite latent prediction/health metrics, explicit decoder-free contract; `artifacts/m14/l12-ijepa.json` | high download/RAM; model-card license/access missing; retain manifest only | planned; license/access blocker; world-model owner |
| L13 | VQ-VAE and discrete latent adapter | `adapters/vq_vae.py` → VQ tests, D2 | `compact-vq-vae-v1`, sklearn digits, scikit-learn 1.9.0 BSD-3-Clause | local CPU; `uv run pytest tests/test_latent_anything/test_vq_vae.py -q` | perplexity >1, dead-code <1, encode/decode finite; `artifacts/m14/l13-vq.json` | <1 GB, offline; delete generated codebook | planned; no blocker; discrete owner |
| L14 | TokenizedWorldModel, TokenPrediction/Rollout | `tokenized_world_model.py` → tokenized tests, D2 with recorded failure | compact VQVAE + synthetic controlled dynamics; PyTorch CPU | local; `uv run pytest tests/test_latent_anything/test_tokenized_world_model.py -q` | prediction/rollout schema, bounded sequence, failure reproduced or fixed; `artifacts/m14/l14-tokenized.json` | <2 GB, offline; preserve failed artifact, clean temp | planned; early rollout failure must remain visible; world-model owner |
| L15 | Deterministic/Stochastic/RSSM transitions | `transition.py`, `rssm.py` → transition/RSSM tests, D2 synthetic | compact latent dynamics fixture; PyTorch CPU | local; `uv run pytest tests/test_latent_anything/test_transition.py -q` | one-step/rollout finite, seeded stochastic reproducibility, state carry; `artifacts/m14/l15-transitions.json` | <2 GB, offline; remove temp state | planned; real temporal model gap; runtime owner |
| L16 | reward/value, CEM, MPPI planning | `reward_value.py`, `cem.py`, `mppi.py` → planner tests, D2 synthetic | real recorded trajectory substitute + compact transition; NumPy/PyTorch CPU | local; `uv run pytest tests/test_reward_value.py tests/test_cem.py tests/test_cem_rollout.py tests/test_mppi.py tests/test_mppi_rollout.py -q` | predeclared return/regret and budget, deterministic seed, no invalid action; `artifacts/m14/l16-planning.json` | <2 GB, offline; preserve trace, clean scratch | planned; policy-grounded D3 gap; planning owner |
| L17 | Gaussian renderer/3D manipulation | `adapters/gaussian_renderer.py`, `integrations/gsplat_renderer.py` → renderer tests, D2/D3 candidate | named gsplat checkpoint required; `LATENT_ANYTHING_3DGS_CHECKPOINT`; gsplat 1.4–<2.0 | remote CUDA; `remote-cuda-test` runs `uv run pytest tests/test_latent_anything/test_gaussian_3d_renderer_network.py -m network -q` in disposable clone | render finite, multi-view PSNR/SSIM thresholds, intervention artifact; `artifacts/m14/l17-3dgs.json` | high VRAM/model download; access/license/hash required; disposable clone/cache | blocked; no named checkpoint; 3D owner |
| L18 | LeRobotDataset bridge/streaming | `integrations/lerobot_dataset.py` → dataset tests, D1/D2 | `lerobot/aloha_sim_insertion_human@cc571a3c661df81b566dbfde3d5c1e85fcdf7884`; LeRobot 0.6.1 | remote/Linux recommended; `uv run python scripts/lerobot_dataset_inspection.py lerobot/aloha_sim_insertion_human --revision cc571a3c661df81b566dbfde3d5c1e85fcdf7884 --output artifacts/m14/l18-dataset.json` | episode order, schema, bounded stream/resume, dataset license; `artifacts/m14/l18-dataset.json` | dataset download/storage; no credentials expected; delete local dataset | planned; upstream/license capture; LeRobot owner |
| L19 | ACT policy capture/intervention | `integrations/lerobot_act.py` → ACT tests, D2/D3 | `lerobot/act_aloha_sim_insertion_human@33259aa86eb45fdf85350280044a33d9d50e40c3`; same ALOHA dataset | remote CUDA/Linux; `remote-cuda-test` runs `uv run pytest tests/test_lerobot_act.py::test_pinned_public_act_checkpoint_pair_loads_through_lerobot_factories -m network -q` in disposable clone | action shape, paired intervention/control, simulator metric and revision manifest; `artifacts/m14/l19-act.json` | GPU/large download; model-card license not recorded; clean HF/cache | planned; license/access blocker; LeRobot owner |
| L20 | Diffusion Policy capture | `integrations/lerobot_diffusion.py` → Diffusion tests, D2/D3 | `LeTau/diffusion_aloha_insertion@6126e33`; dataset `lerobot/aloha_sim_insertion_human_image@d93d36a`; `aloha/AlohaInsertion-v0` | remote CUDA/Linux; `remote-cuda-test` runs `uv run pytest tests/test_lerobot_diffusion.py::test_pinned_public_diffusion_checkpoint_pair_loads_through_lerobot_factories -m network -q` in disposable clone | action distribution/causal simulation threshold, exact env/model/data IDs; `artifacts/m14/l20-diffusion-policy.json` | GPU/large download; access/license capture; disposable env/cache | planned; upstream access; LeRobot owner |
| L21 | SmolVLA capture/intervention | `integrations/lerobot_smolvla.py` → SmolVLA tests, D2 candidate | `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`; `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`; `libero/libero_spatial` | remote CUDA/Linux, ~16 GB GPU; `LATENT_ANYTHING_RUN_NETWORK=1 uv run pytest tests/test_lerobot_smolvla.py::test_smolvla_gpu_checkpoint_intervention_lane -m network -q` | action validity, causal intervention, simulator success threshold; `artifacts/m14/l21-smolvla.json` | ~450M bf16, large download; model/data license/access required; clean clone/cache | planned; Linux/GPU/account/access blockers; LeRobot owner |
| L22 | ArtifactStore, portable envelopes, cache, stream, recorder/tracking | `portable.py`, `artifact_store.py`, `runtime/cache.py`, `rollout_pipeline.py`, `run_record.py`, `experiment_recorder.py` → focused tests, D1/D2 | real Arrow/SQLite filesystem; MLflow/W&B optional backends | local isolated temp; `uv run pytest tests/test_portable.py tests/test_portable_results.py tests/test_artifact_store.py tests/test_latent_anything/test_cache.py tests/test_sprint75_streaming.py tests/test_run_record.py tests/test_run_record_portable.py tests/test_experiment_recorder.py tests/test_mlflow_recorder.py tests/test_wandb_recorder.py -q` | schema-v1 round trip/migration, path safety, resume/cancel, provider atomicity; `artifacts/m14/l22-runtime.json` | <1 GB, offline; tracking credentials only in opt-in lane; delete temp DB/artifacts | planned; external tracking account may block; runtime owner |
| L23 | registry/plugins/config/CLI/serialization/security | `registry.py`, `_plugin_builtins.py`, `plugin_discovery.py`, `plugin_groups.py`, `plugin_metadata.py`, `cli.py`, `config.py`, `pipeline_config.py` → registry/plugin/CLI tests, D1 | all 32 built-ins, 5 entry-point groups, hello-world separately installed plugin; base + each extra | local clean env; `uv run pytest tests/test_plugin_discovery.py tests/test_plugin_groups.py tests/test_plugin_installation.py tests/test_plugin_metadata.py tests/test_cli.py tests/test_registry_migration.py tests/test_latent_anything/test_registry.py tests/test_latent_anything/test_config.py tests/test_api_compatibility.py -q` | import isolation, discovery, install/uninstall, schema migrations, negative paths, secret/path/zip safety; `artifacts/m14/l23-contract.json` | <2 GB, no network except plugin install fixture; uninstall temp plugin/cache | planned; external GitHub Actions account blocker; API owner |
| L24 | packaging/docs/performance/release gates | `pyproject.toml`, `docs/M14_REAL_SYSTEM_VALIDATION.md`, `scripts/validate_evidence_ledger.py`, `.github/workflows/ci.yml`, `.github/workflows/optional-extras.yml`, `.github/workflows/release.yml` → full tests, MkDocs, Ruff/Pyright | clean base/all 12 profiles, supported Python/platform tiers, wheel/sdist | local then remote CI; `uv sync --locked --all-extras`; `uv run pytest -q`; `uv run mkdocs build --strict` | no diff/check errors, strict docs, type/lint/test pass, wheel import, performance budgets; `artifacts/m14/l24-rc.json` | build isolated; no secrets in artifacts; delete dist/site/temp | planned; external Actions account and unresolved evidence thresholds; release owner |

### Acceptance contract cho mỗi dòng

Executor phải điền đủ các cột trên vào JSON/YAML artifact, không ghi “TBD” cho
model/revision khi lane đã được chạy. Một artifact tối thiểu chứa `capability`,
`public_symbols`, `source`, `tests`, `evidence_tier`, `model`, `revision`,
`dataset`, `backend`, `license_access`, `environment`, `command`, `seed`,
`acceptance`, `resource_peak`, `network`, `credentials`, `cleanup`, `status`,
`blocker`, `waiver_owner`, `git_sha`, `tool_versions`, `sha256`. Failure artifact
được giữ nguyên và liên kết từ ledger; không sửa metric để che lịch sử.

## Coverage không mồ côi

### 205 top-level exports

Các nhóm sau là phân vùng **đúng theo thứ tự `latent_anything.__all__`**; mỗi
Tên xuất hiện đúng một lần, tổng 205. `E01–E08` giữ nguyên thứ tự beta; `E09`
là các tên canonical RFC0001 được thêm theo kiểu additive. Snapshot API-freeze
giữ projection canonical ổn định gồm 202 entry baseline (thay thế các tên
legacy theo RFC0001), trong khi 205 là runtime surface hiện tại. Các nhóm cần snapshot
signature/import trong L23; các lane chức năng tương ứng được chỉ ra ở L01–L24.

| Group | Symbols (exact public names) | Lane |
|---|---|---|
| E01 (1–28) | `AnalysisPipeline`, `CEMConfig`, `CEMIteration`, `CEMPlanResult`, `CEMPlanner`, `MPPIConfig`, `MPPIIteration`, `MPPIPlanResult`, `MPPIPlanner`, `MPPIRecedingHorizonResult`, `PipelineContract`, `BatchExecutor`, `CacheKey`, `CacheStats`, `DiskCacheError`, `DiskCacheStats`, `ClusterStabilityReport`, `ConceptDataset`, `ConceptDirectionResult`, `ControlBaselines`, `CovarianceConfig`, `CovarianceState`, `CrossSeedReport`, `FeatureAtlas`, `FeatureAtlasEntry`, `FeatureCrossCheck`, `FeatureRanking`, `GLOBAL_REGISTRY` | L01,L05,L06,L16,L22 |
| E02 (29–56) | `InMemoryCache`, `IntegratedGradients`, `IntegratedGradientsConfig`, `IntegratedGradientsResult`, `JEPAWorldModelAdapter`, `JEPAWorldModelConfig`, `JEPALatentHealth`, `JEPAEvaluationReport`, `JEPAPrediction`, `JEPAPredictionMetrics`, `JEPARolloutMetrics`, `KMeans`, `KMeansConfig`, `KMeansResult`, `LatentSpace`, `LatentValue`, `LinearProbe`, `LinearProbeConfig`, `LinearProbeResult`, `MLPProbe`, `MLPProbeConfig`, `MLPProbeResult`, `ManipulationPipeline`, `ManipulationPipelineSpec`, `CEMPlannerSpec`, `MPPIPlannerSpec`, `Method`, `ObjectSpec`, `OrthonormalSubspace` | L01,L03,L04,L07,L12,L16,L22 |
| E03 (57–84) | `PipelineResult`, `PipelineSpec`, `RolloutPipeline`, `RolloutPipelineSpec`, `RolloutResult`, `RewardValueEvaluationSpec`, `PoseConfig`, `PoseMetadata`, `PoseTrajectory`, `ProbeComparison`, `SubspaceProjection`, `SubspaceProjectionConfig`, `TCAV`, `TCAVConfig`, `TCAVResult`, `TCAVScore`, `TransformerLogitTarget`, `ProfileEvent`, `Registry`, `RegistryEntry`, `RuntimeProfile`, `RuntimeProfiler`, `SQLiteDiskCache`, `make_disk_cache_key`, `ArtifactStore`, `ArtifactStoreError`, `StoredArtifact`, `PortableLimits` | L01,L02,L04,L07,L11,L16,L22,L23 |
| E04 (85–112) | `PortableNodeError`, `encode_portable`, `decode_portable`, `PortableEnvelope`, `PortableResultError`, `encode_result_envelope`, `decode_result_envelope`, `ArtifactRef`, `DuplicateRunError`, `FileSystemRunRecorder`, `RunComparisonReport`, `RunRecord`, `build_comparison_report`, `compute_run_identity`, `migrate_run_record`, `SAEConfig`, `SAEEvaluationResult`, `SAEFeatureEvaluation`, `SAEFeatureMetrics`, `SAEStabilityResult`, `SE3`, `SO3`, `SensitivityReport`, `Trajectory`, `DeterministicLatentTransition`, `GaussianPrediction`, `OneStepMetrics`, `RolloutMetrics` | L02,L06,L15,L22,L23 |
| E05 (113–140) | `StochasticGaussianLatentTransition`, `StochasticOneStepMetrics`, `StochasticRollout`, `StochasticRolloutMetrics`, `LatentTransition`, `TokenPrediction`, `TokenPredictionMetrics`, `TokenRolloutMetrics`, `TokenizedEvaluationReport`, `TokenizedWorldModel`, `TokenizedWorldModelConfig`, `HoldoutEvaluation`, `LinearRewardScorer`, `MonteCarloValueEstimator`, `RewardValueDiagnostics`, `RewardValueEvaluationResult`, `RewardValueEvaluator`, `TrajectoryScoreComparison`, `ValueCalibration`, `compare_real_imagined_scores`, `compute_discounted_returns`, `compute_mppi_weights`, `RSSMLatentTransition`, `RSSMTransitionConfig`, `RSSMPrediction`, `RSSMRollout`, `RSSMOneStepMetrics`, `RSSMRolloutMetrics` | L14,L15,L16 |
| E06 (141–169) | `BoundaryMetrics`, `ChangePointResult`, `Segment`, `SegmentationConfig`, `SmoothedTrajectory`, `SmoothingConfig`, `build_feature_atlas`, `build_from_config`, `build_from_dict`, `build_manipulation_pipeline_from_config`, `build_cem_planner_from_config`, `build_mppi_planner_from_config`, `build_pipeline_from_config`, `build_reward_value_evaluator_from_config`, `build_rollout_pipeline_from_config`, `GMMConfig`, `GaussianMixtureDensity`, `DensityResult`, `DensityMetrics`, `DensityEvaluationReport`, `DensityStabilityReport`, `DTWConfig`, `DTWCostSummary`, `DTWResult`, `DensityGeodesic`, `GeodesicConfig`, `GeodesicPath`, `PathOptimizationStatus` | L02,L05,L06,L07,L16,L23 |
| E07 (170–188) | `density_cross_seed_evaluation`, `mahalanobis_baseline`, `fit_covariance_state`, `compute_dtw`, `indexwise_distance`, `detect_change_points`, `evaluate_boundaries`, `smooth_trajectory`, `smoothing_distortion`, `check_clustering_geometry`, `cluster_stability_analysis`, `compare_probes`, `compare_with_labels`, `compute_integrated_gradients`, `compute_tcav`, `coordinate_identity`, `assert_arithmetic_compatible`, `cross_check_feature`, `cross_seed_evaluation` | L02,L03,L04,L05,L06,L07 |
| E08 (189–202) | `cross_seed_sae_stability`, `evaluate_layers`, `evaluate_sae_features`, `evaluate_sensitivity`, `intervention_agreement`, `learn_linear_separator_direction`, `learn_mean_diff_direction`, `list_entries`, `load_feature_atlas`, `lookup_entry`, `nonlinear_memorization_test`, `rank_feature_examples`, `register_entry`, `save_feature_atlas` | L03,L04,L06,L07,L23 |
| E09 (203–205) | `AnalysisMethod`, `Intervention`, `InterventionPipeline` | L01,L02,L23; RFC0001 |

### Registry, plugin groups, optional profiles

| Coverage set | Exact entries | Verification |
|---|---|---|
| 32 built-in registry entries | Adapter: `conv_vae`, `gaussian_3d_renderer`, `gaussian_renderer`, `hidden_state`, `random_projection`, `vae`, `vq_vae`, `jepa_world_model`; Analysis: `pca`, `sae`, `umap`, `kmeans`, `gaussian_mixture_density`, `linear_probe`, `mlp_probe`, `tcav`, `integrated_gradients`, `sae_evaluation`; Intervention: `activation_patch`, `density_geodesic`, `lerp`, `steering`, `subspace_projection`; Runtime: `deterministic_transition`, `stochastic_transition`, `rssm_transition`, `tokenized_world_model`, `jepa_transition`, `linear_reward_scorer`, `monte_carlo_value_estimator`, `cem_planner`, `mppi_planner` | L23 snapshot, registry tests, each entry mapped to L01–L22; count must remain 32 |
| 5 plugin groups | `latent_anything.adapter`, `latent_anything.analysis`, `latent_anything.intervention`, `latent_anything.transition`, `latent_anything.planner` | L23 install/discovery/negative isolation |
| 12 optional profiles | `docs`, `diffusers`, `transformers`, `diffusers-full`, `3d`, `lerobot`, `lerobot-diffusion`, `lerobot-smolvla`, `viz`, `tracking-mlflow`, `tracking-wandb`, `tracking` | L09–L11,L17–L24 clean install matrix; no implicit import/download |

### CLI, schemas, negative/security và composition

| Surface | Exact contract and real validation |
|---|---|
| CLI | `capture-points`, `inspect-policy`, `inspect-dataset`, `replay-run`, `compare-runs`; run `--help`, malformed input, path traversal, interrupted replay, and one real artifact per applicable command. |
| Serialization | `portable-node-v1`, `result-envelope-v1` (including v0 migration), `artifact-envelope-v1`, schema-v1 run records and recorder state; round trip across clean environments and reject unknown/oversized/malformed payloads. |
| Sync/async | Execute every sync public path and its async/streaming counterpart where exposed; cancel after first/partial batch and prove bounded memory, ordering, state carry and no leaked task. |
| Composition | At minimum ConvVAE→probe→intervention, GPT-2→SAE→steering→run record, VQ→tokenized rollout→CEM/MPPI, LeRobot dataset→ACT/Diffusion/SmolVLA→recorder, and Diffusers VAE→portable artifact. Each gets a separate artifact and source-to-test trace. |
| Security/failure | Optional dependency absence, unknown registry/plugin, invalid config, duplicate run, symlink/path escape, malicious archive/member, secret redaction, network-offline assertion, cancellation, corrupted cache, schema downgrade and license metadata omission must fail closed with typed errors. |

## Explicit non-API, backlog và blockers

SAM, OpenCLIP, timm, Torchvision model adapters, Open3D, trimesh và named 3DGS
checkpoint hiện **không phải stable public API**. Không được đưa chúng vào số
liệu D3 hay release claim. Chúng là backlog (nếu người dùng phê duyệt phạm vi)
hoặc blocker của lane liên quan; L17 bị blocked cho đến khi có checkpoint có
tên, revision, license/access và artifact. Các beta artifact cũ còn nói không
có probes/planning/world-model/3D integrations; Sprint 78 phải reconcile chúng
với code hiện tại, không giữ tuyên bố stale. SmolVLA hiện có artifact D3 trong
một tài liệu nhưng ledger override là D2; ledger là nguồn sự thật cho đến khi
chạy lại và ký artifact.

## Trình tự M14 và waivers

1. Sprint 78: freeze inventory/snapshots trước, hoàn thành SRP audit toàn bộ
   `src/` (đặc biệt files lớn), refactor chỉ khi có parity snapshot; dọn xung
   đột docs; chốt migration, schema, plugin, CLI và API ADR.
2. Sprint 79: clean-environment matrix → local real lanes → remote CUDA lanes →
   performance/security/license/evidence ledger; mọi failure giữ artifact.
   Waiver chỉ do owner ký, ghi lý do, phạm vi, hạn hết hạn và không được che
   core gap.
3. Sprint 80: RC review, wheel/sdist/install, release docs và publish. Nếu còn
   bất kỳ blocker, Actions account blocker, missing credential/model license,
   threshold fail hoặc stale claim thì **stop before release**.

Artifact/command names trong bảng là normative; kết quả thực tế phải ghi SHA
code và manifest. Checklist này không tự nâng trạng thái Milestone 14.

## Trạng thái checkpoint API-freeze Sprint 78

Snapshot API và tài liệu migration hiện ghi nhận **205 export runtime** và
projection canonical ổn định **202 entry**; [MIGRATION](MIGRATION.md),
[API_REFERENCE](API_REFERENCE.md), và artifact của task 78.40 là các điểm vào
cho người dùng, còn snapshot/ledger vẫn là nguồn máy móc chuẩn. Checkpoint này
không cho phép xóa alias, bump version, tag, publish, hay tuyên bố release
readiness. Metadata vẫn là `0.1.0b1`; `0.9.0` chỉ là epoch pre-stable dự kiến và
các blocker evidence/workflow phải được giải quyết trước khi release.
