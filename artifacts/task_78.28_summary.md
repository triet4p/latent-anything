# Sprint 78 Atomic Task 78.28 — API-Freeze Inventory and Sprint 28/31 Naming Audit

## Verdict

**PASS for the read-only inventory; API freeze remains BLOCKED.** The current
beta surface is measurable and internally consistent for the existing snapshot,
registry, plugin, profile, CLI, and serialization contracts. The Sprint 28
canonical-symbol migration is not complete, so no alias removal or API-freeze
approval is recommended by this audit.

No source, test, documentation, package metadata, dependency, network, model,
CUDA, commit, or release change was made by this task.

## Deterministic inventory

| Surface | Count | Deterministic evidence |
|---|---:|---|
| `latent_anything.__all__` | 202 | order SHA-256 `4a101a3a30687437958c6d504f9741f962813daf46ee4322c3f122bd1bf3f8e6`; identity+signature SHA-256 `2c5429b14fc9b86d222f86a2b7c2c40967766c1a685ab12af96af24e88fe7c2c`; zero public `torch` signature leaks |
| Exporting implementation modules | 43 module groups / 202 names | exact module-group signature rows were generated from every top-level export; public symbol identity is preserved by `tests/test_api_surface.py` and M14 E01–E08 |
| Python source modules | 137 files / 84 non-private module paths | path SHA-256 `f49b126da5f938395c29fd488e6787fec109cf58d93173855a7c855712e0fbe9`; public-path SHA-256 `b38f6997f203e9bf3561894a48a5556e709259c80a8b246c9efd4964db161a6` |
| `latent_anything.methods.__all__` | 8 | SHA-256 `b23d65dd4776863b3cfeea24ae3be511b69449f7d95ca5dd6bd1ee18bddf9d0f` |
| `latent_anything.adapters.__all__` | 24 | SHA-256 `040ff165a15a51706a01e220682ea5700f2dc3c73af4b5faa5fc58566ec53ebf` |
| Built-in registry | 32 | row SHA-256 `a461b4ef2c6fbf66ea5c9994002843eebd0b62e62b23bc6a25ceab268c1e5f4f`; adapter 8, analysis 10, intervention 5, runtime 9 |
| Plugin entry-point groups | 5 | `latent_anything.adapter`, `.analysis`, `.intervention`, `.transition`, `.planner`; external contract version `1` |
| Optional profiles | 12 | key/value SHA-256 `f1898cd4e6461a3a7b54b6e17a31d010ba127cf3a8861ae6704a6d6cef8b3cd7` |
| Public Protocols | 7 | SHA-256 `6283368f8e416a03d80eea2e0e818f4261152fd25aab6bdccd815adc0e1269fb`; `PipelineContract`, `Method`, `BMethod`, `LatentTransition`, `DecodableAdapter`, `FlatBatchDecodableAdapter`, `ModelAdapter` |
| Public dataclass schemas | 81 | field/default map SHA-256 `bef6e7f02084ead9a422b2da02b8781b423e638e2375c56b46e3b3d9f61bbda6` |
| Config/spec/limits schemas | 28 | field/default map SHA-256 `fee7899b9ac8be728636ff888562e27146a719a082ca7b74089e8d0c12ee7587` |
| Sync/async pairs | 9 | `AnalysisPipeline.run`, `ManipulationPipeline.run_data/run_trajectory`, `RolloutPipeline.run/stream`, and `BatchExecutor.decode/encode/map_array/transform`, each paired with `_async`; no async-only public function |
| CLI commands | 5 | `capture-points`, `inspect-dataset`, `inspect-policy`, `replay-run`, `compare-runs`; aliases are `list-capture-points` and `replay-run-config` |
| Exception taxonomy | 7 custom exceptions | `ArtifactStoreError`, `RecorderContractError`, `PluginContractError`, `PortableResultError`, `PortableNodeError`, `DuplicateRunError`, `DiskCacheError`; taxonomy SHA-256 `e62b8f9d46707080621e5fd5a0b90417926a564ee49cb1044af75c11d5a202a1` |

The exact 202 names remain listed in the authoritative M14 E01–E08 table at
`docs/M14_REAL_SYSTEM_VALIDATION.md:70-83` and in the strict snapshot
`tests/test_api_surface.py:11-215`. The runtime scan resolved every name and
captured module/signature identity; no public signature contains `torch.Tensor`,
`Tensor`, or `torch.nn`.

### Config field/default inventory

The 28-field/default schemas were inspected from Pydantic `model_fields` and
dataclass fields. The following compact table records the complete config
family and required/default boundary; the digest above binds the exact values.

| Model | Required fields | Non-required defaults |
|---|---|---|
| `CEMConfig` / `CEMPlannerSpec` | `horizon`, `action_dim`, `lower_bounds`, `upper_bounds` | `population_size=256`, `elite_count=None`, `elite_fraction=.1`, `iterations=6`, `smoothing=.1`, `min_std=.001`, `initial_mean=None`, `initial_std=None`, `seed=None` |
| `MPPIConfig` / `MPPIPlannerSpec` | `horizon`, `action_dim`, `lower_bounds`, `upper_bounds` | `population_size=256`, `iterations=6`, `temperature=1.0`, `noise_std=(.5,)`, `initial_mean=None`, `seed=None`; input aliases `lambda`/`lambda_` are accepted for `temperature` |
| `CovarianceConfig` | — | `reg_coef=1e-6`, `min_samples_per_dimension=2.0` |
| `IntegratedGradientsConfig` | — | `target_layer=6`, `activation_position=-1`, `activation_batch_index=0`, `baseline='zero'`, `integration_rule='trapezoid'`, `n_steps=32` |
| `JEPAWorldModelConfig` | — | `hidden_dim=32`, `epochs=120`, `learning_rate=.01`, `ema_momentum=.95`, `variance_loss_weight=.01`, `minimum_latent_std=.05`, `variance_floor=1e-6`, `stability_norm_limit=1e6`, `seed=71`, `device='cpu'` |
| `KMeansConfig` | — | `n_clusters=8`, `init='k-means++'`, `n_init=10`, `max_iter=300`, `random_state=0`, `standardize=True`, `tol=.0001` |
| `LinearProbeConfig` | — | `C=1.0`, `solver='lbfgs'`, `max_iter=1000`, `test_size=.3`, `val_size=.1`, `random_state=0`, `standardize=True`, `class_weight='balanced'`, `fit_intercept=True` |
| `MLPProbeConfig` | `hidden_sizes` | `activation='relu'`, `max_epochs=200`, `early_stopping_patience=10`, `learning_rate=.001`, `weight_decay=.0001`, `batch_size=32`, `test_size=.3`, `val_size=.1`, `random_state=0`, `standardize=True` |
| `ObjectSpec` | `kind`, `name`, `params` | — |
| `ManipulationPipelineSpec` | `method` | `adapter=None` |
| `PipelineSpec` | `adapter`, `method` | — |
| `RolloutPipelineSpec` | `transition` | `cache=False`, `reward_value=None` |
| `RewardValueEvaluationSpec` | `reward_scorer`, `value_estimator` | — |
| `PoseConfig` | — | `parent_frame='world'`, `child_frame='tool'`, `position_unit='m'`, `angle_unit='rad'` |
| `SubspaceProjectionConfig` | — | `n_basis=None` |
| `TCAVConfig` | — | `target_layer=8`, `direction_method='mean_diff'`, `n_bootstrap=50`, `n_random_concepts=50`, `n_seeds=5`, `alpha=.05`, `n_concepts_family=1` |
| `PortableLimits` | — | `max_depth=32`, `max_nodes=10000`, `max_array_bytes=268435456`, `max_total_array_bytes=536870912`, `max_shape_dimension=10000000`, `max_shape_rank=64`, `max_record_batches=128`, `max_array_rows=10000`, `max_input_bytes=805306368`, `max_manifest_bytes=1048576` |
| `SAEConfig` | — | `n_components=64`, `l1_coef=.1`, `learning_rate=.01`, `n_epochs=500`, `random_state=0`, `val_fraction=.2`, `min_val_samples=16`, `dead_frequency_threshold=.0001`, `matching_cosine_threshold=.5` |
| `TokenizedWorldModelConfig` | `action_dim`, `codebook_version` | `hidden_dim=32`, `epochs=40`, `learning_rate=.01`, `seed=0`, `model_revision='compact-tokenized-world-model-v1'` |
| `RSSMTransitionConfig` | — | `hidden_dim=16`, `epochs=160`, `learning_rate=.01`, `variance_floor=1e-6`, `posterior_scale_factor=.5`, `stability_norm_limit=1e6`, `seed=65`, `device='cpu'` |
| `SegmentationConfig` | — | `sensitivity=3.0`, `min_segment_length=3`, `context=3`, `threshold=None` |
| `SmoothingConfig` | — | `window=5`, `weighting='uniform'` |
| `GMMConfig` | — | `n_components=2`, `covariance_type='full'`, `reg_covar=1e-6`, `n_init=1`, `max_iter=200`, `tol=.001`, `random_state=0`, `min_samples_per_dimension=2.0` |
| `DTWConfig` | — | `window=None`, `max_step_distance=None`, `normalization='path_length'`, `max_cells=2000000`, `return_cost_matrix=False` |
| `GeodesicConfig` | — | `n_points=16`, `max_iter=200`, `step_size=.1`, `tol=1e-6`, `density_exponent=1.0` |

### Serialization, checkpoints, and canonical digests

| Contract | Version / migration | Deterministic fixture evidence |
|---|---|---|
| Portable NumPy/Arrow node | `portable-node-v1` | `encode_portable({'array':[1,2,3]})` SHA-256 `0367583dbc38ffb896ebd03f4dba1422fcf79e4b05612c229713be9febdc7707` |
| Typed result/config envelope | `result-envelope-v1`; explicit `result-envelope-v0` migration | fixture SHA-256 `815ea47a07cecc202c5312d4c9ff4de5441c9ce74d4315e647615f483ba050ea` |
| ArtifactStore envelope | `artifact-envelope-v1`; payload and identity SHA-256 | `latent-value`/`portable-bytes` fixture identity `3182645e9775ef6d545a148f951aa224fa85ef51c29f4e4e14142a6327af3b29` |
| Run record | schema version `1`; pre-versioned/Windows artifact-path migration | canonical digest `25a8bc21cf19a67ce9a553d469236e06f19aed0b96760f9b97e3d2ed3b3c4964`; migration digest `fa243e2d7ca35695c6d381940d844312fe8df8476c8d88ef4da5f063406f5083` |
| SQLite disk cache | `disk-cache-v1` | bound key fixture SHA-256 `aeef0c4f0875b282f06c5cef5170589809686cdb108cdd2c211d5d7e37537355` |
| JEPA/RSSM checkpoints | portable NPZ with canonical JSON `metadata_json`; no independent public version string | `save/load` and cross-process/tamper tests preserve metadata, arrays, config, and source-space identity |
| Tokenized model checkpoint binding | no file checkpoint API; `codebook_version` and `model_revision` are behavior identity | mutation/mismatch is rejected before fit, predict, evaluate, and rollout |

## Sprint 28/31 comparison and alias decision matrix

Sprint 28 RFC 0001 (`docs/rfcs/0001-semantic-api-vocabulary.md:8-42`)
selects behavior vocabulary and schedules removal at `0.9.0`; Sprint 31
implements registry/config migration only. Current decisions are:

| Legacy/current spelling | Canonical target | Current evidence | Freeze decision |
|---|---|---|---|
| `method_a` | `analysis` | `src/latent_anything/registry_aliases.py:13-26`; one `DeprecationWarning` at `build_from_config` construction; tests pass | **Retain through pre-freeze migration; remove/reject at 0.9.0 only after repository codemod, migration report, and compatibility snapshot.** |
| `method_b` | `intervention` | same seam and tests | **Retain through pre-freeze migration; remove/reject at 0.9.0 with exact replacement error.** |
| `KIND_METHOD_A` | `KIND_ANALYSIS` | `src/latent_anything/registry.py:71-75`; direct constant alias, no warning | **Retain until the same 0.9.0 release; include import-path snapshot and warning/removal test.** |
| `KIND_METHOD_B` | `KIND_INTERVENTION` | same seam | **Retain until the same 0.9.0 release; include import-path snapshot and warning/removal test.** |
| `Method` | planned `AnalysisMethod` | only `Method` Protocol exists at `src/latent_anything/methods/protocols.py:15`; no `AnalysisMethod` export or alias | **Blocking freeze gap. Add canonical symbol, preserve old import with explicit warning policy, then remove at 0.9.0.** |
| `BMethod` | planned `Intervention` | only `BMethod` Protocol exists at `src/latent_anything/methods/b_protocols.py:28`; no `Intervention` export or alias | **Blocking freeze gap. Add canonical symbol and migration snapshot before removal.** |
| `ManipulationPipeline` | planned `InterventionPipeline` | only `ManipulationPipeline` exists at `src/latent_anything/manipulation_pipeline.py:22` and top-level export | **Blocking freeze gap. Add canonical facade alias and config/import migration before 0.9.0.** |
| `list-capture-points` | `capture-points` | CLI parser alias at `src/latent_anything/cli.py:18-20` | **Non-Sprint-28 compatibility alias; retain until a CLI deprecation policy exists.** |
| `replay-run-config` | `replay-run` | CLI parser alias at `src/latent_anything/cli.py:33` | **Non-Sprint-28 compatibility alias; retain until a CLI deprecation policy exists.** |
| `lambda` / `lambda_` | `MPPIConfig.temperature` | Pydantic `AliasChoices` at `src/latent_anything/mppi.py:61` | **Advisory: document and assign a removal/deprecation window; do not remove during this audit.** |
| `selected_actions`, `RolloutResult.states`, transition `predict/std/nll` aliases | canonical payload fields | Explicit compatibility properties in CEM/MPPI, pipeline models, and transition types | **Advisory: retain; these are behavior/result aliases, not Sprint 28 layer names. Add a general alias ledger before any removal.** |

Legacy run-record schema shapes and Windows artifact paths are deliberate data
migrations, not beta naming aliases; their digests and rejection behavior are
already frozen by `tests/test_run_record.py`.

## Findings and ordered next atomic tasks

| Severity | Finding | Evidence / risk | Required next task |
|---|---|---|---|
| **Blocking** | Canonical `AnalysisMethod`, `Intervention`, and `InterventionPipeline` names are absent while RFC 0001 requires canonical symbols before the 0.9.0 removal window. | `methods/protocols.py:15`, `methods/b_protocols.py:28`, `manipulation_pipeline.py:22`; callers and docs still use roadmap-era names. Removing old names now would break the 202-export snapshot. | 78.29: introduce canonical symbols as module/top-level aliases or facades, define warning behavior, add exact import/signature/module/pickle snapshots, and migrate internal docs/config names without changing behavior. |
| **Blocking** | No complete migration guide/API reference or 0.9.0 compatibility snapshot suite exists. | Sprint78 plan leaves migration guide, API reference, public signature/import/config/plugin/serialized snapshots unchecked; 78.26 identified the same release blocker. | 78.30: publish migration/API reference and machine-generated freeze snapshots after owner approval of canonical symbols and alias deadlines. |
| **Advisory** | Alias policy is split between RFC naming aliases, CLI aliases, Pydantic input aliases, result-property aliases, and private compatibility wrappers. | `rg` found the matrix above; only `method_a/method_b` have a removal deadline and warning test. | 78.31: add an append-only alias/deprecation ledger and tests for all user-facing aliases; leave private compatibility seams undocumented/non-public. |
| **Advisory** | Optional-extra error text has one shared seam but no complete per-profile message snapshot. | `integrations/_optional.py:9-16` produces actionable `uv sync --extra` text; profile-specific import isolation is tested but message compatibility is not globally snapshotted. | 78.32: snapshot missing-backend exception type/message and profile mapping without importing optional backends. |
| **Resolved by audit** | Pending decomposition item for `LatentSpace`, pipelines, and adapters. | 78.25 confirms cohesive domain ownership, focused tests, no new SRP blocker, and defers extraction until a fourth geometry, Pipeline #3, or second renderer/different adapter philosophy. | No pre-freeze extraction; revisit only on the stated Rule-of-Three triggers. |

## Gates and review basis

- Focused freeze tests: **58 passed in 18.33s** (`test_api_surface`, registry
  migration, plugin metadata/discovery, CLI, run-record, portable results,
  artifact store, and disk cache tests).
- Existing unchanged-tree gates reused from 78.27: Ruff check PASS, format
  PASS (`315 files already formatted`), strict Pyright PASS (`0 errors`),
  diff-check PASS with existing LF/CRLF warnings, and prior full suite
  `1,545 passed, 36 skipped, 39 warnings`.
- No network, model, CUDA, dependency, source, or test execution outside the
  focused offline suite was needed.
- API-freeze verdict: **BLOCKED** until canonical symbol migration,
  migration/API reference, complete alias ledger, and compatibility snapshots
  are owner-approved and implemented.

## Graphify

Graphify was refreshed after this artifact and the audit-only Sprint78 plan
updates. Final topology: **10,762 nodes / 20,739 edges / 934 communities**.
