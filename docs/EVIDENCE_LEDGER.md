# Evidence Ledger

`docs/evidence-ledger.json` is the machine-readable source of truth for the
1.0 evidence contract. `scripts/validate_evidence_ledger.py` derives one
inventory record from every bold checklist item in `docs/THEORY.md`; it does
not rewrite either file. A topic's canonical ID is
`THY-<tier>-<ASCII-normalized-topic-title>`. Renaming a theory item is an
intentional capability migration: update its ledger key in the same change.

## Evidence levels

| Level | Requirement |
| --- | --- |
| D0 | Theory/research documentation only. It never counts toward stable coverage. |
| D1 | Versioned implementation plus a focused test. |
| D2 | D1 plus a non-trivial benchmark, quantitative acceptance criterion, and reproducible configuration. |
| D3 | D2 plus a reproducible artifact from a real trained or pretrained model. |

Each evidence item is a typed record with a `role` (`source`, `test`,
`benchmark`, `config`, or `artifact`) and a local `path`. The validator checks
that every D1+ item links to versioned local evidence; D2 requires source,
test, benchmark, and config records, while D3 also requires an artifact. D1
requires source and test records.
It deliberately does not download optional models or resolve optional extras.

## Classification and denominator

Every THEORY topic is exactly one of:

- `implementation-applicable`: a framework capability that can eventually be
  implemented or benchmarked. It belongs to the denominator.
- `benchmark-only`: an evaluation claim/control rather than a standalone
  runtime feature. It belongs to the denominator, but can only qualify at D2
  or D3 through a benchmark.
- `contextual-background`: theory, historical model survey, or prerequisite
  knowledge that informs decisions without becoming a product capability. It
  is explicitly excluded only as an `{id: rationale}` ledger record.

The two release percentages are exact:

- **Core coverage**: qualifying (`D2` or `D3`) implementation-applicable and
  benchmark-only topics in T01–T09 (including T03B), divided by all topics in
  those tiers with either classification. Required: **at least 95%**.
- **Overall coverage**: qualifying implementation-applicable and benchmark-only
  topics in every tier, divided by all such topics. Required: **at least 90%**.

The current beta inventory is intentionally below these gates. D1 is useful
evidence but is not a stable-release claim.

## Contract-change evidence

Non-theory API contracts are linked here rather than misclassified as theory
capabilities. [RFC 0001](rfcs/0001-semantic-api-vocabulary.md) and
`tests/test_api_surface.py` define the Sprint 28 semantic-vocabulary baseline;
Sprint 31 will attach migration evidence to this section.

Sprint 34 carryover closure adds a held-out meaningful-integration benchmark
for the compact ConvVAE. `scripts/conv_vae_heldout_benchmark.py` uses a
deterministic 80/20 sklearn-digits split, fits the adapter and composition
methods on training data only, and evaluates held-out reconstruction against
an all-zero baseline plus a stronger train-pixel-mean diagnostic. The typed
JSON/config artifacts and focused regression tests record the dataset revision,
BSD-3-Clause license, split digests, quantitative thresholds, CPU runtime, and
the limitation that this is a compact trained CPU model rather than a
pretrained generative checkpoint. This closes only the Sprint 34 carryover
gate.

Sprint 35 fidelity closure adds the revision-pinned pretrained Diffusers
`AutoencoderKL` evidence lane. The cached public MIT checkpoint is
`stabilityai/sd-vae-ft-mse` at revision
`31f26fdeee1355a5c34592e401dd41e45d25a493`; the safetensors file is 334,643,276
bytes with SHA-256
`a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`. Under
Diffusers 0.39.0, safetensors 0.8.0, and Torch 2.10.0+cpu, direct
`AutoencoderKL` and `DiffusersAutoencoderKLAdapter` execution matched exactly
(zero maximum absolute error) for deterministic mean and independently seeded
posterior-sample latent/decode outputs. The lane used local-only safetensors,
no remote code, and recorded zero network attempts, 2.7973485 seconds, and a
1,446,883,328-byte peak RSS within the declared CPU bounds. This closes the
Sprint 35 fidelity component; the separate interpolation record below closes
the remaining carryover gate.

Sprint 35 interpolation closure adds the corresponding cached-checkpoint D2
artifact. The deterministic sklearn-digits endpoints are ordered digit 0 to
digit 1 with seven coefficients from 0.0 to 1.0. Latent shape is `(7, 4, 4,
4)` and decoded shape is `(7, 3, 32, 32)`; endpoint latent/decoded movement is
8.3936653/51.7986336 and minimum adjacent movement is 1.3989439/9.1108503,
above the declared 1e-3 endpoint and 1e-4 adjacent thresholds. Endpoint
reconstruction error is zero. The JSON/PNG pair has deterministic content
SHA-256 `b3887f0e4b13e5942011275dac77da3d7f92bfb8203d67a46f119d61abeaf0dd`
and PNG SHA-256
`65245a9c171106dc33ce63c3687738721985917aafa7fec9f4751355e3cbfe40`, with
2100×360 RGBA dimensions. It records zero network attempts, 3.5648021 seconds,
and 1,150,750,720-byte peak RSS. Together with Sprint 34 and fidelity evidence,
this closes Milestone 8's declared bounded evidence scope; it does not claim
perceptual quality, CUDA, or a complete diffusion pipeline.

Sprint 57 adds the LeRobot dataset bridge contract. Its evidence is the
bridge source (`src/latent_anything/integrations/lerobot_dataset.py`), focused
offline alignment tests (`tests/test_lerobot_dataset_bridge.py`), the pinned
public metadata inspection (`scripts/lerobot_dataset_inspection.py` and
`artifacts/lerobot_dataset_inspection.json`), and the integration constraints
in `docs/LEROBOT_INTEGRATION.md`.

Sprint 58 extends that contract with observational ACT representation capture.
Its evidence is the ACT adapter and pinned factory loader
(`src/latent_anything/integrations/lerobot_act.py`), focused lifecycle and
lazy-import tests (`tests/test_lerobot_act.py`), the deterministic projection,
probe, trajectory, and control benchmark (`scripts/act_policy_representation_benchmark.py`
and `artifacts/act_policy_representation_benchmark.json`), and the opt-in
public checkpoint smoke marked `network`/`large_download`. This evidence does
not promote a causal-intervention claim; Sprint 61 owns environment-level
effects.

Sprint 59 extends that contract with observational LeRobot Diffusion Policy
capture. Its evidence is the Diffusion adapter and official factory loader
(`src/latent_anything/integrations/lerobot_diffusion.py`), focused offline
axis/queue/parity/analysis tests (`tests/test_lerobot_diffusion.py`), the
deterministic benchmark (`scripts/diffusion_policy_representation_benchmark.py`
and `artifacts/diffusion_policy_representation_benchmark.json`), and the
revision-pinned `lerobot-diffusion` optional profile. The marked public
checkpoint smoke is separate and does not promote causal policy or environment
effects; Sprint 61 owns those claims.

Sprint 60 extends that contract with SmolVLA capture and one bounded
action-expert intervention. Its evidence is the SmolVLA adapter and official
factory loader (`src/latent_anything/integrations/lerobot_smolvla.py`), the
revision-pinned `lerobot-smolvla` optional profile, focused offline
capture/parity/queue/intervention/measurement tests
(`tests/test_lerobot_smolvla.py`), the deterministic intervention benchmark
(`scripts/smolvla_policy_representation_benchmark.py` and
`artifacts/smolvla_policy_representation_benchmark.json`), and a marked CUDA
checkpoint lane for the pinned public pair. The intervention claim is bounded
and observational (strength-controlled additive edits on the action-expert
representation with bit-exact identity at strength zero); it does not promote
environment-level causal effects, which remain Sprint 61 scope.

Sprint 61 records the causal-intervention capability
(`THY-T05-CAUSAL-INTERVENTION-VS-OBSERVATIONAL-STUDY`) at **D2** pending a
corrected pinned CUDA rerun. Its evidence is the four-condition simulation
benchmark protocol and acceptance gate
(`src/latent_anything/integrations/lerobot_benchmark.py`), the offline fixture
suite plus the marked CUDA statistical lane (`tests/test_lerobot_benchmark.py`),
the reproducible evidence script (`scripts/smolvla_simulation_benchmark.py`),
and the frozen `SimulationBenchmarkConfig`. The prior real-model artifact
(`artifacts/smolvla_simulation_benchmark.json`) is retained as historical,
unverified context only and does not qualify as D3 evidence. ACT and Diffusion
have no intervention surface, so their claims remain observational.

Sprint 62 adds the local reproducibility contract without promoting a new
theory topic: the versioned record and atomic recorder
(`src/latent_anything/run_record.py`), LeRobot boundary helpers
(`src/latent_anything/integrations/lerobot_recording.py`), focused contract and
CLI tests (`tests/test_run_record.py` and `tests/test_cli.py`), the inspection
and comparison commands (`src/latent_anything/cli.py` and
`scripts/lerobot_run_comparison.py`), and the pinned ACT-vs-Diffusion
comparison (`artifacts/lerobot_policy_comparison.json`). The Rule-of-Three
decision freezes only the local concrete contract; external tracking backends
remain deferred to Sprint 76.

Sprint 63 promotes the first deterministic latent-transition and rollout
evidence to D2 for `THY-T06-LATENT-TRANSITION-MODEL`,
`THY-T06-ROLLOUT-LATENT-IMAGINATION`, and
`THY-T07-LATENT-IMAGINATION-HORIZON`. The concrete source is
`src/latent_anything/transition.py`, with focused offline tests in
`tests/test_latent_anything/test_transition.py` and the seeded held-out
benchmark/config/artifacts from `scripts/deterministic_transition_benchmark.py`.
This is a flat Euclidean affine-residual baseline only; stochastic and
recurrent transition variants are separately evidenced by Sprints 64–65.

Sprint 66 adds the non-theory pipeline composition contract. Its evidence is
the focused pipeline modules (`src/latent_anything/analysis_pipeline.py`,
`src/latent_anything/manipulation_pipeline.py`, and
`src/latent_anything/rollout_pipeline.py`), the shared metadata contract
(`src/latent_anything/pipeline_contract.py`), focused behavior/config/cache/
async tests (`tests/test_latent_anything/test_rollout_pipeline.py` plus the
existing pipeline/runtime regression suites), and the migration/ownership
documentation in `docs/PIPELINES.md`. This does not promote a new model or
planning claim; it makes the existing latent-transition rollout evidence
available through Pipeline #3.

Sprint 67 promotes the first reward/value evaluation evidence to D2 for
`THY-T07-REWARD-MODEL-TRONG-LATENT` and
`THY-T07-VALUE-FUNCTION-TRONG-LATENT`. The focused NumPy implementation
(`src/latent_anything/reward_value.py`) provides masked discounted returns,
terminal/padding semantics, linear reward scoring, finite-horizon Monte-Carlo
value estimation, held-out calibration, Bellman residuals, and
real-versus-imagined score bias. Its integration evidence is the configured
rollout evaluator and run-record artifact path, the deterministic tests in
`tests/test_reward_value.py`, and the controlled benchmark/config/artifact
from `scripts/reward_value_benchmark.py`. This remains synthetic D2 evidence;
it does not claim a real pretrained world-model or CUDA result.

Sprint 68 promotes the first CEM planning evidence to D2 for
`THY-T07-CROSS-ENTROPY-METHOD-CEM`. `src/latent_anything/cem.py` implements bounded
continuous-action sampling, elite refitting, smoothing, seeded execution,
convergence summaries, and runtime profiling. The planner composes candidates
through `RolloutPipeline` and `RewardValueEvaluator`; focused tests cover an
analytic objective, invalid bounds/population/horizon settings, config and
registry construction, and run-record persistence. The reproducible CPU
benchmark in `scripts/cem_planning_benchmark.py` compares fixed-zero,
random-shooting, and CEM on model-predicted and environment-realized return.
The action-scale mismatch intentionally exposes a positive model-bias gap, so
this remains synthetic D2 planning evidence rather than real-model evidence.

Sprint 69 promotes the first MPPI planning evidence to D2 for
`THY-T07-MPPI-MODEL-PREDICTIVE-PATH-INTEGRAL`. `src/latent_anything/mppi.py`
implements bounded noise sampling, stable temperature weighting, seeded
nominal updates, receding-horizon execution, soft-weight diagnostics, and
runtime profiling. The planner reuses `RolloutPipeline` and
`RewardValueEvaluator`; focused tests cover weighting, numerical stability,
zero-noise behavior, bounds, seeding, rollout integration, configuration,
registry construction, and run-record artifacts. The reproducible CPU
benchmark in `scripts/mppi_planning_benchmark.py` compares fixed-zero, random
shooting, CEM, and MPPI on the same task, reporting return, action smoothness,
sample count, latency, and robustness to a deliberate transition error. This
remains synthetic D2 evidence and does not claim real-model or CUDA evidence.

Sprint 71 promotes the compact decoder-free JEPA/LeWM-style prediction lane
to D2 for `THY-T08-JEPA-JOINT-EMBEDDING-PREDICTIVE-ARCHITECTURE-LECUN-2022`
and to D1 for `THY-X01-LEWM-LEWORLDMODEL-2026`. The source is
`src/latent_anything/adapters/jepa.py`, with stop-gradient, no-decoder,
pipeline, health, baseline, checkpoint, and run-record coverage in
`tests/test_latent_anything/test_jepa.py`. The reproducible CPU benchmark is
`scripts/jepa_world_model_benchmark.py` with the pinned config and artifact
under `artifacts/jepa_world_model_evidence*`. The artifact reports strong
held-out one-step improvement over a collapsed predictor, but also records
anisotropic covariance and open-loop drift; this is not real LeWM or CUDA
evidence. The public I-JEPA checkpoint smoke is separately marked
`network`/`large_download`.

Sprint 72 adds the compact tokenized-world-model lane at D2 for
`THY-T09-TOKENIZED-WORLD-MODEL`. The source is
`src/latent_anything/tokenized_world_model.py`, with integer-token adapter and
mean-transition composition, autoregressive sampling, masking/padding/version
validation, and teacher-forced/free-running tests in
`tests/test_latent_anything/test_tokenized_world_model.py`. The reproducible
offline benchmark is `scripts/tokenized_world_model_benchmark.py` with
configuration and rollout evidence under
`artifacts/tokenized_world_model_evidence*`. The artifact includes decoder and
task-proxy metrics plus failure horizons. The regenerated fitted tokenizer
passes the non-trivial-token-usage gate, while the learned dynamics still show
early greedy free-running error; this is meaningful compact synthetic evidence,
not real-checkpoint or CUDA evidence.

Sprint 79 L02 partially closes the geometry gap plan through the reproducible
held-out sklearn-digits artifact
(`artifacts/m14/l02-geometry.json`, self-digest
`97d26f5fb1d12dc00658ff9cfec12a91b080bb4a1bb3cd96e7bc2ed70f9e5a58`). Four
independent core rows are promoted to D2:
`THY-T03-SLERP-SPHERICAL-LINEAR-INTERPOLATION`,
`THY-T04-LERP-LINEAR-INTERPOLATION`,
`THY-T03-RIEMANNIAN-GEOMETRY-CO-BAN`, and `THY-T04-SLERP`. The
`THY-T01-MANIFOLD-HYPOTHESIS` row remains D1 because held-out ranking AUC was
`0.4560546875` against the `0.55` threshold, and
`THY-T06-TRAJECTORY-SIMILARITY-METRICS` remains D0 because its
self-to-indexwise ratio was `17.015624999997637` against the `0.95` threshold.
The lane records model-induced latent sequences from held-out digits, not
physical trajectories, and makes no Fréchet claim. Coverage is now 30/63 core
(`47.6%`) and 30/65 overall (`46.2%`), leaving 35 D0/D1 rows (33 core and 2
non-core); 30 additional core and 29 additional overall qualifiers are needed
for the release gates.

Sprint 79 L03 promotes exactly three independent core rows to D2 through the
real forward-only pinned GPT-2 lane:
`THY-T03-LINEAR-STRUCTURE-TRONG-LATENT`, `THY-T05-LINEAR-PROBING`, and
`THY-T05-NONLINEAR-PROBING`. The concrete integration is
`TransformerLMIntegration`; this does not claim a separate GPT-2
`ModelAdapter` or promote L11. The accepted artifact
(`artifacts/m14/l03-analysis.json`) has self-digest
`60bda13a4bbf68bbb6c9308cc813913fa653c37fba368fe1e4ea7a1f898ce06b`, and the
final run record has digest
`0bcaf14ef465f2ef5c5c909237d1f573596a77fa2ca51d042db74248cf4ca03a` under
plan `fe2a85a1691c0fe362fc5f39434898d6ea8968aeec8450a7bb61ba55fd94cfd5`.
The raw glyph baseline is an expected diagnostic for GPT-2's synthetic ASCII
task. The focused transformer network suite first returned **6 passed / 2
failed** because tuple-return intervention was incompatible with the capture
seam. After the structured-output fix in `16db80f`, an intermediate
verification returned **7 passed / 1 failed** because its indexing oracle
used the wrong native hidden-state position; `9ebecfa` corrected that test
contract without changing runtime layer mapping. The final exact-SHA
strict-CUDA verification passed **8/8**, including native-index-7 intervention
and hook cleanup. The structured hook/output cleanup blocker is resolved by
`16db80f` + `9ebecfa` and the retained transformer-hook attempt-1 and attempt-2
artifacts. The forward-only L03 evidence and D2 promotions are unchanged.
The separate native hidden-state index-12/direct-logit-lens parity follow-up
is complete as an internal semantic correction, with attempt 1's missing
optional `transformers` dependency and attempt 2's exact-SHA
direct-PowerShell-SSH 8/8 CUDA verification preserved in sanitized metadata
and transcript digests; it is not an L11 promotion. Attempts 1–3 of the
canonical L03 capture workflow remain capture-only failures, while attempt 4's
raw transcript is superseded by its sanitized capture-audit artifact.
Current validator coverage is **33/63 core (52.4%)** and **33/65 overall
(50.8%)**, leaving 32 D0/D1 rows (30 core and 2 non-core); 27 additional core
and 26 additional overall qualifiers are needed for the release gates.

Sprint 73 adds the external plugin discovery contract as a non-theory API
capability. Its source evidence is `src/latent_anything/plugin_groups.py`,
`src/latent_anything/plugin_metadata.py`, and
`src/latent_anything/plugin_discovery.py`; focused contract tests are
`tests/test_plugin_groups.py`, `tests/test_plugin_metadata.py`, and
`tests/test_plugin_discovery.py`. The separately installed distribution proof
is `tests/fixtures/sprint73_hello_plugin/` exercised by
`tests/test_plugin_installation.py`; the authoring contract is documented in
`docs/PLUGIN_AUTHOR_GUIDE.md` and `docs/PLUGIN_TEMPLATE.md`. This is a
versioned API/integration proof, not a theory D2/D3 promotion: loading remains
explicit, third-party code is untrusted, and transition/planner groups retain
the existing `runtime` registry kind. The separately installed proof uses
`uv pip install --offline --no-build-isolation` and asserts distribution,
version, entry-point value, and plugin API metadata. Audit-remediation
summaries under `artifacts/task_sprint73_remediation*` record the stable
duplicate tie-break, fail-closed installation, provenance, and missing-marker
checks.

Sprint 74 adds a non-theory portable-artifact and runtime-cache contract. The
source evidence is `src/latent_anything/portable.py`,
`src/latent_anything/portable_results.py`, `src/latent_anything/artifact_store.py`,
`src/latent_anything/runtime/disk_cache.py`, and the run-record integration in
`src/latent_anything/run_record.py`. Focused tests cover value nodes, typed
envelopes, storage safety, cache eviction/corruption/concurrency, and
RunRecord/plugin metadata in `tests/test_portable.py`,
`tests/test_portable_results.py`, `tests/test_artifact_store.py`,
`tests/test_disk_cache.py`, `tests/test_run_record_portable.py`, and
`tests/test_sprint74_roundtrip.py`. Offline CPU reproduction is
`scripts/sprint74_portable_roundtrip.py`; declared size/latency measurement is
`scripts/sprint74_artifact_benchmark.py`. Task summaries are
`artifacts/task_sprint74_task01_portable_nodes_summary.md` through
`artifacts/task_sprint74_task09_closure_summary.md`; Task09 is the governance
closure record rather than a separate capability claim. This is an implementation
and synthetic CPU validation contract, not a D3 real-model or CUDA claim.

Sprint 75 adds a non-theory bounded streaming runtime contract over the
existing `RolloutPipeline` and `LatentTransition.step` seams. The source and
tests are `src/latent_anything/rollout_pipeline.py`,
`tests/test_latent_anything/test_rollout_pipeline.py`, and
`tests/test_sprint75_streaming.py`. The offline CPU reproduction is
`scripts/sprint75_streaming_benchmark.py`; task records are
`artifacts/task_sprint75_task01_streaming_rollout_summary.md` through
`artifacts/task_sprint75_task08_closure_summary.md`, plus the four post-closure
remediation records under `artifacts/task_sprint75_remediation*.md`. Evidence proves
ordered chunk carry, one-chunk backpressure, async cancellation/source
cleanup, and eager-equivalent output for a 4096-step synthetic rollout. It
requires exact NumPy action chunks and an explicit/reset transition-state
contract before execution. It deliberately excludes masks/padding and seeded
sampling, bypasses cache/run-record persistence to avoid retaining partial or
stateful outputs, and retains only bounded per-stream profiling metadata. The
benchmark reports NumPy chunk bytes plus supplemental `tracemalloc`; it does
not claim native RSS, LeRobot, real-model, or CUDA throughput and does not
introduce a generic streaming Protocol.

Sprint 76 adds optional external tracking adapters behind the local
experiment-recorder contract. Its source evidence is
`src/latent_anything/experiment_recorder.py` and the MLflow/W&B adapters under
`src/latent_anything/integrations/`; focused fake-provider parity and
isolation tests are `tests/test_experiment_recorder.py`,
`tests/test_mlflow_recorder.py`, `tests/test_wandb_recorder.py`,
`tests/test_tracking_parity.py`, and `tests/test_integrations.py`. Opt-in
local MLflow file-store and W&B offline tests skip when extras are absent and
never require cloud credentials. The contract covers bounded canonical JSON
params/tags, stable identity, ordered finite metrics, checksummed artifacts,
resume identity, lifecycle, and parent linkage. W&B parent/child is represented
by a group and explicit parent tag because W&B does not expose the same
nested-run primitive as MLflow. This is an offline integration contract, not a
claim about remote tracking, hosted UI, or team workflows. The parity fixture
is a compact deterministic affine world-model rollout, not a LeRobot or
real-checkpoint benchmark.

The Sprint 76 post-closure remediation adds focused evidence for local resume
identity and exact provider-ID continuity matrices, canonical local MLflow roots,
platform-safe artifact names,
bounded/privacy-safe inputs and reads, provider-object isolation, provider
state commit atomicity, and real local/offline provider round trips in the
remediation artifacts under `artifacts/task_sprint76_remediation*.md`. The
MLflow lane independently downloads and hashes its file-store artifact. W&B
offline has no portable SDK artifact-read API, so its evidence uses the
adapter-owned validated mirror and records that limitation explicitly. Both
real lanes deny remote network requests and exercise parent/cleanup behavior;
MLflow exercises provider resume, while W&B offline rejects resume when the
provider has not persisted adapter provenance or returned the requested run ID
rather than claiming false continuation. The final test seam injects SDKs only
through underscore-prefixed internal parameters; no SDK object is part of the
public recorder contract.

Sprint 77 Phase A adds a reproducible, offline CPU performance contract rather
than a Rust implementation. `scripts/sprint77_phase_a_benchmark.py` measures
fixed geometry/DTW/geodesic, activation, rollout/planning, Arrow/artifact/cache,
streaming, recorder/plugin, and offline LeRobot-boundary workloads with robust
latency, memory, environment, and correctness-digest fields. The profile and
before/after comparison are `scripts/sprint77_phase_a_profile.py` and
`scripts/sprint77_phase_a_compare.py`, with artifacts under
`artifacts/sprint77_phase_a_*.json`. A bounded NumPy Euclidean DTW cost path
reduces the declared fixture median from 38,297.3 to 27,478.2 microseconds
(-28.25%) with unchanged digest. Budgets are advisory and environment-scoped;
semantic correctness/no-network/bounded-input checks remain hard gates. The
owner-approved Phase B decision defers Rust/PyO3 for pre-stable work rather
than permanently prohibiting it; its rationale and reconsideration conditions
are in `.agents/memory/decisions.md` and
`artifacts/task_sprint77_phase_b_task01_rust_deferral_summary.md`. The Phase-A
LeRobot case remains only an offline captured-latent NumPy boundary, native RSS
is unavailable on the recorded Windows run, and no multi-environment or real
policy throughput claim is promoted. Sprint 77 Phase A/B closure validation
and cumulative audit are complete for the supported scope; carryover gates and
Milestone 14 are not started.
The Phase-A task summaries and Phase-B closure/audit summaries are linked in
the typed ledger for atomic traceability.

## Quality gates for a D2/D3 promotion

- Core unit tests: all changed core behavior has deterministic focused tests;
  the full test suite must pass.
- Integration tests: a D2/D3 claim has an offline, version-pinned path; network
  acquisition is a separate marked smoke test.
- Documentation: API, constraints, exact reproduction command, and failure
  cases are linked from the capability entry.
- Explanation validity: headline explanation methods report fidelity, stability,
  a negative/selectivity control, and a causal intervention metric when the
  claim is causal.
- Compatibility: public-name/config snapshots and legacy migration tests pass
  whenever a promoted capability changes a beta surface.

## Validator contract

Run `uv run python scripts/validate_evidence_ledger.py`. It rejects missing,
duplicated, or stale IDs; invalid status/classification; absent D1+ evidence;
malformed typed records; missing level-specific roles; and evidence paths that
do not exist. CI runs the same read-only command before the Python quality gate.
