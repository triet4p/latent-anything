# LeRobot Integration

`latent_anything[lerobot]` is an optional boundary for consuming LeRobot's
policy, processor, dataset, environment, evaluation, and plugin surfaces.
The supported upstream window is LeRobot `>=0.6.0,<0.7.0` (currently locked to
`0.6.1`) on Python `>=3.12`, Torch `>=2.7,<2.12`, and NumPy `>=2.0,<2.3`.

Install the boundary with:

```text
uv sync --extra lerobot
```

The extra activates LeRobot's `dataset` and `evaluation` feature extras so
the v3 dataset and evaluation seams are available. Policy-specific features
remain upstream-selected. ACT is part of the base policy surface. Diffusion
is available through the dedicated `latent_anything[lerobot-diffusion]` profile,
which adds LeRobot's upstream `diffusion` extra and keeps its Diffusers
dependency isolated from the legacy `transformers`/`diffusers-full` profiles.
SmolVLA is available through the dedicated
`latent_anything[lerobot-smolvla]` profile, which adds LeRobot's upstream
`smolvla` extra and keeps its Transformers 5.x dependency isolated the same way.

## Lazy boundary

The base package and `latent_anything.integrations.lerobot` module do not
import LeRobot. Use `load_lerobot()` when the package itself is needed, or
`load_lerobot_api()` to obtain the supported raw factories after the
compatibility check:

```python
from latent_anything.integrations.lerobot import load_lerobot_api

api = load_lerobot_api()
policy = api.make_policy
dataset_type = api.dataset_type
```

`check_lerobot_compatibility()` reports missing or incompatible LeRobot,
Torch, NumPy, and Python versions without importing any upstream module. Its
diagnostic points to `uv sync --extra lerobot` for a clean resolver-managed
environment.

## Supported upstream seams

The bridge consumes these LeRobot-owned objects and entry points:

| Capability | Upstream seam | Bridge treatment |
| --- | --- | --- |
| Policy construction | `lerobot.policies.make_policy` | Keep the returned policy object unchanged. |
| Policy preprocessing | `lerobot.policies.make_pre_post_processors` and `lerobot.processor.PolicyProcessorPipeline` | Keep the official pipelines and normalization state. |
| Dataset access | `lerobot.datasets.lerobot_dataset.LeRobotDataset` | Consume v3 episodes/samples; do not decode Parquet or video independently. |
| Environment access | `lerobot.envs.make_env` | Reuse the environment and its lifecycle. |
| Evaluation | `lerobot-eval` / `lerobot.scripts.lerobot_eval:main` | Return bridge-owned metric summaries while leaving evaluation control upstream. |
| Plugins | LeRobot third-party discovery and `PreTrainedConfig` registration | Load upstream plugins; do not create a parallel policy registry. |

`LeRobotPolicyContext` holds raw policy, processor, dataset, and environment
objects with bridge metadata. `LeRobotEvaluationResult` owns only the compact
episode and metric summary that latent-anything needs for downstream evidence.

## Dataset bridge

`describe_lerobot_dataset(dataset)` reads the canonical `dataset.meta` object
and returns typed feature, normalization, task, episode-range, camera, and
provenance descriptors. It reads metadata only; it does not open Parquet or
video files.

`read_lerobot_episode(dataset, episode_index, start_frame=..., stop_frame=...)`
returns a lazy iterator. Each `LeRobotSample.values` mapping is the original
processor-ready mapping returned by LeRobot, including its PyTorch tensors.
The sample's `LeRobotSampleProvenance` records the dataset revision, episode
and frame indices, timestamp, task index, and task label.

For streaming datasets, wrap the upstream `StreamingLeRobotDataset` with
`LeRobotStreamingReader` or use `stream_lerobot_samples`. The upstream class
continues to own shard iteration, video decoding, temporal windows, and its
shuffle buffer; the bridge retains only a bounded recent sample window for
inspection. A captured policy latent crosses into latent-anything only through
`captured_latent_to_numpy` / `captured_latent`, which makes the copy and NumPy
conversion explicit.

The offline alignment fixture is in
`tests/test_lerobot_dataset_bridge.py`. To inspect a public v3 dataset's
schema and episode provenance without downloading data or videos, run:

```text
uv run python scripts/lerobot_dataset_inspection.py lerobot/aloha_sim_insertion_human --revision v3.0
```

The checked-in result is
`artifacts/lerobot_dataset_inspection.json`. It is an inspection record only;
it makes no policy-quality or model-performance claim.

## ACT representation capture

Sprint 58 pins the public ACT checkpoint
`lerobot/act_aloha_sim_insertion_human` at revision
`33259aa86eb45fdf85350280044a33d9d50e40c3` and its paired dataset
`lerobot/aloha_sim_insertion_human` at revision
`cc571a3c661df81b566dbfde3d5c1e85fcdf7884`. The pair is represented by
`ACTCheckpointSpec` in `src/latent_anything/integrations/lerobot_act.py`.

`load_act_policy()` loads `ACTConfig` and `LeRobotDatasetMetadata` through
LeRobot, then delegates policy construction and official pre/post-processor
construction to the raw factories exposed by `LeRobotAPI`. The
`ACTPolicyAdapter` calls the normal preprocessor → `policy.select_action()` →
postprocessor path. Its single observational capture point is `model.decoder`;
the first decoder query token is retained because it directly feeds ACT's
first selected action through `action_head`.

`capture_episode()` preserves LeRobot's action queue and records only calls
that actually query the policy. `analyze_act_traces()` runs PCA projection,
linear outcome probing, majority/shuffled-label/raw-input controls, and
Euclidean trajectory-length/velocity summaries. These are observational
analyses only; intervention and environment-level causal effects are reserved
for Sprint 61.

The deterministic offline evidence command is:

```text
uv run python scripts/act_policy_representation_benchmark.py
```

It writes `artifacts/act_policy_representation_benchmark.json`. The pinned
checkpoint smoke is marked `network` and `large_download`; opt into it with
`LATENT_ANYTHING_RUN_NETWORK=1` in an environment containing the `lerobot`
extra.

## Diffusion Policy representation capture

Sprint 59 pins the public Diffusion Policy checkpoint
`LeTau/diffusion_aloha_insertion` at model revision `6126e33` and its paired
image dataset `lerobot/aloha_sim_insertion_human_image` at dataset revision
`d93d36a`. The compatible environment/task is `aloha` / `AlohaInsertion-v0`,
as recorded in `DiffusionCheckpointSpec`.

Install the policy profile with:

```text
uv sync --extra lerobot-diffusion
```

`load_diffusion_policy()` loads `DiffusionConfig` and
`LeRobotDatasetMetadata` through LeRobot, then delegates policy construction
and official pre/post-processor construction to the raw factories exposed by
`LeRobotAPI`. The `DiffusionPolicyAdapter` does not reimplement denoising or
normalization. It observes the global conditioning tensor entering
`policy.diffusion.unet` and the U-Net output at every scheduler timestep while
the normal LeRobot action queue remains in control.

`DiffusionEpisodeTrace` keeps environment/episode time, action-chunk position,
and diffusion timestep explicit. Conditioning trajectories are analyzed in
their own space; denoising trajectories are grouped by the recorded
`diffusion_timestep`. The analysis is observational and includes PCA, a linear
probe, majority/shuffled-label controls, and a held-out Gaussian-mixture
density comparison. It does not claim causal policy or environment effects.

The deterministic offline evidence command is:

```text
uv run python scripts/diffusion_policy_representation_benchmark.py
```

It writes `artifacts/diffusion_policy_representation_benchmark.json`. The
marked public checkpoint smoke is in `tests/test_lerobot_diffusion.py` and is
opt-in with `LATENT_ANYTHING_RUN_NETWORK=1`.

## SmolVLA representation capture and bounded intervention

Sprint 60 pins the public SmolVLA checkpoint `lerobot/smolvla_libero` at model
revision `31d453f7edd78c839a8bbc39744a292686daf0de` and its documented training
dataset `lerobot/libero` at dataset revision
`a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` (LIBERO-10 `libero_spatial`, action
dimension 7, state dimension 8). The model card's `train_config.json` records
this exact pair. The compatible environment/task is `libero` /
`libero_spatial`, as recorded in `SmolVLACheckpointSpec` together with the
reproducible hardware profile (`SMOLVLA_HARDWARE_PROFILE`: SmolVLM2-500M
backbone, 16 VLM + 16 expert layers at 75% width, bfloat16, ~450M parameters,
16 GB GPU recommended for the intervention lane).

Install the policy profile with:

```text
uv sync --extra lerobot-smolvla
```

`load_smolvla_policy()` loads `SmolVLAConfig` and `LeRobotDatasetMetadata`
through LeRobot, then delegates policy construction (with the model card's
official camera `rename_map`) and pre/post-processor construction to the raw
factories exposed by `LeRobotAPI`. The device step of the loaded processor
pipelines is overridden to the requested device. The `SmolVLAPolicyAdapter`
does not reimplement image preparation, flow-matching denoising, normalization,
or action queueing; it observes the official `preprocess → select_action →
postprocess` path.

Four module seams are captured per executed action query with token/modality
metadata:

| Representation | Hook location | Token metadata |
| --- | --- | --- |
| Vision context | `model.vlm_with_expert.vlm.model.vision_model` (SigLIP) | patch-token count, camera name, prefix offset |
| Language context | `model.vlm_with_expert.vlm.model.text_model.embed_tokens` | token count, prefix offset |
| State context | `model.state_proj` | single token, prefix offset |
| Action expert | `model.vlm_with_expert.lm_expert.norm` | chunk-token count, denoising step |

The context captures reproduce the model's own prefix layout (vision tokens,
then language tokens, then the projected state token); expert captures are
recorded once per denoising step (10 steps for the pinned checkpoint) in the
suffix. Queue hits execute no model query and produce no captures.

`SmolVLAIntervention` is one bounded additive intervention on the action-expert
representation: `z <- z + strength * direction` at every denoising step, with
finite direction, shape, and bounded-strength validation. Strength zero
short-circuits and returns the unchanged output, so baseline and intervened
actions are bit-identical. Hooks live only for the duration of one
`select_action` call (exception-safe lifecycle).

`measure_smolvla_intervention()` reports the immediate action change
(per-dimension), the on-target/off-target decomposition of that change against
the action-space direction induced by the expert direction through the policy's
own `action_out_proj`, per-token representation drift over denoising steps,
and prompt/camera-order sensitivity of the baseline policy.

The deterministic offline evidence command is:

```text
uv run python scripts/smolvla_policy_representation_benchmark.py
```

It writes `artifacts/smolvla_policy_representation_benchmark.json`. The marked
public checkpoint lane is in `tests/test_lerobot_smolvla.py` and requires a
CUDA device plus `LATENT_ANYTHING_RUN_NETWORK=1` in an environment containing
the `lerobot-smolvla` extra. The SmolVLA claim is an observational and
bounded-intervention claim; environment-level causal effects are the Sprint 61
simulation benchmark below.

## SmolVLA causal simulation benchmark

Sprint 61 runs the pinned SmolVLA checkpoint inside the LIBERO simulation
evaluation loop under four predeclared conditions and judges the explanation
by its controlled behavioral effect:

| Condition | Definition |
| --- | --- |
| `no_hook` | Official preprocess → `select_action` → postprocess path with no capture hooks installed. |
| `baseline` | Capture hooks installed with an intervention at strength zero (bit-exact identity, so behavior must equal `no_hook`). |
| `random` | A seeded unit-norm random action-expert direction added at every denoising step. |
| `targeted` | The action-expert direction inducing the largest change along one declared action axis through the policy's own `action_out_proj` (`action_axis=0`). |

The harness (`latent_anything.integrations.lerobot_benchmark`) owns the
experiment protocol and bridge-owned results; LeRobot owns the environment
(`make_env_config` + `make_env` with `n_envs=1`), the observation conversion
(`get_env_processors` + `preprocess_observation`), the policy, and its
processors. Every condition replays the same episode seeds from the same
initial state with identical fixed noise, so the only behavioral difference
between conditions is the intervention.

Metrics per episode: success, return, episode length, per-step action
deviation against the `no_hook` trajectory, per-query latency, and executed
query count. Aggregates carry a Wilson 95% interval for success rate and a
normal-approximation interval for return. Offline explanation scores (on-target
fraction, action-change norm, representation drift from
`measure_smolvla_intervention`) are paired with the environment-level effects
and checked against predeclared disagreement rules (overstatement,
understatement, reversal).

The real lane requires CUDA, `LATENT_ANYTHING_RUN_NETWORK=1`, and the
`lerobot-smolvla` profile, which since Sprint 61 also carries LeRobot's
Linux-only `libero` environment extra. The deterministic offline fixture and
the separately marked statistical benchmark live in
`tests/test_lerobot_benchmark.py`. The reproducible evidence command is:

```text
uv run python scripts/smolvla_simulation_benchmark.py --seeds 1 2 3
```

It writes `artifacts/smolvla_simulation_benchmark.json` (full config, episode
rows, condition summaries, correlation, acceptance, failure analysis), a
standalone `artifacts/smolvla_simulation_benchmark_config.json`, and
`artifacts/smolvla_simulation_benchmark.png` plots. Videos are intentionally
omitted (large files; the quantitative failure analysis covers behavior).
ACT and Diffusion have no intervention surface, so their claims remain
observational — only the SmolVLA environment-level evidence can promote the
causal-intervention capability, and only when the predeclared acceptance gate
passes.

The retained artifact (seeds 1–3, strengths 1/5/10) is historical evidence from
the earlier pinned public pair. Its reported outcomes remain unchanged: the
baseline is bit-exact; the targeted intervention leaves behavior unchanged at strength
1 (offline on-target 0.86 with zero success delta — a recorded overstatement
disagreement) and harms success from 1.0 to 0.0 at strengths 5 and 10
(recorded reversal disagreements; all six episodes max out at 280 steps),
while the random control never changes success.
The authoritative ledger keeps `THY-T05-CAUSAL-INTERVENTION-VS-
OBSERVATIONAL-STUDY` at D2 pending a corrected, pinned real CUDA rerun; this
document does not promote the retained artifact to D3.

## Sprint 62 run records and inspection commands

Sprint 62 adds a versioned local evidence contract in
`latent_anything.run_record`. A `RunRecord` keeps the declared config,
code/framework version, pinned model and dataset revisions, seeds, execution
environment, metrics, content-addressed artifacts, parent/child links, runtime
profile, and theory evidence identifiers. Its identity is a SHA-256 hash of
the reproducible inputs; lifecycle timestamps and status are deliberately
excluded so interrupted recovery and duplicate detection do not change the run
identity.

`FileSystemRunRecorder` stores records at `runs/<run_id>.json` and artifacts at
`artifacts/<sha256>`. JSON and artifact writes use a temporary file followed by
an atomic replace. A process restart can call `recover_interrupted()` to mark
unfinished records honestly. The recorder is intentionally a concrete local
implementation; external tracking backends remain a later sprint.

The package CLI is available as `latent-anything` or
`uv run python -m latent_anything.cli`:

```text
latent-anything capture-points
latent-anything inspect-policy --policy act
latent-anything inspect-dataset lerobot/aloha_sim_insertion_human --revision v3.0
latent-anything replay-run <run-id> --record-root artifacts/runs --output replay.json
latent-anything compare-runs <run-a> <run-b> --record-root artifacts/runs
```

`capture-points` lists the supported ACT, Diffusion, and SmolVLA seams without
importing LeRobot. `inspect-policy` reports pinned metadata and claim scope;
the dataset command is the only one that loads the optional dataset extra.
The bridge helpers `record_lerobot_dataset_inspection`,
`record_lerobot_policy_capture`, `record_lerobot_intervention`, and
`record_lerobot_evaluation` attach JSON result artifacts while preserving
upstream-owned policy/dataset objects.

The checked-in comparison artifact
`artifacts/lerobot_policy_comparison.json` compares the pinned ACT and
Diffusion observational evidence. It keeps the two protocol/dataset
identities explicit and reports metric deltas without treating different
representation seams as a policy-quality leaderboard.

## Explicitly rejected scope

The bridge does not:

* reimplement LeRobot policies, policy configs, processors, normalizers, or action semantics;
* reimplement `LeRobotDataset`, its Parquet/video readers, streaming decoder, or metadata format;
* create a second robot or simulation environment abstraction;
* replace `lerobot-eval` with a latent-anything evaluation loop;
* discover plugins through a latent-anything-specific policy registry;
* import every policy's optional dependencies as part of the base LeRobot extra.

Latent-anything may add capture hooks, latent descriptors, provenance, and
analysis/evidence results around those upstream objects. Any new shared
interface must be extracted from a concrete third implementation according
to the project's Rule of Three.

## Resolver conflicts and upgrades

The legacy `transformers` and `diffusers-full` extras use the Transformers 4.x
and `huggingface-hub<1` line. LeRobot 0.6.x uses `huggingface-hub>=1` and its
newer policy extras use Transformers 5.x, so uv declares those profiles
mutually conflicting. Do not co-install them in one environment; use a fresh
environment or a separate CI job.

Before changing the supported window:

1. Confirm the latest stable PyPI release and record its wheel hash and upload date.
2. Re-read upstream `pyproject.toml`, policy factory, processor package, dataset, environment, evaluation, and plugin-discovery sources.
3. Run the base and `--extra lerobot` import-isolation/CPU smoke tests.
4. Validate one policy factory, one processor pipeline, one dataset fixture, and one evaluation entry point against the new lock.
5. Update the compatibility report, this document, the sprint artifact, and the changelog in the same change.
