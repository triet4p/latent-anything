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
remain upstream-selected. For example, ACT is part of the base policy
surface, while Diffusion and SmolVLA require their corresponding LeRobot
policy extras in a consuming environment.

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
