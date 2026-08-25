# Optional Integrations

The base package never imports Diffusers, Transformers, gsplat, or LeRobot.
Install one boundary deliberately: `uv sync --extra diffusers`,
`uv sync --extra transformers`, `uv sync --extra 3d`, or `uv sync --extra lerobot`.
The LeRobot extra installs its dataset/evaluation feature dependencies and is
currently pinned to the LeRobot 0.6.x API window; see
[LEROBOT_INTEGRATION.md](LEROBOT_INTEGRATION.md) for the supported seams and
its resolver conflicts with the legacy Transformers/Diffusers-full profiles.

Unit tests are CPU-only and use tiny local fixtures. Tests that require a GPU,
network acquisition, or large checkpoints must be explicitly marked `gpu`,
`network`, or `large_download`; cached model paths are supplied by CI and tests
must never download a model implicitly.

Integration upgrades require a bounded version range, a lower-bound import
smoke test, and a pinned model/backend revision in the consuming sprint.

## Experiment tracking (Sprint 76)

The base package exposes the validated `ExperimentRecorder` contract without
importing either tracking SDK. Install the adapters deliberately with
`uv sync --extra tracking-mlflow`, `uv sync --extra tracking-wandb`, or the
combined `tracking` extra. MLflow is supported only with a local file tracking
URI in this sprint; W&B is supported only in `offline` or `disabled` mode.
MLflow accepts canonical relative paths, absolute `Path` values, and absolute
Windows drive-path strings such as `C:/tracking`; drive-relative, encoded,
UNC/device, traversal, and remote/URI-like forms remain rejected.

```python
from latent_anything.integrations.mlflow_recorder import MLflowRecorder
from latent_anything.integrations.wandb_recorder import WandbRecorder

local_mlflow = MLflowRecorder("runs/mlruns")
offline_wandb = WandbRecorder("latent-anything", mode="offline")
run = local_mlflow.start_run("cpu-fixture", config={"seed": 7})
run.log_metrics({"score": 0.75}, step=0)
run.finish()
```

Inputs are bounded and canonicalized before provider operations: metadata is
explicit caller-provided safe metadata (the process environment is never
captured), secret-like keys/values and non-canonical artifact names are
rejected, and byte/path reads are bounded. Metric steps are non-decreasing and
resuming a run requires complete persisted adapter identity and exact provider
run-ID continuity; missing, changed, or newly-created provider runs fail
closed. W&B expresses parent/child relationships with
an offline group and parent tag; this is intentionally narrower than an
MLflow nested run. W&B offline reinitialization does not reliably persist the
adapter identity, so the real offline lane rejects resume rather than treating
the provider run ID as proof of continuation. In offline mode the adapter also
keeps a validated local artifact mirror because the SDK does not provide a
portable provider-side read API. Remote servers, authentication, hosted
artifact stores, dashboards, and team workflows are delegated to the
providers and are outside the Sprint 76 evidence claim. Use
`tests/test_tracking_parity.py` for the deterministic affine world-model
fixture recorded through all three backends; the real SDK tests are explicitly
opt-in, network-denied, and temporary-local.

Fresh external starts intentionally create a new provider run even when their
canonical identity matches an earlier run. Use `resume_run_id` for explicit
continuation only when the provider persists the adapter identity; W&B offline
fails closed when it does not. The local filesystem adapter retains its
existing identity-based deduplication behavior.
