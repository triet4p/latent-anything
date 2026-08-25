# latent-anything

> *Latent Understanding, Manipulation & Execution Network*

A Python framework that treats latent space as a first-class object: load latent representations from models, inspect them, manipulate them, and execute small runtime pipelines.

## Beta Scope

`0.1.0-beta.1` is a pre-1.0 core-framework beta. It includes:

- Core primitives: `LatentSpace`, `LatentValue`, and `Trajectory`
- Layer A introspection methods: PCA, UMAP, and SAE
- Layer B manipulation methods: Lerp, SteeringVector, and ActivationPatch
- Built-in adapters: VAE, RandomProjection, HiddenStateAdapter, and GaussianRendererAdapter
- Registry/config construction for built-ins
- Concrete analysis and manipulation pipelines
- First runtime helpers: batching, in-memory cache, async wrappers, and profiling hooks
- World-model/planning lanes: deterministic, stochastic-Gaussian, and RSSM transitions; rollout; reward/value evaluation; CEM and MPPI
- Discrete VQ-VAE and tokenized-world-model reference lanes, plus decoder-free JEPA/LeWM-style prediction
- Local run recording with content-addressed evidence artifacts and optional LeRobot ACT/Diffusion/SmolVLA bridges
- Explicit external plugin discovery through canonical Python entry-point groups, with a separately installed hello-world proof
- Versioned Arrow-backed portable NumPy/domain artifacts, allowlisted typed envelopes, checksummed atomic storage, and a bounded SQLite disk cache with cross-process CPU parity evidence
- Concrete bounded-memory rollout streaming over ordered exact-NumPy action chunks, with sync/async eager-equivalence, cancellation, backpressure, and profiling evidence
- Optional MLflow local-file and W&B offline/disabled experiment recorders
  behind a validated contract with offline parity evidence
- Script-level demos and tracked release artifacts

This beta does not claim the full Latent Anything thesis is implemented. The
world-model and tokenized lanes are compact synthetic CPU references, not
real-checkpoint or CUDA claims. Sprint 35 also records bounded local-CPU D2
fidelity and ordered interpolation evidence for one cached Diffusers VAE; this
does not claim perceptual quality or a complete diffusion pipeline. API freeze and stable release work remain
future milestones. Sprint 75 streaming evidence is a synthetic offline CPU
rollout story, not a LeRobot or real-model throughput claim. Sprint 76
tracking evidence is local/offline only; it does not claim hosted tracking,
remote servers, or team workflows.

APIs are still pre-1.0 and may change under normal `0.x` SemVer expectations.

Registry configs now use `adapter`, `analysis`, and `intervention` kinds.
The beta `method_a` and `method_b` spellings remain supported with a migration
warning until `0.9.0`; run `uv run python scripts/report_config_migration.py <config.json>`
to inspect repository-owned JSON configs without rewriting them.

## Installation

The package is not published yet. Clone and install locally using [uv](https://docs.astral.sh/uv/):

```bash
git clone <repo-url>
cd latent-anything
uv sync --locked
```

## Quick Start

```python
import numpy as np

import latent_anything
from latent_anything import LatentSpace, Trajectory
from latent_anything.methods import PCA

print(latent_anything.__version__)
# 0.1.0b1

rng = np.random.default_rng(42)
space = LatentSpace(dim=4)
points = rng.normal(size=(12, space.dim))
trajectory = Trajectory(data=points)

pca = PCA(n_components=2)
projection = pca.fit_transform(trajectory.to_numpy())

print(projection.shape)
# (12, 2)
```

## Demos

Start with the release demo index:

- [Release demo index](artifacts/release_demo_index_0.1.0-beta.1.md)
- [Showcase summary](artifacts/showcase_demo_summary.txt)
- [Showcase plot](artifacts/showcase_demo_plot.png)
- [Async runtime summary](artifacts/async_runtime_demo_summary.txt)

Representative scripts live in `scripts/`, including:

- `scripts/end_to_end_showcase_demo.py`
- `scripts/end_to_end_manipulation_demo.py`
- `scripts/end_to_end_gaussian_renderer_demo.py`
- `scripts/end_to_end_async_runtime_demo.py`

## Release Gate

Before tagging a package release, run:

```bash
uv sync --locked
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pyright
uv run pytest
```

Recommended release tag:

```bash
git tag v0.1.0-beta.1
git push origin v0.1.0-beta.1
```

The GitHub Release workflow also accepts plain tags such as `0.1.0-beta.1`, but the `v` prefix is recommended to keep package releases visually distinct from theory deployment tags such as `theory-v*`.

## Project Structure

```text
latent-anything/
├── src/
│   └── latent_anything/   # Main framework package
├── tests/                 # Pytest suite
├── scripts/               # End-to-end demos
├── artifacts/             # Demo outputs, audits, and task summaries
├── docs/                  # Architecture, theory, and sprint docs
├── latent-anything-theory/ # Standalone theory research sub-project
├── .agents/               # Agent rules, skills, and memory
├── .github/workflows/     # CI, theory deploy, and release workflows
├── pyproject.toml
└── CHANGELOG.md
```

## Documentation

- [docs/IDEA.md](docs/IDEA.md) - Vision and motivation
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Core primitives and layer design
- [docs/THEORY.md](docs/THEORY.md) - Theoretical foundations
- [docs/PLAN.md](docs/PLAN.md) - Incremental project plan
- [docs/PLUGIN_AUTHOR_GUIDE.md](docs/PLUGIN_AUTHOR_GUIDE.md) - External plugin contract and security boundary
- [docs/PLUGIN_TEMPLATE.md](docs/PLUGIN_TEMPLATE.md) - Minimal plugin package template

## License

MIT - see [LICENSE](LICENSE).
