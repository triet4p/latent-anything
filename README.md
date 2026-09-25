# latent-anything

> *Latent Understanding, Manipulation & Execution Network*

A Python framework that treats latent space as a first-class object: load latent representations from models, inspect them, manipulate them, and execute small runtime pipelines.

## Pre-stable Scope

`0.9.0` is a pre-stable API/evidence baseline (pre-1.0). It includes:

- Core primitives: `LatentSpace`, `LatentValue`, and `Trajectory`
- Layer A introspection methods: PCA, UMAP, and SAE
- Layer B manipulation methods: Lerp, SteeringVector, and ActivationPatch
- Representative built-in adapters: VAE, RandomProjection, HiddenStateAdapter,
  and GaussianRendererAdapter; the checked-in source export inventory is in
  the [API reference](docs/API_REFERENCE.md), while
  [M14_REAL_SYSTEM_VALIDATION](docs/M14_REAL_SYSTEM_VALIDATION.md#registry-plugin-groups-optional-profiles) documents registry, plugin, and
  profile validation.
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

This pre-stable baseline does not claim the full Latent Anything thesis is implemented. The
world-model and tokenized lanes are compact synthetic CPU references, not
real-checkpoint or CUDA claims. Sprint 35 also records bounded local-CPU D2
fidelity and ordered interpolation evidence for one cached Diffusers VAE; this
does not claim perceptual quality or a complete diffusion pipeline. Stable release work remains
future milestones. Sprint 75 streaming evidence is a synthetic offline CPU
rollout story, not a LeRobot or real-model throughput claim. Sprint 76
tracking evidence is local/offline only; it does not claim hosted tracking,
remote servers, or team workflows.

APIs are still pre-1.0 and may change under normal `0.x` SemVer expectations.
Sprint 79 closes the broad inventory as the `0.9.0` pre-stable API/evidence
baseline. Package metadata is `0.9.0`, and the GitHub Release is the authoritative
0.9.0 distribution channel. PyPI publication is explicitly deferred and is not a
release or installation gate. Sprint 81 targets `1.0.0`, with publication
stopping if any required supported-claim, packaging, documentation, or workflow
gate is missing; Sprint 80 is the depth-first diagnostic gate.

Registry configs now use `adapter`, `analysis`, and `intervention` kinds.
The beta `method_a` and `method_b` spellings remain supported with a migration
warning in `0.9.0`; removal is deferred past `0.9.0` pending a separate reviewed
migration decision. Run `uv run python scripts/report_config_migration.py <config.json>`
to inspect repository-owned JSON configs without rewriting them.

See the [migration and compatibility guide](docs/MIGRATION.md) and the
[API reference](docs/API_REFERENCE.md) for the complete 18-row alias ledger,
the two separate schema/path migrations, frozen signatures, and current
source API contract.

The checked-in `0.9.0` source snapshot records 214 runtime top-level exports,
211 canonical-stable entries, 28 config schemas, 89 public dataclass/result
schemas, and 8 public exceptions (SHA-256
`3fd8c73e6fd9fa9bae107b4be6727cc3e9c3c907c9b9d757fdae4c267d8691ec`). These
numbers describe the current source tree, not the already-published `0.9.0`
distribution: the release-time snapshot had 205 runtime top-level exports,
202 canonical-stable entries, 7 public exceptions, and digest
`048ac553adabb11c24d3e1f4d86e0c6d469df6590064014c915d6a08c7d53c26`. The
historical release record is preserved in [CHANGELOG.md](CHANGELOG.md).
The current source snapshot is
`artifacts/api_freeze_snapshot_0.9.0.json`; the historical beta snapshot
`artifacts/api_freeze_snapshot_0.1.0b1.json` is preserved unchanged.

## Installation

For the `0.9.0` release, download the wheel or sdist from the
[GitHub Release assets](https://github.com/triet4p/latent-anything/releases/tag/v0.9.0).
Verify the downloaded file against `SHA256SUMS`, then install the local asset:

```bash
uv pip install ./latent_anything-0.9.0-py3-none-any.whl
# or, for the source distribution:
uv pip install ./latent_anything-0.9.0.tar.gz
```

PyPI publication is deferred for this release and is intentionally non-gating.
For development from a checkout, use [uv](https://docs.astral.sh/uv/):

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
# 0.9.0

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

The `0.9.0` release workflow builds one deterministic wheel and sdist, records
`SHA256SUMS` and `PROVENANCE.json`, and uploads those exact files plus the
release notes to the GitHub Release after the gates pass. PyPI is deferred and
does not participate in this release contract.

The `v0.9.0` tag is the release input:

```bash
git push origin v0.9.0
```

The workflow also accepts plain tags such as `0.9.0`; the `v` prefix is
recommended to keep package releases visually distinct from theory deployment
tags such as `theory-v*`. The historical `v0.1.0-beta.1` tag and release remain
unchanged history.

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
