# latent-anything

> *Latent Understanding, Manipulation & Execution Network*

A Python framework that treats latent space as a first-class object: load latent representations from models, inspect them, manipulate them, and execute small runtime pipelines.

## Published 0.9.0 Baseline and 1.0.0 Candidate

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

The published `0.9.0` package remains pre-1.0 and follows the `0.x` compatibility expectation.

Sprint 81's package metadata targets `1.0.0`, but this candidate has not been
tagged or released. Its diagnostic claim is limited to the two accepted
ordinary-DL cases and separate stability supplement in
[`docs/SPRINT_80_DEPTH_EVIDENCE.md`](docs/SPRINT_80_DEPTH_EVIDENCE.md); this is
not broad VLA, GPU/CUDA, or arbitrary-model support. The stable-tag ruleset is
active and the owner-managed App-token proof passed in default-branch workflow
run [36247385348](https://github.com/triet4p/latent-anything/actions/runs/36247385348)
on commit `f87726950d5ae59d4287677f026c39e56baef236`. Push-triggered CI run
[36247369268](https://github.com/triet4p/latent-anything/actions/runs/36247369268)
passed its Python 3.12/3.13/3.14 matrix on that exact commit. The pending
changelog/documentation commit still requires green CI for its own SHA before
release. The PyPI first upload remains pending. Task 4's clean package build
and installation checks are complete.

Registry configs use `adapter`, `analysis`, and `intervention` kinds. The beta
`method_a` and `method_b` spellings remain deprecated warning aliases. The
1.0.0 candidate retains all 18 beta aliases; they remain available through the
1.x line, and removal requires a separately reviewed major-version migration.
Run `uv run python scripts/report_config_migration.py <config.json>` to inspect
repository-owned JSON configs without rewriting them.

See the [migration and compatibility guide](docs/MIGRATION.md) and the
[API reference](docs/API_REFERENCE.md) for the complete 18-row alias ledger,
the two separate schema/path migrations, frozen signatures, and current
source API contract. The proposed 1.x compatibility, security-reporting
status, upstream dependency, and artifact-migration commitments are in the
[support policy](docs/SUPPORT_POLICY.md). Security-reporting availability and
instructions are also summarized in [`SECURITY.md`](SECURITY.md); the repository
does not currently claim a verified private vulnerability-reporting route.

The unpublished `1.0.0` candidate source snapshot records 214 runtime top-level
exports, 211 canonical-stable entries, 28 config schemas, 89 public
dataclass/result schemas, and 8 public exceptions (SHA-256
`46ef4edd2bcb5d6235a02633caba10408b6e2000d5614928bef0794b20b005f4`). These
numbers describe candidate source, not the already-published `0.9.0`
distribution: its release-time snapshot had 205 runtime top-level exports,
202 canonical-stable entries, 7 public exceptions, and digest
`048ac553adabb11c24d3e1f4d86e0c6d469df6590064014c915d6a08c7d53c26`. The
historical release record is preserved in [CHANGELOG.md](CHANGELOG.md). The
candidate snapshot is `artifacts/api_freeze_snapshot_1.0.0.json`; the later
pre-candidate `0.9.0`-labeled source snapshot and historical beta snapshot
remain preserved separately.

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
# 1.0.0 (unpublished candidate source)

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

The audited release workflow runs the readiness preflight, strict documentation,
quality, test, and package checks before its dedicated job creates the stable
tag. The initial candidate commit's push-triggered CI run
[36247369268](https://github.com/triet4p/latent-anything/actions/runs/36247369268)
passed the Python 3.12/3.13/3.14 matrix on `f87726950d5ae59d4287677f026c39e56baef236`.
The App-token proof and PyPI Trusted Publisher configuration are verified, but
the first upload remains pending. The final changelog/documentation commit has
a new SHA and must pass its own exact-commit CI plus all guarded release-workflow
checks before a tag or upload is authorized.

The only published package release remains `v0.9.0`, available from the
[GitHub Release assets](https://github.com/triet4p/latent-anything/releases/tag/v0.9.0).
Its historical install instructions and checksums are documented in
[`docs/MIGRATION.md`](docs/MIGRATION.md). The 1.0.0 candidate notes are a
[draft only](docs/RELEASE_NOTES_1.0.0.md).

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
- [1.0.0 candidate release notes (unpublished draft)](docs/RELEASE_NOTES_1.0.0.md)
- [Support and versioning policy](docs/SUPPORT_POLICY.md) - Candidate 1.x support, SemVer, deprecation, upstream compatibility, and artifact migration.
- [Security policy](SECURITY.md) - Current private-reporting availability and responsible-disclosure handling.

## License

MIT - see [LICENSE](LICENSE).
