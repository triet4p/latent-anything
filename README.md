# latent-anything

> *Latent Understanding, Manipulation & Execution Network*

A Python framework that treats latent space as a first-class object: load latent representations from models, inspect them, manipulate them, and execute small runtime pipelines.

## Published 1.0.0 Release and 0.9.0 Baseline

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

At the published `0.9.0` pre-stable baseline, the project did not claim the
full Latent Anything thesis was implemented. The world-model and tokenized
lanes were compact synthetic CPU references, not real-checkpoint or CUDA
claims. Sprint 35 recorded bounded local-CPU D2 fidelity and ordered
interpolation evidence for one cached Diffusers VAE; this did not claim
perceptual quality or a complete diffusion pipeline. Stable release work was
still a future milestone at that checkpoint. Sprint 75 streaming evidence is
a synthetic offline CPU rollout story, not a LeRobot or real-model throughput
claim. Sprint 76 tracking evidence is local/offline only; it does not claim
hosted tracking, remote servers, or team workflows.

The published `0.9.0` package remains pre-1.0 and follows the `0.x` compatibility expectation.

`1.0.0` was published from release commit
`a449ca33b4b83ce109c29db7cf471919820b1d56` under protected annotated tag
`v1.0.0`. Exact-SHA CI run
[36250365395](https://github.com/triet4p/latent-anything/actions/runs/36250365395)
passed all Python 3.12/3.13/3.14 jobs. Audited release run
[36251907256](https://github.com/triet4p/latent-anything/actions/runs/36251907256)
completed all five jobs, including artifact attestations, GitHub Release, and
the PyPI OIDC upload after owner approval of environment `pypi`
(`22803163260`). The PyPI Trusted Publisher is active; project JSON returns
HTTP 200 for version `1.0.0`, and the wheel/sdist SHA-256 digests match the
GitHub Release assets.

The released diagnostic claim is limited to the two accepted ordinary-DL cases
and separate stability supplement in
[`docs/SPRINT_80_DEPTH_EVIDENCE.md`](docs/SPRINT_80_DEPTH_EVIDENCE.md); this is
not broad VLA, GPU/CUDA, or arbitrary-model support. Task 4's clean package
build and installation checks are complete.

Registry configs use `adapter`, `analysis`, and `intervention` kinds. The
`method_a` and `method_b` spellings remain deprecated warning aliases.
Published `1.0.0` retains all 18 beta aliases; they remain available through
the `1.x` line, and removal requires a separately reviewed major-version
migration.
Run `uv run python scripts/report_config_migration.py <config.json>` to inspect
repository-owned JSON configs without rewriting them.

See the [migration and compatibility guide](docs/MIGRATION.md) and the
[API reference](docs/API_REFERENCE.md) for the complete 18-row alias ledger,
the two separate schema/path migrations, frozen signatures, and current
source API contract. Current 1.x support, security-reporting status, upstream
compatibility, and artifact-migration commitments are in the
[support policy](docs/SUPPORT_POLICY.md). Security-reporting availability and
instructions are also summarized in [`SECURITY.md`](SECURITY.md); the repository
does not currently claim a verified private vulnerability-reporting route.

The published `1.0.0` source API snapshot records 214 runtime top-level
exports, 211 canonical-stable entries, 28 config schemas, 89 public
dataclass/result schemas, and 8 public exceptions (SHA-256
`46ef4edd2bcb5d6235a02633caba10408b6e2000d5614928bef0794b20b005f4`). It
describes the API shipped at tag `v1.0.0`, not a later source inventory. The
previous `0.9.0` release-time snapshot had 205 runtime top-level exports,
202 canonical-stable entries, 7 public exceptions, and digest
`048ac553adabb11c24d3e1f4d86e0c6d469df6590064014c915d6a08c7d53c26`. The
historical release record is preserved in [CHANGELOG.md](CHANGELOG.md). The
1.0.0 snapshot is `artifacts/api_freeze_snapshot_1.0.0.json`; the pre-candidate
`0.9.0`-labeled source snapshot and historical beta snapshot remain preserved
separately.

## Installation

Install the published `1.0.0` package from PyPI:

```bash
uv pip install latent-anything==1.0.0
```

Alternatively, download the wheel or sdist from the
[GitHub Release assets](https://github.com/triet4p/latent-anything/releases/tag/v1.0.0),
verify the file against `SHA256SUMS`, then install the local asset. The earlier
`0.9.0` release record and its migration notes remain available in
[`docs/MIGRATION.md`](docs/MIGRATION.md).

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
# 1.0.0

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

## Release Evidence

The audited release workflow runs the readiness preflight, strict
documentation, quality, test, package, and attestation checks before creating
the stable tag. Release commit
`a449ca33b4b83ce109c29db7cf471919820b1d56` passed exact-SHA CI run
[36250365395](https://github.com/triet4p/latent-anything/actions/runs/36250365395)
for Python 3.12, 3.13, and 3.14. Audited release run
[36251907256](https://github.com/triet4p/latent-anything/actions/runs/36251907256)
completed successfully across all five jobs, including the approved PyPI OIDC
upload. Protected tag `v1.0.0` and its GitHub Release point to that release
commit; provenance binds the tag, commit, and run, and attestations verified
for both distributions.

The active PyPI project JSON endpoint is
[HTTP 200](https://pypi.org/pypi/latent-anything/json), version `1.0.0`; the
wheel and sdist digests match the GitHub Release assets. This README
reconciliation is later than the tagged release commit and does not change the
published tag or assets. The previous `v0.9.0` GitHub Release and its historical
install instructions remain documented in [`docs/MIGRATION.md`](docs/MIGRATION.md).

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
- [1.0.0 release notes](docs/RELEASE_NOTES_1.0.0.md)
- [Support and versioning policy](docs/SUPPORT_POLICY.md) - 1.x support, SemVer, deprecation, upstream compatibility, and artifact migration.
- [Security policy](SECURITY.md) - Current private-reporting availability and responsible-disclosure handling.

## License

MIT - see [LICENSE](LICENSE).
