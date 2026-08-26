# Final Release Readiness: 0.1.0-beta.1

**Sprint:** 26
**Task:** 12
**Status:** Ready to tag

> **Historical/superseded snapshot (2026-07-10 beta):** This artifact records
> the `0.1.0-beta.1` decision at Sprint 26. It is not current release approval.
> Current status is governed by [M14](../docs/M14_REAL_SYSTEM_VALIDATION.md),
> [the evidence ledger](../docs/EVIDENCE_LEDGER.md), and
> [the machine-readable ledger](../docs/evidence-ledger.json). Retain the
> original dates, metrics, and beta decision for historical traceability.

## Release Workflow Behavior

- Workflow file: `.github/workflows/release.yml`
- Triggers:
  - `v[0-9]*.[0-9]*.[0-9]*`
  - `[0-9]*.[0-9]*.[0-9]*`
- Recommended tag: `v0.1.0-beta.1`
- Plain tag supported: `0.1.0-beta.1`
- Release gate runs before GitHub Release creation:
  - `uv sync --locked`
  - `uv run ruff check src tests scripts`
  - `uv run ruff format --check src tests scripts`
  - `uv run pyright`
  - `uv run pytest -v`
- Release body is extracted from `CHANGELOG.md` by `scripts/extract_release_notes.py`.
- Missing changelog section fails before release creation.
- Beta/rc tags are marked as GitHub prereleases.
- No wheel, sdist, binary artifact, or PyPI publish is performed.

## Release Metadata

- Changelog section: `## [0.1.0-beta.1] - 2026-07-10`
- Package metadata version: `0.1.0b1`
- Generated title: `Latent Anything 0.1.0-beta.1 - Core latent-space framework beta`
- Generated prerelease flag: `true`
- Generated body file: `artifacts/release_notes_0.1.0-beta.1.md`

## Demo Coverage

Release-quality demo coverage exists for:

- Layer A: PCA, UMAP, SAE
- Adapters: VAE, RandomProjection, HiddenStateAdapter, GaussianRendererAdapter
- Geometry: unit-norm spherical and Gaussian-set
- Layer B: Lerp, SteeringVector, ActivationPatch
- Composition: showcase, AnalysisPipeline, ManipulationPipeline
- Runtime: cache, batch executor, async/profiling

Primary release index: `artifacts/release_demo_index_0.1.0-beta.1.md`

## Probe And Visualization Decision

- No shipped probe, `LinearProbe`, or TCAV implementation exists in this beta.
- Probe/TCAV is explicitly documented as future work in README and changelog.
- Static matplotlib/text artifacts are sufficient for this beta.
- Interactive Plotly/notebook widgets and dashboard-style visualization are future work.

## Architecture And SRP Decision

No architecture/SRP issue is release-blocking.

Post-beta refactor candidates:

- `pipeline.py`: revisit after another runtime/pipeline execution story.
- `latent_space.py`: revisit after a fourth geometry of a different philosophy.
- `gaussian_renderer.py`: revisit after another renderer/backend appears.

No no-behavior refactor was performed because it would add release risk without improving the beta contract.

## Theory Coverage

The release is theory-informed, not theory-complete.

Shipped coverage includes core primitives, selected representation adapters, selected geometry-aware operations, first manipulation methods, registry/config, pipelines, and runtime helpers.

Future work explicitly includes probing/TCAV, planning, rollout, discrete latents, full world-model/VLA integrations, and interactive visualization.

Reference matrix: `artifacts/release_theory_coverage_matrix_0.1.0-beta.1.md`

## Final Gate Results

- `uv sync --locked`: passed
- `uv run ruff check src tests scripts`: passed
- `uv run ruff format --check src tests scripts`: passed
- `uv run pyright`: passed with 0 errors, 0 warnings, 0 informations
- `uv run pytest`: passed with 601 tests and 9 existing UMAP warnings
- `uv run python scripts/extract_release_notes.py v0.1.0-beta.1 --body-file artifacts/release_notes_0.1.0-beta.1.md`: passed

## Exact Tag Command

```bash
git tag v0.1.0-beta.1
git push origin v0.1.0-beta.1
```

The pushed tag will run the release workflow and create a GitHub prerelease if the gate passes in GitHub Actions.
