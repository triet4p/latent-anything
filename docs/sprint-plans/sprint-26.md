# Sprint 26 Plan

## Sprint Goal

Prepare the `0.1.0-beta` release with a real tag-driven GitHub Release workflow, an honest beta scope statement, and a release-readiness audit across demos, visualization, architecture/SRP, and theory coverage.

This sprint is a release-preparation sprint, not a broad feature sprint. New features are allowed only when the audit shows that the beta claim would be misleading without them.

## Release Position

Recommended tag shape: `v0.1.0-beta.1`.

The release workflow should also tolerate plain semver tags such as `0.1.0-beta.1`, but the documented path should prefer the `v` prefix because it keeps release tags visually distinct from theory deployment tags like `theory-v*`.

`0.1.0-beta` should be positioned as:

- A working pre-1.0 framework core for latent-space primitives, adapters, Layer A dimensionality reduction/sparse decomposition, Layer B manipulation, registry/config, concrete pipelines, and first Layer C runtime helpers.
- Not a claim that the full Latent-Anything thesis is implemented.
- Not an API-stability promise beyond normal `0.x` SemVer expectations.

## Release Questions

These questions must be answered explicitly before tagging:

1. Do the current demos show enough of Layer A and Layer B to justify the beta title?
2. Is there a credible probe demo, or should probing/TCAV be called out as future work instead of implied as shipped?
3. Is visualization good enough for a beta release, or is it still mostly script-level static matplotlib output?
4. Are the main architecture seams stable enough for beta despite large files/classes, especially `pipeline.py`, `latent_space.py`, and model/method classes that mix orchestration, validation, and execution?
5. Which parts of `docs/THEORY.md` are actually represented in code, which are represented only by research notes/notebooks, and which remain future work?

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Task 1: Produce a release surface inventory for `0.1.0-beta` covering public exports, built-in registry entries, pipelines, runtime helpers, demos, artifacts, README, and changelog state.
- [x] Task 2: Add a tag-driven release workflow at `.github/workflows/release.yml`. Trigger on release tags such as `v0.1.0-beta.1` and `0.1.0-beta.1`; run the release gate first; create a GitHub Release without binary artifacts; mark beta/rc tags as prereleases.
- [x] Task 3: Define the release title and description contract. The workflow must produce a complete title and body, not an empty auto-generated release. The body should be extracted from the matching `CHANGELOG.md` section for the pushed tag after normalizing an optional leading `v` (for example `v0.1.0-beta.1` → `## [0.1.0-beta.1] - <date>`). It should include headline capability summary, demo/artifact links, install/test notes, known limitations, and theory coverage caveats.
- [x] Task 4: Audit Layer A/Layer B demo readiness. Run or inspect the existing PCA, UMAP, SAE, VAE, RandomProjection, spherical geometry, Lerp, SteeringVector, ActivationPatch, showcase, pipeline, manipulation, Gaussian, cache, batch, and async demos; record which ones are release-quality and which are smoke-only.
- [x] Task 5: Decide the probe/analysis gap for beta. If no probe demo exists, either add the smallest honest `LinearProbe`/probe-demo increment with tests and artifact, or explicitly scope probing/TCAV as future work in release notes and README.
- [x] Task 6: Audit visualization readiness. Record whether current visualization is sufficient for beta, whether static matplotlib artifacts are enough, and whether interactive Plotly/notebook widgets should be deferred. Add a release demo index if existing artifacts are hard to discover.
- [x] Task 7: Audit architecture and SRP risk. Identify files/classes carrying multiple responsibilities, especially pipeline orchestration/config/async/profiling in one file and geometry validation/dispatch in one class. Classify each as beta-acceptable, release-blocking, or post-beta refactor.
- [x] Task 8: Perform only low-risk no-behavior refactors that the SRP audit marks as release-blocking. Do not split large modules merely for tidiness if that would risk destabilizing the beta.
- [x] Task 9: Produce a theory coverage matrix mapping `docs/THEORY.md` layers to shipped code, demos, docs-only coverage, and future work. Use this to avoid overclaiming in the release description.
- [x] Task 10: Update README and CHANGELOG for beta readiness. README should show the shortest working install/import/demo path and the honest beta scope. CHANGELOG should cut `[0.1.0-beta.1] - <date>` only when the release gate is clean.
- [x] Task 11: Run the full release gate: `uv sync --locked`, `uv run ruff check src tests scripts`, `uv run ruff format --check src tests scripts`, `uv run pyright`, `uv run pytest`, and any release workflow validation that can run locally.
- [x] Task 12: Create a final release readiness artifact summarizing demo coverage, architecture/SRP decision, theory coverage, release workflow behavior, gate results, and the exact tag command to use.

## Initial Assessment

Current code appears strong enough for a beta focused on core primitives, dimensionality reduction, sparse decomposition, manipulation, registry/config, pipelines, and runtime helpers.

Current code does not yet appear strong enough to claim the full Layer A scope from theory. In particular, probing/TCAV, clustering, feature attribution, trajectory similarity, rollout, planning, discrete latent adapters, and interactive visualization should either be added in future increments or named clearly as not shipped in this beta.

Architecture is acceptable for a beta only if the release notes are honest that APIs remain pre-1.0 and if Sprint 26 records an SRP/refactor backlog. The current large-file shape is not automatically a release blocker, but it is a warning sign: future work should split responsibilities once the next concrete pipeline/runtime stories reveal stable seams.

## Release Workflow Acceptance Criteria

- Tag push to `v0.1.0-beta.1` creates a GitHub Release after the gate passes.
- Tag push to `0.1.0-beta.1` is also supported or deliberately rejected with documentation; the documented recommended tag remains `v0.1.0-beta.1`.
- Release title is explicit, for example: `Latent Anything 0.1.0-beta.1 - Core latent-space framework beta`. It may be generated from the tag plus a small workflow-side suffix, or read from optional changelog metadata if Sprint 26 adds it.
- Release body is populated from the matching `CHANGELOG.md` version section, not left blank. The workflow must fail if a pushed release tag has no matching changelog section.
- Beta/rc tags create prereleases; stable tags create normal releases.
- No wheel, sdist, binary, or PyPI publish is required for this sprint unless the release audit explicitly adds it.

## Notes / Blockers

* This sprint may conclude that one small probe demo is necessary before beta, but that decision must be made after the demo audit rather than assumed upfront.
* The release must not imply that theory layers 6-9 are implemented merely because the theory docs cover them.
* Refactoring for Single Responsibility should be evidence-led. The project rule remains: extract stable seams from working code, not from aesthetic discomfort alone.
