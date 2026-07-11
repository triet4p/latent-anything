# Sprint 33 Plan

## Sprint Goal

Define optional dependency boundaries so real integrations can grow without making the base `latent-anything` install import or resolve every ML ecosystem.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Define extras for the first real integrations, including at least `diffusers`, `transformers`, `3d`, and `lerobot`, with bounded compatible versions.
- [ ] Move integration imports behind focused modules and actionable missing-extra errors.
- [ ] Add base-install tests proving `import latent_anything` works when every optional backend is absent.
- [ ] Add resolver/installation smoke jobs for each extra on the supported Python matrix where upstream allows it.
- [ ] Define upstream version pins, lower-bound tests, and an upgrade policy for fast-moving integrations.
- [ ] Document CPU-only, GPU-required, large-download, and network-required test markers.
- [ ] Add integration fixture/cache policy that avoids uncontrolled model downloads in unit tests.
- [ ] Record the dependency ADR, evidence/changelog/artifact, and strict base gate.

## Notes / Blockers

LeRobot currently has a materially heavier dependency surface than the base package. The optional-extra boundary is a stable-product requirement, not packaging polish.

