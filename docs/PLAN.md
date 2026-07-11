# Global Project Plan

## Overview

Latent Anything will reach `1.0.0` by proving the framework on real models and tasks, not by accumulating attractive visualizations. The roadmap keeps the incremental Rule of Three from [INCREMENTAL.md](INCREMENTAL.md), but adds an explicit evidence contract for theory coverage, explanation validity, integration quality, and API stability.

Sprints 1-26 established the beta foundation. Sprints 27-80 are the planned path from `0.1.0-beta.1` to `1.0.0`. Future sprint details are provisional: a sprint may revise later plans when running code disproves an assumption, but it must record that revision in the ADR log and keep this file synchronized.

## Definition of Stable

`1.0.0` is allowed only when all gates below pass.

1. **Theory evidence:** at least 95% of implementation-applicable topics in core tiers 1-9, and at least 90% across the complete theory index, have code or benchmark evidence. Research notes and notebooks alone do not count toward this gate.
2. **Meaningful explanations:** every headline explanation method has fidelity, stability, selectivity/control, and causal-intervention evidence where applicable. A visually clean projection is not accepted as an explanation by itself.
3. **Real-model matrix:** the supported matrix includes a real VAE, a pretrained generative-image model, a transformer hidden-state model, a structured 3D representation, at least two LeRobot policies including one VLA, and a temporal/world-model path.
4. **LeRobot bridge:** `latent_anything[lerobot]` can consume LeRobotDataset data, capture policy representations, apply supported interventions, and return metrics through LeRobot evaluation without reimplementing LeRobot policies, datasets, or environments.
5. **API quality:** public names describe domain behavior rather than roadmap layers; entry classes delegate feature logic to focused modules; deprecated beta aliases have a documented migration path; the public signature and serialization compatibility suites pass.
6. **Extension quality:** optional integrations are isolated, external plugins work through Python entry points, a hello-world plugin is documented, and plugin/config artifacts are versioned and reproducible.
7. **Release quality:** strict lint/type/test gates pass, core coverage meets the threshold set in Sprint 27, integration tests are version-pinned, documentation builds strictly, and the package is publishable from a clean environment.

## Coverage Vocabulary

Each theory-led capability has one of four evidence levels:

- **D0 - documented:** research note or notebook only; does not count as implemented coverage.
- **D1 - implemented:** production code and focused tests exist.
- **D2 - validated:** D1 plus an end-to-end benchmark on non-trivial data with quantitative acceptance criteria.
- **D3 - model-proven:** D2 plus evidence on at least one real pretrained/trained model and a reproducible artifact.

The `1.0.0` percentage gate counts D2 or D3 only. Headline model and integration claims require D3.

## Milestones

- [x] **Milestones 0-6 - Beta foundation (Sprints 1-26):** theory research, core primitives, first adapters/methods, registry/config, concrete pipelines, runtime helpers, and `0.1.0-beta.1` release.
- [ ] **Milestone 7 - Stable contract and semantic API (Sprints 27-31):** measurable 1.0 gates, domain vocabulary, structured latent values, geometry decomposition, and registry migration.
- [ ] **Milestone 8 - Real generative-model proof (Sprints 32-38):** activation capture, optional backend isolation, real VAE and diffusion integrations, and explanation-validity benchmarks.
- [ ] **Milestone 9 - Meaningful introspection (Sprints 39-47):** probes, concepts, clustering, density/OOD, attribution, transformer inspection, SAE evaluation, and interactive exploration.
- [ ] **Milestone 10 - Geometry, trajectory, and 3D depth (Sprints 48-55):** anisotropic/Riemannian/Lie-group operations, temporal comparison/segmentation, and a real 3D Gaussian backend.
- [ ] **Milestone 11 - LeRobot and VLA bridge (Sprints 56-62):** optional extra, dataset bridge, ACT/Diffusion/SmolVLA policy capture, causal simulation benchmark, and run recording.
- [ ] **Milestone 12 - World models and planning (Sprints 63-72):** transition models, rollout, reward/value evaluation, CEM/MPPI, discrete latents, JEPA/LeWM, and tokenized-world-model evidence.
- [ ] **Milestone 13 - Ecosystem and runtime hardening (Sprints 73-77):** external plugins, portable artifacts/disk cache, streaming, tracking backends, and an evidence-based Rust decision.
- [ ] **Milestone 14 - API freeze and stable release (Sprints 78-80):** `0.9`, release-candidate matrix, and `1.0.0` publication.

## Active Sprints

None. Sprint 33 is the next sprint to activate.

## Planned Sprints

### Milestone 7 - Stable contract and semantic API

- [Sprint 27](sprint-plans/sprint-27.md) - Define the 1.0 evidence ledger, coverage denominator, and acceptance gates.
- [Sprint 28](sprint-plans/sprint-28.md) - Decide domain vocabulary and the beta-to-stable deprecation map.
- [Sprint 29](sprint-plans/sprint-29.md) - Add a first-class latent value container proven across flat and structured representations.
- [Sprint 30](sprint-plans/sprint-30.md) - Add discrete geometry and extract geometry-specific logic when the fourth case proves the seam.
- [Sprint 31](sprint-plans/sprint-31.md) - Replace `method_a`/`method_b` registry vocabulary with semantic kinds and compatibility aliases.

### Milestone 8 - Real generative-model proof

- [Sprint 32](sprint-plans/sprint-32.md) - Add safe PyTorch activation capture and intervention lifecycle.
- [Sprint 33](sprint-plans/sprint-33.md) - Isolate optional ML integrations and define the extras/version test matrix.
- [Sprint 34](sprint-plans/sprint-34.md) - Integrate and evaluate a convolutional VAE on a real image dataset.
- [Sprint 35](sprint-plans/sprint-35.md) - Add a pretrained Diffusers `AutoencoderKL` adapter.
- [Sprint 36](sprint-plans/sprint-36.md) - Validate VAE explanations with reconstruction, factor, stability, and causal metrics.
- [Sprint 37](sprint-plans/sprint-37.md) - Add a real diffusion pipeline adapter with timestep-aware latent capture.
- [Sprint 38](sprint-plans/sprint-38.md) - Prove diffusion latent editing with task metrics and decoded outputs.

### Milestone 9 - Meaningful introspection

- [Sprint 39](sprint-plans/sprint-39.md) - Add a label-aware linear probe with controlled evaluation.
- [Sprint 40](sprint-plans/sprint-40.md) - Add a nonlinear probe as an information upper bound with capacity controls.
- [Sprint 41](sprint-plans/sprint-41.md) - Add concept activation vectors and TCAV sensitivity scoring.
- [Sprint 42](sprint-plans/sprint-42.md) - Add K-means structure discovery with stability diagnostics.
- [Sprint 43](sprint-plans/sprint-43.md) - Add Gaussian-mixture density and out-of-distribution scoring.
- [Sprint 44](sprint-plans/sprint-44.md) - Add Integrated Gradients through the capture/intervention seam.
- [Sprint 45](sprint-plans/sprint-45.md) - Add a real transformer hidden-state/logit-lens integration.
- [Sprint 46](sprint-plans/sprint-46.md) - Add SAE feature-quality evaluation and a feature atlas artifact.
- [Sprint 47](sprint-plans/sprint-47.md) - Add interactive Plotly/notebook exploration backed by typed analysis results.

### Milestone 10 - Geometry, trajectory, and 3D depth

- [Sprint 48](sprint-plans/sprint-48.md) - Add anisotropic geometry and Mahalanobis-aware operations.
- [Sprint 49](sprint-plans/sprint-49.md) - Add latent projection, removal, and arithmetic with coordinate-system checks.
- [Sprint 50](sprint-plans/sprint-50.md) - Add density-aware or pullback-metric geodesic interpolation.
- [Sprint 51](sprint-plans/sprint-51.md) - Add SO(3)/SE(3) pose geometry and valid interpolation.
- [Sprint 52](sprint-plans/sprint-52.md) - Add DTW trajectory similarity for unequal-length sequences.
- [Sprint 53](sprint-plans/sprint-53.md) - Add trajectory smoothing and change-point segmentation.
- [Sprint 54](sprint-plans/sprint-54.md) - Integrate a real 3D Gaussian splatting renderer backend.
- [Sprint 55](sprint-plans/sprint-55.md) - Validate 3D Gaussian manipulation with multi-view rendering metrics.

### Milestone 11 - LeRobot and VLA bridge

- [Sprint 56](sprint-plans/sprint-56.md) - Ship the `latent_anything[lerobot]` dependency boundary and compatibility smoke tests.
- [Sprint 57](sprint-plans/sprint-57.md) - Bridge LeRobotDataset v3 episodes and streaming samples into typed latent inputs.
- [Sprint 58](sprint-plans/sprint-58.md) - Capture and analyze ACT policy representations.
- [Sprint 59](sprint-plans/sprint-59.md) - Capture and analyze LeRobot Diffusion Policy representations.
- [Sprint 60](sprint-plans/sprint-60.md) - Capture and intervene on SmolVLA representations.
- [Sprint 61](sprint-plans/sprint-61.md) - Run a causal policy-explanation benchmark through LeRobot simulation evaluation.
- [Sprint 62](sprint-plans/sprint-62.md) - Add reproducible experiment records and LeRobot-facing inspection commands.

### Milestone 12 - World models and planning

- [Sprint 63](sprint-plans/sprint-63.md) - Add a deterministic latent transition instance.
- [Sprint 64](sprint-plans/sprint-64.md) - Add a stochastic Gaussian transition instance.
- [Sprint 65](sprint-plans/sprint-65.md) - Add an RSSM-style third transition and extract the proven transition contract.
- [Sprint 66](sprint-plans/sprint-66.md) - Add a rollout pipeline as Pipeline #3 and decompose pipeline responsibilities from evidence.
- [Sprint 67](sprint-plans/sprint-67.md) - Add reward/value evaluation over imagined trajectories.
- [Sprint 68](sprint-plans/sprint-68.md) - Add CEM planning over latent rollouts.
- [Sprint 69](sprint-plans/sprint-69.md) - Add MPPI planning for continuous control.
- [Sprint 70](sprint-plans/sprint-70.md) - Add a VQ/discrete-latent model adapter with codebook health metrics.
- [Sprint 71](sprint-plans/sprint-71.md) - Add a JEPA/LeWM-style decoder-free world-model adapter.
- [Sprint 72](sprint-plans/sprint-72.md) - Validate tokenized world-model prediction and rollout.

### Milestone 13 - Ecosystem and runtime hardening

- [Sprint 73](sprint-plans/sprint-73.md) - Add external plugin discovery and a separately installed hello-world plugin.
- [Sprint 74](sprint-plans/sprint-74.md) - Add Arrow-backed portable artifacts and a coherent disk cache.
- [Sprint 75](sprint-plans/sprint-75.md) - Add bounded-memory streaming execution.
- [Sprint 76](sprint-plans/sprint-76.md) - Add MLflow and Weights & Biases tracking backends behind a small recorder contract.
- [Sprint 77](sprint-plans/sprint-77.md) - Run performance gates and make the Rust core go/no-go decision.

### Milestone 14 - API freeze and stable release

- [Sprint 78](sprint-plans/sprint-78.md) - Cut `0.9.0`, freeze the public API, and remove scheduled beta aliases.
- [Sprint 79](sprint-plans/sprint-79.md) - Run the full real-model/integration release-candidate matrix.
- [Sprint 80](sprint-plans/sprint-80.md) - Publish `1.0.0` with migration, plugin, model-integration, and reproducibility documentation.

## Completed Sprints

- [Sprints 1-2](sprint-plans/sprint-1.md) - Theory foundation and practical 3D research expansion.
- [Sprints 3-16](sprint-plans/sprint-3.md) - Core primitives, geometry cases, adapters, Layer A/B methods, and the first end-to-end showcase.
- [Sprints 17-21](sprint-plans/sprint-17.md) - Registry/config extraction and two concrete pipelines.
- [Sprints 22-24](sprint-plans/sprint-22.md) - Batch, cache, async, and profiling runtime foundation.
- [Sprint 25](sprint-plans/sprint-25.md) - Cross-sprint review corrections and strict gate restoration.
- [Sprint 26](sprint-plans/sprint-26.md) - `0.1.0-beta.1` release readiness, architecture audit, and theory coverage audit.
- [Sprint 27](sprint-plans/sprint-27.md) - Evidence-ledger contract, stable-coverage denominator, and read-only CI validation.
- [Sprint 28](sprint-plans/sprint-28.md) - Semantic API vocabulary, beta-name snapshot, and compatibility migration contract.
- [Sprint 29](sprint-plans/sprint-29.md) - Immutable `LatentValue` for flat batches and structured states.
- [Sprint 30](sprint-plans/sprint-30.md) - Discrete-code geometry and focused geometry algorithm extraction.
- [Sprint 31](sprint-plans/sprint-31.md) - Canonical semantic registry kinds and beta config migration diagnostics.
- [Sprint 32](sprint-plans/sprint-32.md) - Safe internal PyTorch activation capture and intervention lifecycle.

## Planning Rules

- Every sprint adds one primary evidence-bearing concern. Supporting tests, docs, artifacts, and migrations belong to that concern.
- Every sprint ends with the Rule of Three check, ADR reconciliation, evidence-ledger update, changelog update for user-visible changes, and the strict project gate.
- Real integrations must pin a tested upstream version range and include an import-isolation test so optional extras do not burden the base package.
- A model demo must include quantitative acceptance criteria and a failure analysis. Screenshot-only success is insufficient.
- Later sprint files are planning hypotheses. When upstream APIs or running code invalidate one, update the affected sprint files and this plan before implementation continues.
