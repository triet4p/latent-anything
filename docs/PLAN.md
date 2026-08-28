# Global Project Plan

## Overview

Latent Anything will reach `1.0.0` by proving the framework on real models and tasks, not by accumulating attractive visualizations. The roadmap keeps the incremental Rule of Three from [INCREMENTAL.md](INCREMENTAL.md), but adds an explicit evidence contract for theory coverage, explanation validity, integration quality, and API stability.

Sprints 1-26 established the beta foundation. Sprints 27-80 are the planned path
from `0.1.0-beta.1` through the planned `0.9.0` pre-stable API-freeze
compatibility epoch to `1.0.0`. The package metadata remains `0.1.0b1` until
all Sprint 78 gates and the release workflow are verified; no `v0.9.0` tag or
publication is authorized while the external GitHub Actions account blocker is
open. Future sprint details are provisional: a sprint may revise later plans
when running code disproves an assumption, but it must record that revision in
the ADR log and keep this file synchronized.

Sprint 78's API-freeze checkpoint is recorded in the owner decision and
[`task_78.40_summary.md`](../artifacts/task_78.40_summary.md): the documented
205-runtime/202-canonical surface now requires reviewed compatibility handling,
but this is not a release-readiness claim. Metadata remains `0.1.0b1`, aliases
remain retained, and version/tag/publication stay blocked by the evidence and
workflow gates.

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
- [x] **Milestone 7 - Stable contract and semantic API (Sprints 27-31):** measurable 1.0 gates, domain vocabulary, structured latent values, geometry decomposition, and registry migration.
- [x] **Milestone 8 - Real generative-model proof (Sprints 32-38):** implementation, Sprint 34 held-out ConvVAE evidence, and Sprint 35 revision-pinned fidelity/interpolation evidence are complete as bounded local CPU D2 lanes.
- [x] **Milestone 9 - Meaningful introspection (Sprints 39-47):** real transformer inspection, probes, concepts, clustering, density/OOD, attribution, SAE evaluation, and interactive exploration.
- [x] **Milestone 10 - Geometry, trajectory, and 3D depth (Sprints 48-55):** anisotropic/Riemannian/Lie-group operations, temporal comparison/segmentation, and a real 3D Gaussian backend.
- [x] **Milestone 11 - LeRobot and VLA bridge (Sprints 56-62):** optional extra, dataset bridge, ACT/Diffusion/SmolVLA policy capture, causal simulation benchmark, and run recording.
- [x] **Milestone 12 - World models and planning (Sprints 63-72):** implementation increments and the bounded evidence/governance remediation closure are complete. Current evidence remains synthetic CPU for the compact world-model lanes; early tokenized rollout failure is recorded rather than hidden.
- [x] **Milestone 13 - Ecosystem and runtime hardening (Sprints 73-77):** Sprints 73-76 and Sprint 77 Phase A are complete; Sprint 77 Phase B recorded the owner-approved Rust/PyO3 deferral, passed the closure gates, completed the cumulative audit, and reconciled typed-ledger traceability.
- [ ] **Milestone 14 - API freeze and stable release (Sprints 78-80):** planning contract is recorded in [M14_REAL_SYSTEM_VALIDATION](M14_REAL_SYSTEM_VALIDATION.md); Sprint 79 L04.1 is design-frozen in the machine-readable [L04 plan](../artifacts/m14/l04-explanations.plan.json), while implementation, real-system evidence, release-candidate gates, and `1.0.0` publication remain pending.

## Carryover Evidence Gates

- [x] Close the Sprint 35 real, revision-pinned checkpoint fidelity evidence before declaring the related Milestone 8 claims complete.
- [x] Close the Sprint 35 real-checkpoint interpolation artifact before declaring the related Milestone 8 claims complete.
- [x] Close the Sprint 34 held-out meaningful-integration benchmark before declaring Milestone 8 complete.

These items do not reopen already delivered implementation. They keep the evidence status honest and prevent downstream real-model claims from being built on unverified prerequisites.

## Active Carryover Work

- [Sprint 34](sprint-plans/sprint-34.md) - *Status: Complete for the offline held-out meaningful-integration gate; Sprint 35 carryover evidence is complete.*
- [Sprint 35](sprint-plans/sprint-35.md) - *Status: Complete for the cached revision-pinned fidelity and interpolation D2 evidence lanes; claims remain local CPU and non-perceptual.*

The Sprint 34 carryover closure uses the deterministic sklearn-digits dataset
with an 80/20 index split, fits the ConvVAE only on the training partition,
and evaluates held-out reconstruction against both an all-zero baseline and a
training-pixel-mean diagnostic. The hard acceptance gate requires finite
metrics, at least 10% improvement over the all-zero baseline, non-degenerate
latent utilization, and successful held-out PCA/SAE/steering composition;
runtime is recorded against a 30-second CPU budget but is not used to hide a
failed quality result. The stronger train-mean baseline remains a reported
diagnostic and is not silently replaced.

Sprint 35 fidelity and interpolation acceptance was predeclared before
implementation and is now closed for the cached snapshot: use only
`stabilityai/sd-vae-ft-mse` at revision
`31f26fdeee1355a5c34592e401dd41e45d25a493`, record repository/revision/hash,
file-size, model-card license, and locked Diffusers/safetensors/Torch
versions, and compare the direct `AutoencoderKL` backend with
`DiffusersAutoencoderKLAdapter` on identical deterministic NCHW inputs. The
lane must define mean versus posterior-sample semantics and seeded RNG,
scaling-factor handling, latent/decode tolerances, shape/dtype/device/finiteness
checks, local-only safetensors loading with no remote code or network, and
bounded CPU runtime/RAM. The direct and adapter paths achieved exact parity
under these checks with zero network attempts, and the separate interpolation
lane preserved ordered endpoints with non-degenerate latent/decoded movement.
These are bounded local CPU D2 evidence lanes, not perceptual-quality or
general diffusion-pipeline claims.

## Active Sprints

- [Sprint 73](sprint-plans/sprint-73.md) - *Status: Complete; delivery and bounded audit-remediation closure finished.*
- [Sprint 74](sprint-plans/sprint-74.md) - *Status: Complete; bounded post-closure remediation of Arrow decoding, state fidelity, path safety, cache coherence, and traceability finished.*
- [Sprint 75](sprint-plans/sprint-75.md) - *Status: Complete; bounded-memory rollout streaming delivery and post-closure async/boundedness remediation are validated.*

- [Sprint 76](sprint-plans/sprint-76.md) - *Status: Complete; final lexical URI/Windows-path, exact provider-ID resume, provider atomicity, offline network, private SDK seam, and evidence-count remediation passed the supported gates.*
- [Sprint 77](sprint-plans/sprint-77.md) - *Status: Complete for Phase A/B; Rust/PyO3 is deferred with conditional reconsideration criteria, closure gates pass, the cumulative audit found no implementation blocker, and typed-ledger traceability is complete.*

Sprint 73 delivery and its audit-remediation closure are complete. Sprint 74 delivery and bounded post-closure remediation are complete. Sprint 75 delivery and its post-closure audit remediation are complete. Sprint 76 delivery and final post-audit remediation are complete. Sprint 77 Phase A/B are complete for the supported scope, including the owner-approved Rust/PyO3 deferral, closure gates, cumulative audit, and typed-ledger traceability. The Sprint 34 and Sprint 35 carryover evidence gates are complete within their declared local CPU scopes; no stable-release claim follows from these compact lanes.

Sprints 63–72 are complete as implementation increments, with the evidence limitations recorded below and in the ledger. Milestone 8 is complete for its declared bounded evidence scope; broader real-model and perceptual-quality claims remain outside these gates.

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
- [Sprint 37](sprint-plans/sprint-37.md) - Add a concrete conditional diffusion integration with native scheduler-state capture and separate denoiser-activation semantics.
- [Sprint 38](sprint-plans/sprint-38.md) - Test selective scheduler-latent intervention against paired controls and predeclared promotion thresholds.

### Milestone 9 - Meaningful introspection

- [Sprint 39](sprint-plans/sprint-39.md) - Add a real decoder-only transformer integration and direct logit lens as the hidden-state foundation for later analyses.
- [Sprint 40](sprint-plans/sprint-40.md) - Add a label-aware linear classification probe with controlled evaluation on real VAE and transformer representations.
- [Sprint 41](sprint-plans/sprint-41.md) - Add a bounded nonlinear classification probe with capacity and memorization controls.
- [Sprint 42](sprint-plans/sprint-42.md) - Add concept activation vectors and target-specific TCAV sensitivity with statistical and intervention controls.
- [Sprint 43](sprint-plans/sprint-43.md) - Add K-means structure discovery with geometry, uncertainty, and stability diagnostics.
- [Sprint 44](sprint-plans/sprint-44.md) - Add representation-bound Gaussian-mixture density and calibrated out-of-distribution scoring.
- [Sprint 45](sprint-plans/sprint-45.md) - Add activation-space Integrated Gradients on the real transformer seam with completeness and sensitivity checks.
- [Sprint 46](sprint-plans/sprint-46.md) - Add SAE feature-quality evaluation and a feature atlas artifact.
- [Sprint 47](sprint-plans/sprint-47.md) - Add interactive Plotly/notebook exploration backed by typed analysis results.

### Milestone 10 - Geometry, trajectory, and 3D depth

- [x] [Sprint 48](sprint-plans/sprint-48.md) - Add anisotropic geometry and Mahalanobis-aware operations.
- [x] [Sprint 49](sprint-plans/sprint-49.md) - Add latent projection, removal, and arithmetic with coordinate-system checks.
- [x] [Sprint 50](sprint-plans/sprint-50.md) - Add density-aware or pullback-metric geodesic interpolation.
- [x] [Sprint 51](sprint-plans/sprint-51.md) - Add SO(3)/SE(3) pose geometry and valid interpolation.
- [x] [Sprint 52](sprint-plans/sprint-52.md) - Add DTW trajectory similarity for unequal-length sequences.
- [x] [Sprint 53](sprint-plans/sprint-53.md) - Add trajectory smoothing and change-point segmentation.
- [x] [Sprint 54](sprint-plans/sprint-54.md) - Integrate a real 3D Gaussian splatting renderer backend.
- [x] [Sprint 55](sprint-plans/sprint-55.md) - Validate 3D Gaussian manipulation with multi-view rendering metrics.

### Milestone 11 - LeRobot and VLA bridge

- [Sprint 56](sprint-plans/sprint-56.md) - Ship the `latent_anything[lerobot]` dependency boundary and compatibility smoke tests.
- [Sprint 57](sprint-plans/sprint-57.md) - Bridge LeRobotDataset v3 episodes and streaming samples into typed latent inputs.
- [Sprint 58](sprint-plans/sprint-58.md) - Capture and analyze ACT policy representations.
- [Sprint 59](sprint-plans/sprint-59.md) - Capture and analyze LeRobot Diffusion Policy representations.
- [Sprint 60](sprint-plans/sprint-60.md) - Capture and intervene on SmolVLA representations.
- [Sprint 61](sprint-plans/sprint-61.md) - Run a causal policy-explanation benchmark through LeRobot simulation evaluation.

### Milestone 12 - World models and planning

- [x] [Sprint 63](sprint-plans/sprint-63.md) - Add a deterministic latent transition instance.
- [x] [Sprint 64](sprint-plans/sprint-64.md) - Add a stochastic Gaussian transition instance.
- [x] [Sprint 65](sprint-plans/sprint-65.md) - Add an RSSM-style third transition and extract the proven transition contract.
- [x] [Sprint 66](sprint-plans/sprint-66.md) - Add a rollout pipeline as Pipeline #3 and decompose pipeline responsibilities from evidence.
- [x] [Sprint 67](sprint-plans/sprint-67.md) - Add reward/value evaluation over imagined trajectories.
- [x] [Sprint 68](sprint-plans/sprint-68.md) - Add CEM planning over latent rollouts.
- [x] [Sprint 69](sprint-plans/sprint-69.md) - Add MPPI planning for continuous control.
- [x] [Sprint 70](sprint-plans/sprint-70.md) - Add a VQ/discrete-latent model adapter with codebook health metrics.
- [x] [Sprint 71](sprint-plans/sprint-71.md) - Add a JEPA/LeWM-style decoder-free world-model adapter.
- [x] [Sprint 72](sprint-plans/sprint-72.md) - Validate tokenized world-model prediction and rollout from encoded image observations.

### Milestone 13 - Ecosystem and runtime hardening

- [x] [Sprint 73](sprint-plans/sprint-73.md) - Add external plugin discovery and a separately installed hello-world plugin.
- [x] [Sprint 74](sprint-plans/sprint-74.md) - Add Arrow-backed portable artifacts and a coherent disk cache; complete bounded post-closure remediation.
- [x] [Sprint 75](sprint-plans/sprint-75.md) - Add bounded-memory streaming execution with ordered state carry and async cancellation.
- [x] [Sprint 76](sprint-plans/sprint-76.md) - Add MLflow and Weights & Biases tracking backends behind a small recorder contract; close final post-audit remediation.
- [x] [Sprint 77](sprint-plans/sprint-77.md) - Run Phase-A performance gates, record the evidence-based Rust/PyO3 deferral, and complete the bounded closure audit.

### Milestone 14 - API freeze and stable release

- [Sprint 78](sprint-plans/sprint-78.md) - Record the API-freeze checkpoint after the exhaustive inventory, SRP audit, compatibility snapshots, migration/API docs, and docs-conflict cleanup; retain aliases and keep release authorization pending in [M14_REAL_SYSTEM_VALIDATION](M14_REAL_SYSTEM_VALIDATION.md).
- [Sprint 79](sprint-plans/sprint-79.md) - Run the 24-lane real-system release-candidate matrix and the exhaustive [theory evidence-gap plan](EVIDENCE_GAP_PLAN.md), including every export/registry/plugin/profile group and the explicit non-API/backlog lanes.
- [Sprint 80](sprint-plans/sprint-80.md) - Publish `1.0.0` only after signed evidence, clean packaging, workflow/account clearance, and the stop-before-release gate.

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
- [Sprint 33](sprint-plans/sprint-33.md) - Optional dependency boundaries and resolver smoke matrix.
- [Sprint 34](sprint-plans/sprint-34.md) - ConvVAE implementation, real-image smoke evidence, and the held-out meaningful-integration benchmark are delivered as compact D2 CPU evidence.
- [Sprint 35](sprint-plans/sprint-35.md) - Diffusers `AutoencoderKL` adapter with revision-pinned fidelity and ordered latent interpolation D2 artifacts delivered under local CPU bounds.
- [Sprint 36](sprint-plans/sprint-36.md) - Control-aware VAE explanation validity benchmark.
- [Sprint 37](sprint-plans/sprint-37.md) - Conditional diffusion scheduler-state and denoiser-activation capture.
- [Sprint 38](sprint-plans/sprint-38.md) - Selective scheduler-latent intervention benchmark.
- [Sprint 39](sprint-plans/sprint-39.md) - Real decoder-only transformer integration and direct logit lens.
- [Sprint 40](sprint-plans/sprint-40.md) - Label-aware linear classification probe on real representations.
- [Sprint 41](sprint-plans/sprint-41.md) - Bounded nonlinear classification probe.
- [Sprint 42](sprint-plans/sprint-42.md) - Concept activation vectors and target-specific TCAV intervention controls.
- [Sprint 43](sprint-plans/sprint-43.md) - K-means structure discovery with geometry and stability diagnostics.
- [Sprint 44](sprint-plans/sprint-44.md) - Representation-bound density estimation and calibrated OOD scoring.
- [Sprint 45](sprint-plans/sprint-45.md) - Activation-space Integrated Gradients with completeness and transformer sensitivity evidence.
- [Sprint 46](sprint-plans/sprint-46.md) - Sparse-autoencoder feature evaluation with reconstruction, sparsity, stability, cross-check, and a queryable feature atlas.
- [Sprint 47](sprint-plans/sprint-47.md) - Interactive Plotly/notebook exploration backed by typed analysis results (renderer inputs, 2D/3D explorer, widget path, optional `viz` extra, and a digits ConvVAE walkthrough).
- [Sprint 48](sprint-plans/sprint-48.md) - Anisotropic Gaussian geometry with a fitted covariance metric, Mahalanobis distance, whitening/inverse transforms, declared interpolation semantics, provenance-bound serialization, and a Euclidean-vs-Mahalanobis neighbors/OOD benchmark.
- [Sprint 49](sprint-plans/sprint-49.md) - Orthonormal subspace projection and concept removal with an identity-bound, origin-tagged basis; latent arithmetic gated on a proven shared coordinate system; and D2 benchmarks for concept removal and basis non-interchangeability.
- [Sprint 50](sprint-plans/sprint-50.md) - Density-aware geodesic interpolation with bounded optimization, diagnostics, caching, and reconstruction evidence.
- [Sprint 51](sprint-plans/sprint-51.md) - Matrix-backed SO(3)/SE(3) pose geometry, valid group interpolation, pose trajectories, explicit frame metadata, and controlled interpolation evidence.
- [Sprint 52](sprint-plans/sprint-52.md) - Geometry-aware dynamic time warping for unequal-length trajectories.
- [Sprint 53](sprint-plans/sprint-53.md) - Geometry-aware trajectory smoothing and change-point segmentation.
- [Sprint 54](sprint-plans/sprint-54.md) - Optional real 3D Gaussian splatting renderer backend with deterministic CPU fixtures and GPU evidence.
- [Sprint 55](sprint-plans/sprint-55.md) - Constrained 3D Gaussian manipulation with held-out multi-view metrics and deterministic failure evidence.
- [Sprint 56](sprint-plans/sprint-56.md) - LeRobot 0.6.x optional dependency boundary, lazy raw seams, bridge-owned results, and CPU compatibility smoke tests.
- [Sprint 57](sprint-plans/sprint-57.md) - LeRobot v3 schema/episode descriptors, lazy episode reads, bounded streaming samples, alignment provenance, and explicit captured-latent conversion.
- [Sprint 58](sprint-plans/sprint-58.md) - Pinned ACT policy/dataset capture through LeRobot's official factories, decoder-query representations, observational analysis controls, and an opt-in public checkpoint smoke.
- [Sprint 59](sprint-plans/sprint-59.md) - Pinned Diffusion Policy capture through LeRobot's official factories, explicit conditioning/denoising axes, observational outcome analysis, and an opt-in public checkpoint smoke.
- [Sprint 60](sprint-plans/sprint-60.md) - Pinned SmolVLA capture through LeRobot's official factories, vision/language/state/action-expert representation seams with token metadata, one bounded strength-controlled action-expert intervention with bit-exact identity, quantitative change/drift/sensitivity measurements, and a marked CUDA checkpoint lane.
- [x] [Sprint 61](sprint-plans/sprint-61.md) - Causal policy-explanation benchmark through LIBERO simulation evaluation: four predeclared conditions (no-hook control, zero-strength baseline, random, targeted), seeded episode replay with fixed noise, success/return/action-deviation/latency metrics with Wilson intervals, offline-explanation-to-environment correlation with declared disagreement rules, an offline fixture suite plus a marked CUDA statistical lane, and a D3 promotion gated on the acceptance checks.
- [x] [Sprint 62](sprint-plans/sprint-62.md) - Versioned local run records, atomic content-addressed artifacts, LeRobot evidence recording, inspection/replay/comparison CLI commands, and the frozen local recorder contract.
- [x] [Sprint 63](sprint-plans/sprint-63.md) - Deterministic latent transition, recursive rollout, horizon drift metrics, immutable trajectories, and seeded synthetic evidence.
- [x] [Sprint 64](sprint-plans/sprint-64.md) - Stochastic Gaussian transition, predictive uncertainty, calibration, and seeded particle rollout evidence.
- [x] [Sprint 65](sprint-plans/sprint-65.md) - RSSM-style recurrent stochastic transition, masked temporal evaluation, stateful checkpointing, and the minimal three-instance transition contract.
- [x] [Sprint 66](sprint-plans/sprint-66.md) - Rollout Pipeline #3, story-specific pipeline modules, the shared metadata contract, cache/profiling/async parity, and compatibility migration evidence.
- [x] [Sprint 67](sprint-plans/sprint-67.md) - Reward/value scoring, masked discounted returns, held-out calibration and Bellman diagnostics, imagined-trajectory bias comparison, and run-record evidence.
- [x] [Sprint 68](sprint-plans/sprint-68.md) - Bounded CEM planning over latent rollouts, controlled random/fixed baselines, model-bias comparison, and reproducible run-record evidence.
- [x] [Sprint 69](sprint-plans/sprint-69.md) - MPPI planning with stable temperature weighting, receding-horizon execution, CEM/random comparison, smoothness/latency/robustness metrics, and reproducible run-record evidence.
- [x] [Sprint 70](sprint-plans/sprint-70.md) - Compact VQ/discrete-latent adapter, codebook health diagnostics, and explicit collapse evidence.
- [x] [Sprint 71](sprint-plans/sprint-71.md) - Compact decoder-free JEPA/LeWM-style prediction, target-encoder health diagnostics, rollout/record integration, and marked public I-JEPA checkpoint smoke.
- [x] [Sprint 72](sprint-plans/sprint-72.md) - Tokenized world-model next-token prediction, seeded rollout, codebook/likelihood/drift metrics, and reproducible synthetic evidence.

## Planning Rules

- Every sprint adds one primary evidence-bearing concern. Supporting tests, docs, artifacts, and migrations belong to that concern.
- Every sprint ends with the Rule of Three check, ADR reconciliation, evidence-ledger update, changelog update for user-visible changes, and the strict project gate.
- Real integrations must pin a tested upstream version range and include an import-isolation test so optional extras do not burden the base package.
- A model demo must include quantitative acceptance criteria and a failure analysis. Screenshot-only success is insufficient.
- Evidence promotion is conditional on predeclared thresholds. A failed threshold keeps the claim at its prior tier and must produce a negative-result or counterexample artifact rather than being promoted by judgment after the run.
- Registry kinds are semantic taxonomy, not automatic execution protocols. New analyses must not be forced through `Method` or `AnalysisPipeline` when their inputs, fitting lifecycle, or result semantics differ.
- Later sprint files are planning hypotheses. When upstream APIs or running code invalidate one, update the affected sprint files and this plan before implementation continues.
