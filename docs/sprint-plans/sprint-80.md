# Sprint 80 Plan

## Sprint Goal

Deliver and prove one trustworthy representation-diagnostic workflow for ordinary deep-learning models: capture internal representations, detect and localize defects, test explanations with statistical controls, validate causality through intervention, compare runs, and emit a reproducible engineering report.

## Product Claim

An AI engineer can start with a supported PyTorch model plus data and obtain a defensible diagnosis that states what failed, where it failed, how strong the evidence is, whether intervention supports the explanation, what changed between runs, and what remains unknown.

The supported core is proved on one encoder/autoencoder and one transformer hidden-state case. SmolVLA is a secondary, non-blocking portability lane under the permanent 16 GiB ceiling. Model-family count and historical theory-coverage percentages are not stable-release gates.

## Entry State

- `0.9.0` is published and verified through the GitHub Release asset contract.
- `main` contains the depth-first 1.0 contract and the Sprint 80 / Sprint 81 split.
- Historical Sprint 79 evidence remains immutable; negative, partial, blocked, and excluded results are not promoted.
- Sprint 80 is implementation and evidence work. Sprint 81 owns `1.0.0` publication.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

### Phase A — Frozen diagnostic contracts

- [ ] **80.1 — Freeze the representation-problem taxonomy.** Add one versioned machine-readable taxonomy for collapse/rank loss, anisotropy/inactive dimensions, redundancy/superposition, separability/probe leakage, density/OOD/distribution drift, sparse-feature instability, and sequence/trajectory drift where applicable. **Accept:** stable identifiers, applicability rules, required evidence, and unsupported-case behavior validate without model-specific code.
- [ ] **80.2 — Freeze the benchmark-manifest schema.** Define one schema for model and revision, dataset and split, layer/slice axes, injected or known defect, metrics, seeds, controls, uncertainty, causal expectation, and pass/fail thresholds. **Accept:** malformed, post-hoc, or incomplete manifests fail closed.
- [ ] **80.3 — Freeze the diagnostic-report schema.** Define versioned machine-readable fields for capture provenance, symptoms, localization, hypotheses, statistical evidence, interventions, comparisons, limitations, and next action. **Accept:** the schema distinguishes observation, explanation, causal result, and unsupported conclusion.
- [ ] **80.4 — Implement the report validator.** Validate taxonomy references, evidence completeness, control outcomes, artifact hashes, and conclusion strength independently of report construction. **Accept:** missing controls, mismatched provenance, and overclaimed conclusions are rejected.
- [ ] **80.5 — Predeclare the two core benchmark manifests.** Pin one encoder/autoencoder case and one transformer hidden-state case before workflow evidence is generated. **Accept:** revisions, data splits, defect/counterexample, thresholds, controls, expected localization, and causal success/falsification rules are immutable inputs.

### Phase B — High-level workflow substrate

- [ ] **80.6 — Define the high-level diagnostic request/result API.** Introduce the smallest public configuration and result types needed to select captures, diagnostics, controls, interventions, comparisons, and output location. **Accept:** the API contains domain terms, no architecture-specific fields, and no duplicate configuration path.
- [ ] **80.7 — Generalize capture selection and axis metadata.** Bind requested modules/layers and sample, slice, token, time, or checkpoint axes to existing capture, `LatentValue`, and `Trajectory` primitives. **Accept:** deterministic capture identity and complete provenance are preserved for both core models.
- [ ] **80.8 — Implement the diagnostic workflow state machine.** Orchestrate capture → detect → localize → explain → intervene → compare → report without embedding method-specific algorithms in the coordinator. **Accept:** stages have explicit inputs/outputs, failure states, and resumable artifact boundaries.

### Phase C — Detection families

- [ ] **80.9 — Integrate collapse, rank-loss, anisotropy, and inactive-dimension detection.** Reuse existing geometry/statistics primitives behind the taxonomy contract. **Accept:** positive/counterexample and negative controls separate real defects from benign low variance.
- [ ] **80.10 — Integrate redundancy, superposition, separability, and probe-leakage detection.** **Accept:** leakage-safe splits, capacity controls, label randomization, and non-separable negatives prevent probe accuracy from becoming a diagnosis by itself.
- [ ] **80.11 — Integrate density, OOD, and distribution-drift detection.** **Accept:** reference/test identity, calibration, uncertainty, shuffled controls, and distribution-free failure behavior are recorded.
- [ ] **80.12 — Integrate sparse-feature and temporal-drift detection.** Reuse SAE/dictionary and trajectory primitives only where the representation exposes those axes. **Accept:** cross-seed feature stability and sequence controls gate the claim; non-applicable cases are omitted rather than passed.

### Phase D — Localization and explanatory validity

- [ ] **80.13 — Localize findings across layers and dataset slices.** **Accept:** the workflow identifies the earliest affected layer and affected sample/slice under predeclared correctness criteria, not only a global score.
- [ ] **80.14 — Localize findings across checkpoint, token, and time axes.** **Accept:** applicable axes preserve alignment and identity; absent axes are explicit non-applicability, not empty success.
- [ ] **80.15 — Implement the statistical-control executor.** Centralize seeded repetitions, confidence intervals, null, shuffled, randomized, and cross-seed controls. **Accept:** failed controls block the associated diagnostic conclusion.
- [ ] **80.16 — Integrate probe, TCAV, and Integrated-Gradients explanation evidence.** **Accept:** each method runs only for a declared hypothesis and reports fidelity, stability, selectivity, leakage checks, and uncertainty.
- [ ] **80.17 — Integrate SAE, lens, geometry, density, and clustering explanation evidence.** **Accept:** feature labels or projections are never promoted without frozen stability/selectivity criteria and a declared relationship to the diagnosed symptom.

### Phase E — Causality, comparison, and reporting

- [ ] **80.18 — Implement activation patching, ablation, and removal trials.** **Accept:** identity/zero-strength, random, shuffled, and off-target controls share the same captured inputs and downstream metric.
- [ ] **80.19 — Implement steering and dose-response trials.** **Accept:** intervention strength, direction provenance, monotonic/non-monotonic response, random-direction behavior, off-target effects, and downstream task change are recorded.
- [ ] **80.20 — Implement aligned checkpoint/run comparison.** **Accept:** comparisons require identical dataset slices, preprocessing, layer identity, seeds, and diagnostic configuration, and distinguish representation change from task-metric change.
- [ ] **80.21 — Persist the content-addressed diagnostic artifact.** Reuse run-record and portable-artifact conventions. **Accept:** schema version, hashes, inputs, environment, seeds, intermediate stages, and validator result reproduce without hidden local state.
- [ ] **80.22 — Render the concise AI-engineer report.** **Accept:** the report names symptom, location, evidence strength, causal support or falsification, limitations, and next engineering action; unsupported statements fail closed.

### Phase F — Core proof

- [ ] **80.23 — Prove the encoder/autoencoder diagnosis end to end.** Execute the frozen manifest through the single high-level workflow. **Accept:** detect → localize → explain → intervene → compare → report passes with the declared positive/counterexample and negative controls.
- [ ] **80.24 — Prove the transformer diagnosis end to end.** Execute the frozen leakage-safe transformer manifest through the same workflow. **Accept:** the same chain passes without a transformer-only reporting or orchestration path.
- [ ] **80.25 — Reproduce both core cases from a clean environment.** **Accept:** pinned inputs regenerate validator-clean run records and content-addressed artifacts with bounded runtime/resource evidence.
- [ ] **80.26 — Publish the AI-engineer guide and executable examples.** **Accept:** a user starts from model plus data and reproduces both diagnoses without manually assembling low-level capture, statistics, intervention, comparison, or artifact plumbing.

### Phase G — Secondary proof and handoff

- [ ] **80.27 — Resolve the bounded SmolVLA lane.** Verify checkpoint, dataset, license, and 16 GiB execution feasibility before running one secondary diagnostic proof. **Accept:** either a truthful bounded artifact passes or an explicit blocker remains; neither outcome gates the ordinary-DL core.
- [ ] **80.28 — Run proportionate quality and compatibility gates.** Use focused checks per atomic task; run full lint, type, test, packaging, documentation, clean-environment, and compatibility gates only for the final code/release candidate. Docs/evidence-only changes use focused validation and `[skip ci]` where appropriate.
- [ ] **80.29 — Publish the Sprint 80 depth-evidence report.** Map every supported claim to the frozen manifests, reports, controls, interventions, clean reproductions, and review artifacts. **Accept:** negative results and excluded breadth lanes remain explicit, and the report gives Sprint 81 a binary readiness decision for the supported core.

## Execution Order

1. Complete Phase A before generating benchmark evidence.
2. Complete the shared workflow substrate before detector- or model-specific integration.
3. Evidence-review each atomic task or explicitly tagged consecutive batch before advancing.
4. The encoder and transformer proofs may run in parallel only after their shared contracts and workflow dependencies pass.
5. Run the sprint-wide deep review only after all 29 task evidence gates pass.

Each completed task records a concise artifact summary under `artifacts/`, exact validation evidence, affected claims, negative results, and its evidence-review verdict.

## Stable-Depth Acceptance Gates

Sprint 80 passes only when all of the following are true:

1. **Capture:** supported core models expose deterministic, provenance-bound internal representations through the documented workflow.
2. **Detection:** each claimed representation-problem family has a non-trivial positive/counterexample benchmark and a negative/control case.
3. **Localization:** reports identify affected layers and relevant sample, slice, checkpoint, token, or time axes with predeclared correctness criteria.
4. **Statistical validity:** applicable results include uncertainty and leakage-safe randomized, shuffled, null, and cross-seed controls; a failed control blocks the claim.
5. **Explanatory validity:** projections and probes are not explanations by themselves; headline explanations pass frozen fidelity, stability, and selectivity gates.
6. **Causal validity:** at least one intervention confirms or falsifies each headline diagnosis with identity/zero-strength, dose-response, random, and off-target behavior where applicable.
7. **Reproducibility:** both core cases reproduce from a clean environment with pinned inputs and validator-clean, content-addressed artifacts.
8. **Usability:** both cases use the same high-level workflow and end in an actionable report that states what failed, where, evidence strength, causal result, limitations, and missing evidence.

## Non-Goals

- Maximizing historical D2/D3 theory-row percentages.
- Supporting every adapter, VLA, 3D representation, world model, or planner before 1.0.
- Treating visualization, import, checkpoint loading, or smoke execution as diagnostic evidence.
- Lowering thresholds or replacing failed controls after observing results.
- Requiring hardware above the approved 16 GiB release-evidence ceiling.
- Publishing `1.0.0`; Sprint 81 owns publication after the depth-evidence handoff passes.

## Notes / Blockers

Historical Sprint 79 evidence remains immutable. Existing TCAV significance, SAE cross-seed stability, steering randomized-control, and selected manifold/geometry failures remain blockers for only the corresponding diagnostic claims. The workflow must either resolve them under the frozen Sprint 80 contracts or omit those claims.

The first execution task is **80.1**. No product implementation begins before tasks 80.1-80.5 freeze the taxonomy, schemas, and core benchmark manifests.
