# Sprint 80 Plan

## Sprint Goal

Prove that Latent Anything provides a complete, trustworthy representation-diagnostic loop for ordinary deep-learning models: capture internal representations, detect and localize defects, form and test explanations with statistical controls, validate causality through intervention, compare runs, and emit a reproducible report. A bounded small-VLA lane may demonstrate portability, but breadth across heavyweight model families is not a stable-release gate.

## Product Claim

An AI engineer can use one documented workflow to turn model activations into a defensible representation diagnosis with quantitative evidence, uncertainty, controls, causal validation, provenance, and actionable localization.

The supported 1.0 core is ordinary PyTorch deep-learning representations, proven end to end on at least one encoder/autoencoder model and one transformer hidden-state model. SmolVLA is a secondary proof lane under the permanent 16 GiB hardware ceiling. OpenVLA, unnamed 3DGS checkpoints, broad world-model coverage, and integration count are not proxies for diagnostic depth.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Replace the breadth-led stable-release contract with a depth-first diagnostic contract in the global plan, M14 governance documents, evidence vocabulary, and ADR log without promoting or deleting historical evidence.
- [ ] Freeze a machine-readable representation-problem taxonomy covering collapse/rank loss, anisotropy and inactive dimensions, redundancy/superposition, separability and probe leakage, density/OOD and distribution drift, sparse-feature instability, and sequence/trajectory drift where applicable.
- [ ] Predeclare the diagnostic benchmark matrix: supported models, datasets and splits, issue injections or known counterexamples, metrics, uncertainty, null/random/shuffled/off-target controls, causal expectations, and pass/fail rules before implementation evidence is generated.
- [ ] Define a versioned diagnostic input and report contract that records captured layers and slices, detected symptoms, localization, hypotheses, controls, interventions, downstream effects, limitations, and complete run/artifact provenance.
- [ ] Deliver one high-level diagnostic workflow over existing capture, `LatentValue`/`Trajectory`, analysis, intervention, runtime, and artifact primitives; ordinary supported PyTorch models must not require architecture-specific glue for every layer.
- [ ] Deliver detection and localization across layer, sample or dataset slice, checkpoint/run, and token/time axes where present; diagnostics must identify where a problem begins rather than emit only a global score.
- [ ] Integrate probes, TCAV, Integrated Gradients, SAE/dictionary features, lens methods, geometry, density, and clustering only where each method answers a declared hypothesis; expose uncertainty and reject leakage, unstable seeds, degenerate inputs, and failed controls.
- [ ] Integrate activation patching, ablation or removal, and steering into the same workflow so observational findings can be tested with identity/zero-strength, random-direction, shuffled/null, dose-response, and off-target controls.
- [ ] Add checkpoint/run comparison that distinguishes representation change from task-metric change and preserves identical dataset slices, layer identity, preprocessing, seeds, and diagnostic configuration.
- [ ] Produce a durable machine-readable diagnostic artifact and a concise human report that state symptom, affected location, evidence strength, causal result, limitations, and the next engineering action; unsupported conclusions must fail closed.
- [ ] Prove the complete workflow on a real encoder/autoencoder case with a predeclared representation defect or controlled counterexample and a successful detect → localize → explain → intervene → report chain.
- [ ] Prove the complete workflow on a pinned transformer hidden-state case with leakage-safe splits and the same detect → localize → explain → intervene → report chain.
- [ ] Run one bounded SmolVLA diagnostic proof only if the pinned checkpoint, dataset, license, and 16 GiB execution path are available; otherwise retain an explicit secondary-lane blocker without blocking the ordinary-DL 1.0 core or authorizing a VLA diagnostic claim.
- [ ] Publish an AI-engineer guide and executable example that start from a model plus data and reproduce both core diagnoses without requiring users to assemble low-level capture, statistics, intervention, and artifact plumbing manually.
- [ ] Run the strict quality, packaging, documentation, compatibility, and clean-environment gates; publish a Sprint 80 depth-evidence report that lists every supported claim, negative result, excluded breadth lane, and remaining blocker for Sprint 81.

## Stable-Depth Acceptance Gates

Sprint 80 passes only when all of the following are true:

1. **Capture:** the supported core models expose deterministic, provenance-bound internal representations through the documented workflow.
2. **Detection:** every supported representation-problem family has at least one non-trivial positive or counterexample benchmark and one negative/control case; unsupported families are omitted from the 1.0 claim rather than counted by documentation coverage.
3. **Localization:** the report identifies affected layers and relevant sample, slice, checkpoint, token, or time axes with predeclared correctness criteria.
4. **Statistical validity:** applicable results include uncertainty and leakage-safe randomized, shuffled, null, and cross-seed controls; a failed control blocks that diagnostic claim.
5. **Explanatory validity:** a projection or probe alone is never called an explanation; headline explanations must meet their frozen fidelity, stability, and selectivity criteria.
6. **Causal validity:** at least one intervention confirms or falsifies each headline diagnosis, with identity/zero-strength, dose-response, random, and off-target behavior recorded where applicable.
7. **Reproducibility:** each core case can be reproduced from a clean environment using pinned inputs and emits validator-clean run records and content-addressed artifacts.
8. **Usability:** both core cases run through the same documented high-level workflow and terminate in a report that tells an AI engineer what failed, where, how strong the evidence is, and what evidence remains missing.

## Non-Goals

- Maximizing the percentage of researched theory topics with D2/D3 artifacts.
- Supporting every adapter, model family, VLA, 3D representation, world model, or planner before 1.0.
- Treating attractive visualization, successful import, checkpoint loading, or smoke execution as diagnostic evidence.
- Replacing failed controls with waivers or lowering a predeclared threshold after observing results.
- Requiring hardware above the approved 16 GiB release-evidence ceiling.

## Notes / Blockers

Historical Sprint 79 evidence remains immutable and truthfully classified. Existing TCAV significance, SAE cross-seed stability, steering randomized-control, and any selected manifold/geometry failures remain hard blockers only for the corresponding supported diagnostic claims. They are not converted into passes by this plan revision.

The old 95% core / 90% overall theory-coverage percentages and broad real-model matrix become portfolio-health metrics and backlog inputs, not 1.0 release gates. Sprint 81 owns `1.0.0` publication only after this sprint's depth-evidence report has no unresolved blocker for the supported core claims.
