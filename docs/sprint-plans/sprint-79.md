# Sprint 79 Plan

## Sprint Goal

Run the release-candidate matrix across supported Python versions, optional extras, real models, LeRobot policies, world models, plugins, tracking, and artifact compatibility.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Build clean environments for base, each optional extra, and supported combined extras on every supported Python/platform tier.
- [ ] Run unit/property/integration tests plus strict docs, packaging, security, license, and dependency audits.
- [ ] Execute every applicable row of the 24-lane [M14 real-system matrix](../M14_REAL_SYSTEM_VALIDATION.md), with one artifact per independently verifiable capability.
- [ ] Execute the exhaustive [theory evidence-gap plan](../EVIDENCE_GAP_PLAN.md) and its row-level [machine-readable map](../../artifacts/task_78.38_gap_map.json); keep D0/D1 statuses unchanged until the validator-backed D2/D3 artifacts exist.
- [ ] Use the [migration guide](../MIGRATION.md) and [API reference](../API_REFERENCE.md) as the human entry points to the checked-in compatibility snapshot during RC verification.
- [ ] Execute the pinned real-model matrix: Diffusers VAE/conditional diffusion, GPT-2, I-JEPA, VQ/tokenized/world-model paths, ACT, Diffusion Policy, and SmolVLA; record the named 3DGS checkpoint or keep L17 blocked.
- [ ] Execute explanation-validity controls and confirm the theory ledger meets 95% core / 90% overall D2-or-D3 thresholds.
- [ ] Verify all 202 exports, 32 built-in registry entries, 5 entry-point groups, 12 optional profiles, CLI commands, schema migrations, negative/security cases, sync/async paths, cross-adapter composition, external plugin install/discovery, cache, streaming, and tracking backends.
- [ ] Measure performance budgets and LeRobot policy overhead against Sprint 77 gates.
- [ ] Run remote CUDA only through the remote-cuda-test skill invariants and preserve disposable-clone/cache cleanup evidence; do not use remote CUDA as a substitute for missing local tests.
- [ ] Publish an RC evidence report with failures, waivers, confidence intervals, hardware, upstream revisions, and exact reproduction commands.
- [ ] Fix only release blockers, rerun the complete affected matrix, reconcile docs/ledger conflicts, and cut the release candidate only after the external GitHub Actions account is available.

## Notes / Blockers

No percentage waiver may hide an implementation-applicable core theory gap. Any exclusion must have been classified and justified in the Sprint 27 ledger. SAM, OpenCLIP, timm, Torchvision model adapters, Open3D, trimesh, and unnamed 3DGS are not stable API claims; they remain explicit backlog/blocker rows.

Sprint 78.38 records the current denominator and all 40 non-qualifying rows;
Sprint 79 owns execution in dependency order. The named 3DGS checkpoint and
corrected SmolVLA rerun remain explicit blockers, not evidence promotions.
