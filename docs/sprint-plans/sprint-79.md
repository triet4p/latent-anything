# Sprint 79 Plan

## Sprint Goal

Run the release-candidate matrix across supported Python versions, optional extras, real models, LeRobot policies, world models, plugins, tracking, and artifact compatibility.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Build clean environments for base, each optional extra, and supported combined extras on every supported Python/platform tier.
- [ ] Run unit/property/integration tests plus strict docs, packaging, security, license, and dependency audits.
- [ ] Execute the pinned real-model matrix: VAE, diffusion, transformer, 3DGS, ACT, Diffusion Policy, SmolVLA, and world-model/discrete paths.
- [ ] Execute explanation-validity controls and confirm the theory ledger meets 95% core / 90% overall D2-or-D3 thresholds.
- [ ] Verify external plugin install/discovery, config migration, artifact schema migration, disk cache, streaming, and tracking backends.
- [ ] Measure performance budgets and LeRobot policy overhead against Sprint 77 gates.
- [ ] Publish an RC evidence report with failures, waivers, confidence intervals, hardware, upstream revisions, and exact reproduction commands.
- [ ] Fix only release blockers, rerun the complete affected matrix, and cut the release candidate.

## Notes / Blockers

No percentage waiver may hide an implementation-applicable core theory gap. Any exclusion must have been classified and justified in the Sprint 27 ledger.

