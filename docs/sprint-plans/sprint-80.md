# Sprint 80 Plan

## Sprint Goal

Publish `1.0.0` with a stable API, reproducible evidence, complete integration guidance, and an explicit post-1.0 compatibility commitment.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Confirm the Sprint 79 RC has no unresolved release blocker and all stable gates are signed off in the evidence report.
- [ ] Enforce the [M14 stop-before-release contract](../M14_REAL_SYSTEM_VALIDATION.md): no tag or publish while any model/license/access, evidence threshold, docs conflict, SRP, workflow/account, or named-3DGS blocker remains.
- [ ] Finalize version metadata, changelog, release notes, API reference, migration guide, plugin SDK, model/LeRobot guides, and theory coverage report.
- [ ] Build wheel/sdist from a clean checkout and verify install/import/examples in clean base and optional-extra environments.
- [ ] Publish signed/checksummed package artifacts and the stable GitHub/PyPI release through the audited workflow.
- [ ] Tag versioned documentation and archive benchmark/model/dataset revision manifests without redistributing restricted weights/data.
- [ ] Verify post-publication install, links, plugin discovery, and one lightweight end-to-end example.
- [ ] State semantic-versioning, deprecation, security-reporting, upstream-compatibility, and artifact-migration policies.
- [ ] Mark completed sprints/milestones, publish the final artifact, and open the evidence-led post-1.0 backlog.

## Notes / Blockers

The version number follows evidence. If a stable gate fails, this sprint remains pending; the project does not lower the gate to meet a date. External GitHub Actions account access is a prerequisite for workflow-backed verification and release; absence is a blocker, not a waiver.
