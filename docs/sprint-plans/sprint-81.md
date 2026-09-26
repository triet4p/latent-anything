# Sprint 81 Plan

## Sprint Goal

Publish `1.0.0` only after Sprint 80 proves the end-to-end representation-diagnostic depth contract, with a stable API, reproducible evidence, complete integration guidance, and an explicit post-1.0 compatibility commitment.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Confirm the Sprint 80 diagnostic-depth gate has no unresolved release blocker and every supported 1.0 claim is signed off in its evidence report.
- [x] Enforce the depth-first stop-before-release contract: no tag or publish while any supported capture, detection, localization, statistical-control, explanation, causal-validation, reporting, packaging, documentation, or workflow blocker remains.
- [x] Finalize version metadata, changelog, release notes, API reference, migration guide, plugin SDK, model/LeRobot guides, and theory coverage report.
- [x] Build wheel/sdist from a clean checkout and verify install/import/examples in clean base and optional-extra environments.
- [x] State semantic-versioning, deprecation, security-reporting, upstream-compatibility, and artifact-migration policies.
- [~] Publish signed/checksummed package artifacts and the stable GitHub/PyPI release through the audited workflow.
- [ ] Tag versioned documentation and archive benchmark/model/dataset revision manifests without redistributing restricted weights/data.
- [ ] Verify post-publication install, links, plugin discovery, and one lightweight end-to-end example.
- [ ] Mark completed sprints/milestones, publish the final artifact, and open the evidence-led post-1.0 backlog.

## Notes / Blockers

The version number follows diagnostic evidence. If a Sprint 80 depth gate fails, this sprint remains pending; the project narrows unsupported claims rather than substituting broad integration counts for diagnostic validity. External GitHub Actions access remains required for workflow-backed publication.

Task 8 was moved ahead of Task 5 with owner approval because the audited release preflight requires the stable-policy contract before it permits a tag. Task 8 passed its final evidence review (`agent://Review81Task8Final`) after the security-setting update.

Task 5 remains `[~]` pending Main's evidence review. The owner-managed App ruleset is active for `refs/tags/v*`, and the PyPI pending-publisher bootstrap is configured for first upload. The default-branch nonpublishing App-token proof passed in run 36247385348 on candidate commit `f87726950d5ae59d4287677f026c39e56baef236`. Push-triggered CI run 36247369268 passed all Python 3.12/3.13/3.14 jobs on that exact commit. The pending changelog/documentation update will create a new candidate SHA; its exact-commit CI and the guarded release-workflow checks remain required. No tag or upload has occurred.
