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

Task 8 was moved ahead of Task 5 with owner approval because the audited release preflight requires the stable-policy contract before it permits a tag. Task 8 passed its final evidence review (`agent://Review81Task8Final`) after the security-setting update. Task 5's evidence reviews remain FAIL: the owner-managed App ruleset is active for `refs/tags/v*`, and PyPI's pending-publisher bootstrap is configured for first upload, but nonpublishing App-token proof, release-candidate CI, and publication remain open. No tag or upload has occurred.
