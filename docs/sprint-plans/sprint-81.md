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
- [x] Publish signed/checksummed package artifacts and the stable GitHub/PyPI release through the audited workflow.
- [x] Tag versioned documentation and archive benchmark/model/dataset revision manifests without redistributing restricted weights/data.
- [x] Verify post-publication install, links, plugin discovery, and one lightweight end-to-end example.
- [~] Mark completed sprints/milestones, publish the final artifact, and open the evidence-led post-1.0 backlog.

## Notes / Blockers

The version number follows diagnostic evidence. If a Sprint 80 depth gate fails, this sprint remains pending; the project narrows unsupported claims rather than substituting broad integration counts for diagnostic validity. External GitHub Actions access remains required for workflow-backed publication.

Task 8 was moved ahead of Task 5 with owner approval because the audited release preflight requires the stable-policy contract before it permits a tag. Task 8 passed its final evidence review (`agent://Review81Task8Final`) after the security-setting update.

Task 5 passed its final evidence review (`agent://Review81Task5Final`). CI run 36250365395 passed on exact release commit `a449ca33b4b83ce109c29db7cf471919820b1d56`; the nonpublishing App-token proof passed in run 36247385348. Audited release run 36251907256 published protected tag `v1.0.0`, the GitHub Release with five assets and verified attestations, and the first PyPI wheel/sdist through OIDC. PyPI project JSON is HTTP 200. Follow-up commit `d5bc328ac850f68db3240b85a1bc01c072e31402` reconciled the active-PyPI gate metadata and present-tense documentation without altering the tag or release assets. Task 7 later passed its evidence review (`agent://Review81Task7Initial`); Task 9 remains `[~]` pending final evidence review and the final deep-review gate.

Task 6 passed its evidence review (`agent://Review81Task6Tags`). The authoritative metadata-only revision archive and versioned documentation are publicly available at immutable `docs-v1.0.0-r1` (commit `3cbc925de464a090bc1d65a2fc8858a11a0dcbf3`); the earlier `docs-v1.0.0` snapshot remains immutable but is superseded because one manifest digest used checkout line endings instead of release Git blob bytes. The package release tag and its five assets are unchanged. Package documentation is versioned at GitHub source URLs, not GitHub Pages; the latter remains the separate theory site.

Task 7 passed its evidence review (`agent://Review81Task7Initial`). A clean public-PyPI wheel install, matching GitHub/PyPI distribution hashes, corrected versioned-document links, separately installed external-plugin behavior, and a pinned CPU encoder diagnostic with independent report validation all passed. The immutable PyPI 1.0.0 long description and `docs-v1.0.0-r1` plugin guides retain pre-release wording; current-source corrections and the limitation must be preserved in the final handoff rather than misrepresented as edits to those immutable snapshots.

Task 9's final release-evidence handoff is in [`artifacts/sprint-81/task-9.md`](../../artifacts/sprint-81/task-9.md). It records the exact public release/CI/provenance and PyPI hashes, bounded claims, immutable-metadata caveat, and prioritized backlog. Task 9 remains `[~]` pending Main's evidence review; Sprint 81 and Milestone 14 remain open until the final deep-review gate passes. Main owns the final `[x]` status transition after that gate.
