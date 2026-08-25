# Sprint 73 Plan

## Sprint Goal

Add external plugin discovery and prove it with a separately installed hello-world plugin package.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Define Python entry-point groups for canonical adapter, analysis, transformation/intervention, transition, and planner kinds actually present by this sprint.
- [x] Define entry-point metadata and the group-to-existing-registry-kind mapping without loading plugin code.
- [x] Implement lazy discovery, deterministic ordering, duplicate-name policy, and actionable isolated load failures.
- [x] Keep built-in registration behavior compatible while removing import-time coupling where possible.
- [x] Create a separate minimal plugin fixture/package that is installed during integration tests and modifies no core source.
- [x] Prove config construction, execution, listing, and reproducibility metadata for the external plugin.
- [x] Add plugin API/version compatibility checks and clear unsupported-contract errors.
- [x] Publish a plugin author guide, template, and test harness.
- [x] Log the plugin ADR and update evidence/changelog/artifact/gates.

## Audit Remediation Closure (2026-08-25)

The original Sprint 73 delivery is complete. These bounded follow-up tasks
close findings from the owner-requested implementation audit before Sprint 74
may begin.

- [x] Make duplicate ordering deterministic when declarations have identical group/name/value metadata, including a provider-order regression test.
- [x] Make the separately installed fixture proof explicitly offline and fail closed if build requirements are unavailable locally.
- [x] Assert entry-point value and plugin API version in the real installed-fixture subprocess proof.
- [x] Add missing API-marker coverage and verify actionable isolation of incompatible plugins.
- [x] Align final author guidance and discovery artifact wording with the distribution-aware ordering contract.

Remediation closure completed on 2026-08-25. At that Sprint 73 closure
snapshot, Sprint 74 had not started; Sprint 74 was subsequently completed,
as recorded in [PLAN.md](../PLAN.md) and its closure artifact.

## Notes / Blockers

The plugin surface should expose only contracts already proven by concrete built-ins. Entry points are discovery, not a second dependency-injection framework.
