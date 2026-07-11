# Sprint 73 Plan

## Sprint Goal

Add external plugin discovery and prove it with a separately installed hello-world plugin package.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Define Python entry-point groups for canonical adapter, analysis, transformation/intervention, transition, and planner kinds actually present by this sprint.
- [ ] Implement lazy discovery, deterministic ordering, duplicate-name policy, version metadata, and actionable load failures.
- [ ] Keep built-in registration behavior compatible while removing import-time coupling where possible.
- [ ] Create a separate minimal plugin fixture/package that is installed during integration tests and modifies no core source.
- [ ] Prove config construction, execution, listing, and reproducibility metadata for the external plugin.
- [ ] Add plugin API/version compatibility checks and clear unsupported-contract errors.
- [ ] Publish a plugin author guide, template, and test harness.
- [ ] Log the plugin ADR and update evidence/changelog/artifact/gates.

## Notes / Blockers

The plugin surface should expose only contracts already proven by concrete built-ins. Entry points are discovery, not a second dependency-injection framework.

