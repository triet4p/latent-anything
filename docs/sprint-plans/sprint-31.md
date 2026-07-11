# Sprint 31 Plan

## Sprint Goal

Introduce semantic registry kinds and config vocabulary while preserving beta configurations through explicit aliases and migration diagnostics.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Implement the canonical registry kinds selected in Sprint 28 for analysis, transformation/intervention, adapter, and later runtime/planning families.
- [ ] Add legacy `method_a` and `method_b` lookup aliases with one well-scoped deprecation warning per construction path.
- [ ] Migrate built-in registrations, `ObjectSpec` validation, pipeline specs, demos, and documentation to canonical kinds.
- [ ] Add config round-trip and legacy-config migration tests, including nested specs and clear errors for ambiguous names.
- [ ] Split registry alias/migration policy from core lookup storage if the responsibility boundary is now stable.
- [ ] Update plugin-author documentation with canonical naming examples and reserved namespaces.
- [ ] Add a machine-readable migration report for repository-owned configs and confirm no silent behavior change.
- [ ] Reconcile the naming ADR, evidence ledger, changelog, sprint artifact, and strict gate.

## Notes / Blockers

This is the first code migration from the naming RFC. Legacy aliases stay until the removal version fixed in Sprint 28; they must not become permanent duplicate concepts.

