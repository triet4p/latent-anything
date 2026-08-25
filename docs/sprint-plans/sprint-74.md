# Sprint 74 Plan

## Sprint Goal

Add versioned Arrow-backed artifacts and a disk cache that restores complete behavior-affecting state coherently.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Define the Arrow-backed portable node schema for NumPy arrays, latent values, trajectories, immutable metadata, and bounded nested values.
- [x] Add safe typed-result/config/checkpoint envelopes with an explicit allowlist, migrations, and behavior-affecting state validation.
- [x] Implement versioned artifact envelopes with canonical identity, checksums, atomic writes, path/symlink safety, and bounded reads.
- [x] Add the approved custom SQLite disk cache with stable state-aware keys, coherent restoration, bounded size, and deterministic eviction.
- [x] Add corruption, partial-write, version-mismatch, allocation-guard, eviction, and concurrent-reader/writer tests.
- [x] Prove cross-process round-trips and restored behavior parity for latent values, trajectories, typed results, and cache hits.
- [x] Integrate portable artifacts with run records and Sprint 73 plugin provenance/configuration.
- [x] Measure portable/disk/in-memory size and latency on declared offline CPU workloads.
- [x] Log serialization/cache ADRs and update English docs, evidence ledgers, changelog, artifacts, and gates.

## Post-closure remediation (2026-08-25)

The initial implementation passed its recorded closure gates, but the owner
audit found bounded-decoding, exact-state-fidelity, nested-immutability,
Windows reparse safety, cache-row validation, and current-state traceability
gaps. Sprint 74 was the sole active sprint while these focused tasks were
being remediated; all remediation tasks now pass.

- [x] Bound Arrow input, manifest, rows, ranks, dtypes, and allocations before materialization; add adversarial tests.
- [x] Preserve exact tuple/list contracts for every allowlisted typed result and add CEM/MPPI/profile regressions.
- [x] Recursively freeze provenance, behavior-state, and artifact metadata with mutation tests.
- [x] Harden artifact and cache paths against symlink/junction/reparse traversal and document residual race guarantees.
- [x] Validate SQLite stored size before BLOB loading, reject oversized/corrupt rows, and require coherent cache identities/payloads.
- [x] Rewrite stale Sprint 73 closure wording as historical-at-closure text pointing to current Sprint 74 status.
- [x] Add Task 09 closure traceability to the evidence ledger and reconcile governance wording.
- [x] Mark all remediation tasks complete only after focused tests and graphify refreshes.

## Notes / Blockers

Prior cache lessons require state coherence. Serialization convenience must not reintroduce output-only caching of fitted operations.

Post-closure remediation completed on 2026-08-25. At that Sprint 74 closure
snapshot, Sprint 75 remained planned and had not started; Sprint 75
subsequently completed its bounded streaming story. At that same historical
snapshot, Sprint 76 remained planned and unopened; Sprint 76 was subsequently
completed in its own plan.
