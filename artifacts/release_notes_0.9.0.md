## [0.9.0] - 2026-09-17

### Release Summary

`0.9.0` is an intentionally pre-stable API/evidence baseline for the Sprint 80
diagnostic-depth program. It freezes the broad Sprint 79 inventory as a
reproducible baseline and does **not** claim `1.0.0` diagnostic readiness,
broad VLA support, or that all M14 lanes passed.

Package metadata and runtime `latent_anything.__version__` are `0.9.0`. The
current compatibility snapshot is
`artifacts/api_freeze_snapshot_0.9.0.json` (SHA-256
`048ac553adabb11c24d3e1f4d86e0c6d469df6590064014c915d6a08c7d53c26`); the only
content change from the preserved historical beta snapshot is the
version-stamped fixture envelope (observed digest
`0007706863773ddf25bbdf97313e998c0e83250fbe936945628505b3f7da2d71`, golden digest
`815ea47a07cecc202c5312d4c9ff4de5441c9ce74d4315e647615f483ba050ea` unchanged).
The historical `0.1.0-beta.1` section, snapshots, release notes, and tag below
are preserved unchanged.

### Accepted Scope

- 205 current runtime top-level exports with a 202-entry canonical-stable
  projection; 32 built-in registry entries; 5 entry-point groups; 12 optional
  profiles; 5 CLI commands with 2 retained aliases; 28 config schemas; 9
  sync/async pairs; 7 public exceptions; versioned serialization envelopes
  (`portable-node-v1`, `result-envelope-v1` with `v0` migration,
  `artifact-envelope-v1`, `disk-cache-v1`, run-record `schema-v1`).
- All 18 beta compatibility aliases retained with exact identity/behavior;
  removal is deferred past `0.9.0` pending a separate reviewed migration
  decision. Registry `method_a`/`method_b` construction warnings now state the
  deferral explicitly.
- Depth-first planning baseline: Sprint 79 breadth inventory closed honestly,
  Sprint 80 owns the end-to-end representation-diagnostic depth contract, and
  Sprint 81 owns `1.0.0` publication.

### Negative, Partial, and Blocked Evidence (Preserved)

- Theory ledger: **41/63 core and 41/64 scoped overall** qualifying rows. These
  are portfolio-health facts, not passes; the former 95%/90% breadth gates are
  retired as release thresholds.
- M14 real-system matrix (23 applicable lanes): 13 accepted, 2 partial (L02 with
  5/6 accepted records; L04 partial), 1 pending, 7 externally/prerequisite
  blocked. Full row evidence is in `artifacts/task_79_line602_rc_evidence_report.md`.
- Explanation validity: TCAV, SAE cross-seed stability, steering
  randomized-control, and selected manifold/geometry failures are preserved as
  blockers for their corresponding Sprint 80 diagnostic claims, not as `0.9.0`
  blockers.
- Hardware ceiling: the supported release-evidence ceiling remains **16 GiB**.
  L19/OpenVLA stays a historical D0 feasibility record and is hardware-excluded
  from the active release scope because its canonical BF16 contract requires
  unavailable >=24 GiB hardware; it authorizes no capability, performance, or
  quality claim and is **excluded** from `0.9.0`.
- No named trustworthy 3DGS checkpoint exists (L17 blocked); unavailable
  real-policy and LeRobot overhead measurements are secondary integration work;
  no unsupported performance claim is made.
- Publication status: annotated tag `v0.9.0` points to merged commit
  `75341e4292fcdf1703186c58b626666b4a923c19`; Release workflow run
  `35516933525` passed the gate/build and publish jobs. The GitHub Release
  contains exactly the wheel, sdist, `SHA256SUMS`, `PROVENANCE.json`, and
  release notes; all five downloaded asset hashes and the clean local-wheel
  import smoke passed. PyPI publication is explicitly deferred and non-gating.
