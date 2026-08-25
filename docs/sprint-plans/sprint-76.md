# Sprint 76 Plan

## Sprint Goal

Add MLflow and Weights & Biases tracking backends behind the locally proven experiment-recorder contract.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Compare the local recorder from Sprint 62 with MLflow and W&B concepts and extract only the common operations needed by real runs.
- [x] Implement an optional MLflow backend mapping config, metrics, artifacts, tags, and parent/child runs.
- [x] Implement an optional W&B backend with equivalent semantics and documented unavoidable differences.
- [x] Keep both SDKs optional and prove base/LeRobot imports without them.
- [x] Add mocked contract tests plus opt-in local integration tests; never require cloud credentials in CI.
- [x] Verify run identity, resumability, artifact checksums, and metric-step ordering across all backends.
- [x] Document backend selection and show the same offline CPU fixture recorded locally and through both external adapter contracts.
- [x] Run Sprint 76 closure gates and record the final optional-SDK skips and residual provider limitations.

## Notes / Blockers

The framework records latent-specific evidence; it delegates server UI, auth, remote storage, and team workflows to established trackers. External resume is accepted only when adapter identity persists and the provider returns the requested run ID; otherwise the adapter fails closed rather than treating a reused or newly-created provider ID as continuation.

## Post-closure remediation (completed after technical audit)

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Validate local resume inputs against the stored identity across the full provenance matrix.
- [x] Harden canonical local MLflow roots and artifact names against URI, UNC, encoded, drive, symlink, junction, and reparse traversal.
- [x] Add fail-closed bounded artifact reads and canonical, privacy-safe provenance/config/tag validation.
- [x] Remove public provider-object escape hatches and enforce cumulative external payload bounds with normalized contract errors.
- [x] Strengthen real local/offline provider evidence with independent checksums, retrieval, resume/parent coverage, network-denial, and cleanup assertions.
- [x] Reconcile closure commands, docs, evidence ledgers, ADR/lesson records, and API snapshots with the hardened contract.
- [x] Run the full supported Sprint 76 closure gates and perform the final security/Rule-of-Three/worktree audit.

## Final post-audit remediation

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Harden MLflow lexical root validation for both string and `Path` inputs, with cross-platform regressions.
- [x] Make W&B resume fail closed when stored provenance is missing, empty, or malformed.
- [x] Commit provider state only after successful SDK operations and prove rollback/retry behavior.
- [x] Strengthen the real W&B offline lane with socket-level network denial and cleanup assertions.
- [x] Reconcile all focused validation counts and closure evidence with the current test suite.

## Final technical re-audit remediation

Status legend: [ ] pending / [~] in progress / [x] done

- [x] Enforce exact provider-ID continuity and fail-closed cleanup for MLflow and W&B resume.
- [x] Accept only unambiguous absolute Windows drive-path strings alongside `Path`/file-URI roots.
- [x] Hide SDK injection from public recorder constructors while retaining an internal test seam.
- [x] Reconcile final evidence, documentation, artifacts, and closure gates after the re-audit fixes.
