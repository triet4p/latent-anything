# Sprint 76 Plan

## Sprint Goal

Add MLflow and Weights & Biases tracking backends behind the locally proven experiment-recorder contract.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Compare the local recorder from Sprint 62 with MLflow and W&B concepts and extract only the common operations needed by real runs.
- [ ] Implement an optional MLflow backend mapping config, metrics, artifacts, tags, and parent/child runs.
- [ ] Implement an optional W&B backend with equivalent semantics and documented unavoidable differences.
- [ ] Keep both SDKs optional and prove base/LeRobot imports without them.
- [ ] Add mocked contract tests plus opt-in local integration tests; never require cloud credentials in CI.
- [ ] Verify run identity, resumability, artifact checksums, and metric-step ordering across all backends.
- [ ] Document backend selection and show the same LeRobot benchmark recorded locally and through one external backend.
- [ ] Log the recorder-contract ADR and update evidence/changelog/artifact/gates.

## Notes / Blockers

The framework records latent-specific evidence; it delegates server UI, auth, remote storage, and team workflows to established trackers.

