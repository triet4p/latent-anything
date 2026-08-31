# Sprint 79 L04.9 v2 incident remediation

## Scope

This local atomic task records the explicit owner-exception cleanup of the
identified incident evidence without network, SSH, CUDA, holdout, retry, or
promotion.

## Changes

- Stage A retains bounded live operation counters and valid hook state when a
  finalizer result is rejected; invalid resource peaks become explicit
  unavailable measurements rather than stale observed zeros.
- Finalizer rejection is represented by one canonical producer-side checker
  shared by acceptance and diagnosis. It emits only an allowlisted category
  for top-level shape, identity, counters, hook/intervention, cleanup,
  resource-peak, or cross-field failures; arbitrary keys, values, traceback,
  and exception text never cross the boundary.
- Stage A's real-runtime regression executes the injected production closure
  after live selection and verifies a single finalizer call, nonzero live
  counters, canonical serialization, and validator-clean semantic D0/D1
  output. A late artifact failure strips the already-consumed closure before
  constructing its fallback, preventing a second side-effectful call. Stage B
  exercises the shared tracker envelope and idempotent finish path.
- The transport seam canonicalizes the case-insensitive UseCase binding, then
  uses a host-wide `Global\\` named OS mutex keyed by canonical stage and exact
  lowercase source SHA. On ACL-capable runtimes it grants only Synchronize and
  Modify to the current user and LocalSystem; modern runtimes explicitly clear
  session/user-only options. The mutex is acquired before process creation and
  released after process wait, capture, and raw postprocessing. Its optional
  metadata is observational only, atomically replaced, and never used to decide
  ownership or delete a contender's guard.
- Historical retention coverage binds to the exact recorded evidence paths,
  preserving absence guarantees without generic `attempt1` collisions.

## Current incident assessment

The canonical sanitized assessment is
`artifacts/m14/l04-explanations.ssh.L049V2StageA.13bf46e7b748f6fa64bf5f44cd80c194d1ca889d.incident-assessment.sidecar.json`.
It records one completed payload, two reported SSH launches, an aborted
second launch with uncertain remote reachability, D0 status, and false
promotion/finalization. The five current raw/audit/triad files were verified
against their exact recorded sizes and SHA-256 values, then deleted by explicit
owner exception. Their hashes, sizes, the bundle digest metadata, and
pre/post absence proof remain in the sidecar; this is not standard retention
finalization and cannot be recovered from Git.

## Verification

Focused Stage A and transport validation pass, including canonical UseCase,
Global-mutex concurrent-launch, and lock-release coverage on pwsh 7 and native
Windows PowerShell 5.1. A production-wrapper regression with an isolated fake
`ssh.exe` also proves mixed-case contenders share one canonical key and launch
exactly one child. The current incident's five evidence paths are now absent,
and the canonical sidecar/test assertions verify the exact owner-exception
record. The full L04 family and repository gates pass locally; no remote
invocation, retry, holdout access, promotion, or standard finalization was
performed. Graphify was updated after the source and test changes, with its
pre-existing zero-node JSON warning retained.
