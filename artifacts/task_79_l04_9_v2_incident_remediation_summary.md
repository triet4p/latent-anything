# Sprint 79 L04.9 v2 incident remediation

## Scope

This local atomic task records the local remediation and sanitized assessment
of the current b295a506 incident without network, SSH, CUDA, holdout, retry, or
promotion. The captured evidence was later deleted under the explicit owner
exception recorded below; it was never finalized or promoted.

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
- The ResourceTracker now publishes allocated and reserved CUDA peaks as one
  validated pair. Exceptions, invalid types, negative values, zero values, and
  asymmetric pairs clear both values and publish an allowlisted unavailable
  reason. Stage A normalization derives hook/intervention projections only
  from the complete live operation-count snapshot.
- v2 retained triad files use source-unique local names keyed by the full
  lowercase source SHA, while the audit keeps the original generic archive
  member names in `archive_member_names`. The current b295a506 audit was
rebuilt from the prior 2638-byte/SHA `c9fdfcfa5f60a010c403e143062ee6f7a9820f588308b7f13912240093456fd4`
to 3276 bytes/SHA `914d374748e803513d3aeba04c556fa17584ebdabd3efdad4dcde6bf26e43a91`
to make this distinction; raw, bundle, and triad bytes/digests are unchanged.

## Current incident assessment

The canonical sanitized assessment is
`artifacts/m14/l04-explanations.ssh.L049V2StageA.b295a506933e18f6d9139b0439f0e80d6ed441e8.assessment.sidecar.json`.
It binds one completed payload and one SSH/CLI invocation, D0 status, false
promotion/finalization, the proven stale projection bug, and an explicitly
unknown GPU subfield because the raw capture does not expose which CUDA peak
query was asymmetric. Selection evaluation count 2592 is discarded and not
reusable. Raw, bundle, and relocated triad bytes/digests remain unchanged and
pending before the explicit owner-exception cleanup; the audit was rebuilt and
is bound by its new exact size/SHA-256. The
bundle member names remain generic only inside the archived bundle metadata.

The earlier 13bf incident remains separately recorded as an explicit
owner-exception deletion and is not standard retention finalization.

## Verification

Focused Stage A, ResourceTracker, and retention validation cover the atomic
CUDA-pair and source-unique-path contracts. The earlier transport and mutex
gates remain passing from the reviewed b295a506 baseline. The five exact
sidecar-bound evidence files were deleted after literal pre-delete verification;
the post-delete absence proof is recorded in the sidecar. No remote invocation,
retry, holdout access, promotion, or standard finalization was performed.
Graphify is updated after source and test changes.
