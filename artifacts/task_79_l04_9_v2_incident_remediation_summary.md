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

## Follow-up: a205ca7 remote fixture-path incident

The first owner-authorized real Stage A attempt on source
`a205ca7f0f4714c045027094208804c479a85445` reached the CUDA host and passed
transport decode, but the remote CLI failed before model/adapter/integration
execution because the wrapper exported a caller-local Windows train-fixture
path into the Linux checkout. The CLI exited `1`, bundle status was `66`, and
both remote and transport cleanup markers passed. No holdout, semantic
selection, resource, or finalizer result was reached, and no retry was made.

The remediation makes Stage A derive the tracked
`artifacts/m14/l04-l049-v2-train.jsonl` inside the fresh clone, rejecting
symlinks, traversal, resolution outside the clone, and untracked substitutions.
Stage B no longer requires or exports a train fixture and rejects Windows
drive/UNC/backslash, workspace-leaking, relative, and traversal paths before
bootstrap creation. Native WinPS 5.1 and pwsh fake-remote regressions cover
safe quoting and argv/env separation.

The sanitized raw-only assessment is
`artifacts/m14/l04-explanations.ssh.L049V2StageA.a205ca7f0f4714c045027094208804c479a85445.assessment.sidecar.json`.
It bound the pre-delete 6598-byte raw capture (SHA-256
`6080a35c40369c225e8611891f5403b0b53c194b065473c885ea73d58464b674`) and
records D0 status, no audit/bundle/triad, false finalization and promotion,
and the sanitized path-leakage root cause. The raw was deleted under an
explicit owner exception after exact pre-delete verification; the sidecar
records its post-delete absence. This is irreversible and is not standard
finalization or promotion.

## Follow-up: 5d6d8fb resource-peak diagnostic hardening

The owner-authorized real Stage A attempt on source
`5d6d8fb5e06890cf9615936f049681a6d1e52228` completed transport, CUDA, bundle,
and cleanup markers, then produced a D0 `runtime_exception` during finalizer
resource validation. The sanitized artifact records 2592 selection
evaluations, but no score records, folds, OOF metric, or semantic gate; those
partial evaluations are discarded and not reusable.

The finalizer checker now emits mutually exclusive, allowlisted resource-peak
subcodes for shape, primitive types, elapsed time, source/device,
status/reason, availability provenance, budget, and cross-field invariants.
It rejects arbitrary GPU source/device text and retains the independent public
validator. A committed resource-only probe uses the production
`ResourceTracker`, permits at most one minimal CUDA allocation, and emits only
fixed sanitized markers; it was not run in this local remediation.

The raw capture was 9779 bytes with SHA-256
`6dd1741c94b5af2fc084667129197304b5de2b51d023920a22164a33d342c4d2`.
The 2419-byte bundle is SHA-256
`87e063d18d0f654792af9afc9de3b3b4519a82348367c672bcc191f290543efe`.
The audit and three triad files are source-unique locally; their exact paths,
sizes, and digests are bound by
`artifacts/m14/l04-explanations.ssh.L049V2StageA.5d6d8fb5e06890cf9615936f049681a6d1e52228.assessment.sidecar.json`;
all five local evidence files were later deleted under an explicit owner
exception after exact sidecar-bound verification, and the sidecar records
their local absence. This is irreversible cleanup, not remote cleanup,
finalization, promotion, or a reusable selection.
The archive member names remain generic by design and are distinct from the
source-unique local paths. The exact peak subcondition and query asymmetry are
unknown from the sanitized artifact; no resource measurement, semantic
result, promotion, or finalization is inferred. The new canonical attempted
CUDA-device contract is a preventive hardening change, not a proven root cause
for this incident: the sanitized artifact does not identify a device/source
mismatch.

## Follow-up: 67d2fb7 resource-only probe assessment

The owner-authorized resource-only diagnostic ran once from native Windows
PowerShell through one authenticated `ssh.exe` process against exact source
SHA `67d2fb7649543ffc679e521f4f2a2ee970c55c63`. A fresh remote clone verified
that SHA, the suppressed CUDA preflight passed, and the committed
`scripts/m14_l049_v2_resource_probe.py` entry point invoked the production
`ResourceTracker` with at most one minimal CUDA allocation/synchronization.
No model, adapter, integration, fixture, dataset, selection, holdout, seed,
or Stage B input was accessed.

The raw capture was 344 bytes with SHA-256
`e812a99ccd7388a5183879d54a40e5d5f5f4913b1ce3944772cc17e24ffb49aa` and was
deleted under an explicit owner exception after exact sidecar-bound
verification. The canonical sanitized assessment sidecar
`artifacts/m14/l049-v2-resource-probe.67d2fb7649543ffc679e521f4f2a2ee970c55c63.assessment.sidecar.json`
records the deletion and local absence proof.
Its eight fixed markers independently parse as PASS, with measurement
`available`, reason `none`, finalizer code `NONE`, and coherent sanitized
provenance booleans. The sidecar separates executor-recorded exact-SHA/CUDA
facts from raw-marker facts.

The marker `L049_V2_RESOURCE_PROBE_CLEANUP=PASS` means only the internal
probe/`ResourceTracker` cleanup self-attestation. It is not evidence that the
remote disposable checkout or caches were removed. Remote checkout cleanup is
therefore recorded as `unverified`, with no remote absence proof; only local
raw absence is proven, and no semantic result, promotion, or finalization is
inferred.
