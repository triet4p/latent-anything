# Sprint 79 L04 reusable remote transport prerequisite

## Scope

Implemented the reusable Windows PowerShell/native `ssh.exe` transport boundary
and separate remote Bash payload for L04 real-model lanes. This is Phase A:
local/offline only; it does not claim remote Bash/find/tar execution, remote
CUDA execution, or semantic evidence. Phase B starts only after commit/push and
owner approval from authenticated Windows PowerShell.

## Files

- `scripts/m14_l04_remote_transport.ps1`: exact-byte LF/UTF-8/no-BOM payload
  normalization, SHA-256 and Base64 bootstrap construction, decoded-byte remote
  integrity gates, direct `ProcessStartInfo` invocation, asynchronous capture,
  raw-before-parse persistence, and sanitized build-only manifest.
- `scripts/_m14_l04_transport_seam.psm1`: internal-only generic
  `ProcessStartInfo`/stdin/async-stream/raw-capture lifecycle seam used by the
  production entry after validation and by offline fake-process tests. The seam
  now uses one monotonic deadline across process start, async stdin, process
  wait, stream drain, and raw publication; timeout termination kills the whole
  process tree with a bounded 30-second grace and retains best-effort evidence.
- `scripts/m14_l04_remote_payload.sh`: disposable exact-SHA clone, isolated UV/
  Hugging Face caches, pinned dependency/CUDA preflight, one canonical L04 CLI
  invocation, bundle-before-cleanup, and full-workdir cleanup.
- `tests/test_m14_l04_remote_transport.py`: offline build-only and static
  contract coverage, including CR normalization, hashes, redaction, invalid
  parameter rejection, and forbidden-shell checks.
- `docs/M14_REAL_SYSTEM_VALIDATION.md`, `docs/EVIDENCE_GAP_PLAN.md`, and
  `docs/sprint-plans/sprint-79.md`: authoritative replacement workflow and
  prerequisite status. The frozen M14 plan remains unchanged.
- `.agents/memory/decisions.md` and `.agents/memory/lessons-learned.md`: durable
  transport boundary decision and historical failure lesson.

The public helper exposes `-TransportTimeoutSeconds` with a default of 3600 and
a validated range of 2400–7200 seconds. This transport budget covers setup,
the semantic invocation, bundle creation, and cleanup; the semantic protocol's
1800-second cap remains independent. The payload emits a sanitized absolute
`L04_WORKDIR` marker immediately after `mktemp`; cleanup is reported only when
the remote trap verifies removal. The build-only manifest's expected-marker set
includes `L04_WORKDIR` alongside the transport, status, bundle, and cleanup
markers.

## Validation

The focused transport suite passes (`29 passed`), including production timeout
default/min/max validation, non-reading stdin, hanging process, child-held
stdout/stderr, bounded kill/drain, raw-before-parse, and stale-target cases.
The full suite passes (`1840 passed, 36 skipped, 39 warnings`). PowerShell
parser, Ruff, Pyright, strict MkDocs, `git diff --check`, and final Graphify
update all pass. No SSH, CUDA, model, or network operation is performed by
this hardening task.

## Risks / owner gates

Remote execution still requires authenticated Windows PowerShell, a committed
exact SHA, and owner review of the raw capture and bundle markers. The payload
currently routes all supported canonical use cases through the existing L04
fixture CLI; exact remote Bash/find/tar behavior and lane-specific real
execution remain owner-gated Phase B steps. The single L04.8 Phase B attempt at
the pushed SHA timed out during dependency setup before the CLI, bundle, or
remote cleanup marker; its sanitized D0 audit is retained separately and must
not be retried implicitly.
