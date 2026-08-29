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
  production entry after validation and by offline fake-process tests.
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

## Validation

The focused offline tests and PowerShell parser/build-only checks are required
before owner review. Full pytest, Ruff/Pyright, strict docs, and Graphify update
remain part of the owner gate. No commit, push, SSH, CUDA, model, or network
operation is performed by this task.

## Risks / owner gates

Remote execution still requires authenticated Windows PowerShell, a committed
exact SHA, and owner review of the raw capture and bundle markers. The payload
currently routes all supported canonical use cases through the existing L04
fixture CLI; exact remote Bash/find/tar behavior and lane-specific real
execution remain owner-gated Phase B steps.
