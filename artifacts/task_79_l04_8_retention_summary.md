# Task 79 L04.8+ Remote Evidence Retention Summary

## Scope

Implemented the local-only, fail-closed retention boundary for L04.8 and later
remote captures. The PowerShell helper remains capture-only; it does not delete
raw evidence. The postprocessor accepts an explicit raw capture, source SHA,
use-case, artifact output directory, and sanitized audit path.

## Implementation

- Added the small CLI `scripts/m14_l04_remote_postprocess.py` plus private
  parser/archive and transaction modules with `--retain`, `--validate-only`,
  `--finalize-delete`, and `--dry-run` modes.
- Parse singleton transport/CLI/bundle/status markers and exactly three
  repeatable member markers; reject duplicates, omissions, inconsistent exits,
  wrong use-case, wrong source SHA, and malformed Base64.
- Require announced bundle bytes/SHA-256 and per-member bytes/SHA-256.
- Inspect tar headers before extraction and reject traversal, absolute or
  backslash paths, links, special files, duplicate members, history, extra
  members, and mixed attempts.
- Extract only validated regular JSON members into a collision-proof temporary
  directory; run the existing artifact, run-record, failure, and use-case lane
  validators before retention.
- Install the exact three files atomically with collision rejection and exact
  byte idempotence. Reopen/re-hash/revalidate final paths before deletion.
- Write/reopen a sanitized audit containing provenance, marker exits, archive
  and member hashes, final paths/hashes, and validation results. No prompt,
  raw capture, or bundle payload is copied into the audit.
- `--retain` leaves raw evidence pending finalization after audit and payload
  reopen with mode `retained_pending_finalize`; separate `--finalize-delete`
  rejects any other mode, reparses/rebuilds the pending audit,
  atomically quarantines raw in the same directory, publishes
  `quarantined_pending_delete`, then deletes only after revalidation and
  verifies both paths absent. A quarantine publication failure reverses the
  rename (or retains the exact quarantine with a structured error). If final
  audit publication fails after quarantine deletion, the in-memory raw
  snapshot and exact pending audit are atomically restored for retry; a
  double failure publishes `raw_restore_failed` without claiming success.
- `--validate-only` and `--dry-run` perform no writes or deletes.
- Updated the Bash payload to emit bundle/member digest markers and the
  PowerShell contract to optionally invoke the postprocessor with `-Postprocess`.

## Historical evidence rule

The d9 Disentanglement audit remains immutable: semantic D2 was observed and
validators passed, but it is non-closeable because the exact payload was lost
under the earlier cleanup policy. Attempt 2 must not be reconstructed or
promoted. The sanitized retention-failure sidecar records the remote semantic
promotion claim separately from `repository_promotion=false`; a remote
`promotion=true` is not repository closure. A fresh owner-authorized real run
is required for current evidence.

## Verification

- `uv run pytest tests/test_m14_l04_remote_postprocess.py -q` — 42 passed.
- `uv run pytest tests/test_m14_l04_remote_transport.py tests/test_m14_l04_remote_postprocess.py -q` — 73 passed.
- All L04 tests (`tests/test_m14_l04_*.py`) — 259 passed, 1 skipped.
- Full repository — 1891 passed, 37 skipped.
- Ruff check and format check for changed Python files — passed.
- Strict Pyright for changed Python files — 0 errors, 0 warnings, 0 informations.
- No network, SSH, CUDA, commit, or push was performed.

## Remaining owner gates

The postprocessor is offline-tested only. A fresh owner-authorized L04.8 real
CUDA invocation must use the committed helper/payload, retain its raw capture,
and invoke `--retain` locally before any promotion decision.
