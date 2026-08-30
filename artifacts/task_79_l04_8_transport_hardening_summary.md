# Task 79 L04.8 Transport-Hardening Summary

## Scope

Implemented the owner-approved local transport hardening patch and recorded the
follow-up forensic diagnosis of one owner-authorized real CUDA execution. The
transport/semantic process succeeded, but retention failed closed because
non-protocol diagnostics were emitted on stdout; the fresh raw remains
preserved for the next owner-approved fix validation.

## Changes

- Added validated `SshConnectTimeoutSeconds` with default `15` and inclusive
  range `1..300` seconds.
- Forced native OpenSSH `BatchMode=yes`, `ConnectTimeout=<value>`, and
  `ConnectionAttempts=1` while preserving the existing outer
  `TransportTimeoutSeconds` deadline semantics.
- Kept the retention protocol's stdout marker/Base64-only contract by routing
  `nvidia-smi` and CLI JSON diagnostics to stderr without changing their exit
  statuses.
- Added a forensic retention finding: the embedded archive and all three
  announced member hashes remain intact despite the failed retain attempt.
- Exposed the SSH timeout, attempt count, and batch-mode settings in the
  sanitized BuildOnly manifest.
- Added exact-argv, manifest, bounds, and active-runbook URL invariant tests.
- Updated active L04 examples to derive `RepoUrl` from
  `(git remote get-url origin).Trim()`; frozen plans and historical audits were
  not changed.
- Added the required changelog, sprint status, decision, and lesson entries.

## Verification

- `uv run pytest -q tests/test_m14_l04_remote_transport.py` — 36 passed.
- `uv run pytest -q tests/test_m14_l04_*.py` (PowerShell-expanded file list) —
  264 passed, 1 skipped.
- Ruff check and format check for the changed transport test — passed.
- Pyright (repository strict configuration) for the changed transport test —
  0 errors, 0 warnings, 0 informations.
- PowerShell parser check — passed.
- Explanation contract, evidence ledger, `mkdocs build --strict`, and
  `git diff --check` — passed.
- `graphify update .` — passed; graph rebuilt successfully.

## Preserved evidence

The failed raw capture remains untracked and byte-for-byte unchanged:

- `artifacts/m14/l04-disentanglement.raw.txt`
- bytes: `195`
- SHA-256: `5a91d5534d5c47c20df4e176895734b379b342a55817e3f8feb61a325e50535b`

## Working-tree note

The approved source, test, documentation, changelog, memory, and this summary
remain uncommitted for owner review. The frozen plan remains unchanged.
