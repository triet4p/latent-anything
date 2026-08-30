# Task 79 L04.8 Transport-Hardening Summary

## Scope

Implemented the owner-approved local transport hardening patch, corrected the
marker-only stdout violation, and closed one owner-authorized real CUDA
Disentanglement execution at exact SHA
`4d3a4b6551d6091ce96c73a704e642867c2f2580`. The semantic result is validator-
clean D2/eligible evidence; its exact triplet and sanitized audit are tracked
after `deleted_verified` raw finalization. Earlier timeout and strict-retention
failure captures remain preserved as non-promoting historical evidence.

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
  266 passed, 1 skipped.
- Ruff check and format check for the changed transport test — passed.
- Pyright (repository strict configuration) for the changed transport test —
  0 errors, 0 warnings, 0 informations.
- PowerShell parser check — passed.
- Explanation contract, evidence ledger, `mkdocs build --strict`, and
  `git diff --check` — passed.
- `graphify update .` — passed; graph rebuilt successfully.

- Final owner-authorized direct PowerShell `ssh.exe` execution at the exact
  pushed SHA completed with transport decode/cleanup PASS, semantic
  `passed_real_cuda`, D2/eligible Disentanglement evidence, and remote cleanup
  PASS. The single local `--retain` and single `--finalize-delete` operations
  both passed; payload reopen/validators passed and final raw deletion is
  recorded as `deleted_verified`.

## Current closure

- The final run's semantic status is `passed_real_cuda`, with all
  Disentanglement controls and the strict held-out gain gate passing.
- The final archive is 26241 bytes with SHA-256
  `c222475e9591eed5fbc45f6202aff7edaf83926a8fa7a6a0e85699aacce614d0`; the
  tracked triplet is reopened and validator-clean.
- The final audit is
  `artifacts/m14/l04-explanations.ssh.Disentanglement.4d3a4b6551d6091ce96c73a704e642867c2f2580.audit.json`
  with final SHA-256
  `a08f46d6da86c88948b9216637f0ed3216967efeacae9018f5c23aeedec64db2` and
  `raw_status=deleted_verified`.
- The 195-byte transport timeout and 266082-byte strict-retention-failure
  captures each have a minimal sanitized sidecar with exact raw hash/size,
  stage/reason, semantic reachability, and `repository_promotion=false`.

## Preserved evidence

The failed raw capture remains untracked and byte-for-byte unchanged:

- `artifacts/m14/l04-disentanglement.raw.txt`
- bytes: `195`
- SHA-256: `5a91d5534d5c47c20df4e176895734b379b342a55817e3f8feb61a325e50535b`

The prior 266082-byte strict-retention-failure raw remains untracked at
`artifacts/m14/l04-disentanglement.0c7cfa208bc7e9c9d4c1f848f76a0ac189735b19.attempt1.raw.txt`
with SHA-256
`176a663119a70d02a440aab4f03de2ba4a2f0f85e3c29d8d53be4bc101d205b0`; its
sanitized failure sidecar records semantic-known D2 but
`repository_promotion=false`.

## Working-tree note

The source, tests, documentation, changelog, memory, tracked final triplet and
final audit are prepared for the reviewed closure. The two earlier raw captures
remain untracked and byte-exact with their sanitized non-promoting sidecars.
The frozen plan remains unchanged.
