# Task Summary: Sprint 79 L04.8 recovery fix

**Sprint:** Sprint 79
**Task:** L04.8 recovery fix after `ce4e66eb5c70bee7852e07ec239415643bd74493`

## Summary of Work

Closed the local recovery defects exposed by the preserved Disentanglement
audit. The shared model-boundary reader now preserves the exact authored
`clean`/`corrupted` condition and rejects missing or invalid values. Linux
`ru_maxrss` is normalized to bytes and validated with coherent source/unit
provenance. The remote payload uses the exact NUL-safe tar order, emits CLI and
bundle statuses separately, captures tar without errexit, bundles exactly the
three current-attempt artifacts after a semantic CLI failure, and applies the
CLI-first/bundle-second exit policy. The original audit remains byte-for-byte
unchanged and remains D0; no remote rerun, commit, or promotion was performed.

## Files Modified

* `scripts/_m14_l04_tcav_runtime.py` — preserve and validate fixture conditions.
* `scripts/_m14_l04_disentanglement.py` — correct Linux RSS unit provenance.
* `scripts/_m14_l04_validate_disentanglement.py` — require `rss_unit=bytes`.
* `scripts/m14_l04_remote_payload.sh` — correct tar order and status/exit handling.
* `scripts/m14_l04_remote_transport.ps1` — declare CLI/bundle markers.
* `tests/test_m14_l04_disentanglement.py` — actual-reader, production-path, RSS,
  and mislabeled-unit regressions.
* `tests/test_m14_l04_remote_transport.py` — exact tar/status static contract.
* `docs/sprint-plans/sprint-79.md` — recovery checkpoint and D0 audit record.
* `docs/M14_REAL_SYSTEM_VALIDATION.md` — recovery procedure and immutable audit.
* `docs/EVIDENCE_GAP_PLAN.md` — D0 evidence-gap correction record.
* `.agents/memory/lessons-learned.md` — append-only transport/provenance lesson.

## Testing

* **Focused tests:** `uv run pytest -q tests/test_m14_l04_disentanglement.py tests/test_m14_l04_remote_transport.py`
* **Result at first run:** 69 passed, 1 skipped, 1 static-contract failure before
  the final `L04_CLI_STATUS` insertion.
* **Post-fix targeted result:** 4 passed, 1 skipped for reader/handler/RSS and
  tar/status regressions.
* **Full L04 result:** `uv run pytest -q tests/test_m14_l04_*.py` (PowerShell
  expanded file list) — 210 passed, 1 skipped.
* **Full repository result:** `uv run pytest -q` — 1842 passed, 37 skipped,
  39 warnings.
* **Static gates:** Ruff check/format, explicit strict-config Pyright (0 errors,
  0 warnings, 0 informations), contract check, evidence ledger, docs strict,
  and `git diff --check` passed.
* **Graphify:** `graphify update .` passed after the final code/test change;
  the latest rebuild contains 12,930 nodes, 26,202 edges, and 1,017
  communities.
* **Shell syntax:** local Bash executable is unavailable on Windows; no SSH,
  CUDA, or remote execution was attempted.

## Additional Notes

The preserved forensic audit SHA-256 is
`b7f0d54740c4a7f0dfe71eb626f4f752ce88b511369f70bafc4b3f0415930fd3`.
The audit records actual `KeyError('condition')`, CLI-only validator success,
absent bundle, unverified raw deletion, cleanup PASS markers, and no
promotion. Owner review must run the full offline gates and authorize a new
direct authenticated PowerShell `ssh.exe` execution before any D2 claim.

## Final transport follow-up

The bundle-gate helper now accepts the original CLI status. A semantic CLI
failure remains the final status even when bundling returns 66; a semantic
success with bundle status 66 returns 66. `L04_CLI_STATUS`,
`L04_BUNDLE_STATUS`, and `L04_STATUS` remain distinct markers, while setup
preflight failures retain their own status path.

The final gates included 31 transport tests, 86 focused L04 tests (one skip),
Ruff check/format, strict-config Pyright, contract and evidence-ledger checks,
strict MkDocs build, and `git diff --check`; all passed. The MkDocs temporary
directory was revalidated as an untracked normal directory (not a symlink) and
removed with native PowerShell `Remove-Item -LiteralPath -Recurse -Force`, then
verified absent. Final Graphify rebuild: 12,310 nodes, 24,349 edges, and 974
communities. No commit or remote execution was performed.
