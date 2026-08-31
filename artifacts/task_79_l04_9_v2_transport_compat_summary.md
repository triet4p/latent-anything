# Task Summary: Sprint 79 L04.9 v2 remote transport compatibility

**Sprint:** Sprint 79
**Task:** Make the L04 remote transport compatible with native Windows PowerShell 5.1 and pwsh 7.

## Summary of Work

Updated the private process/capture seam and production PowerShell helper for
cross-version Windows execution. SHA-256 hashing now uses the .NET Framework
compatible API. Process arguments use reflected `ArgumentList` when available
and a Windows CRT quoting fallback otherwise. Raw captures use synchronous
byte writes, flush/close, and atomic same-directory publication with a unique
non-null backup for existing targets. Every nonempty stdin payload uses
deadline-bounded async write/flush with synchronous best-effort close on
failure; process termination selects compatible APIs while preserving timeout
behavior, raw-before-parse ordering, exact-once process start, and exit-code
capture.

## Files Modified

* [scripts/_m14_l04_transport_seam.psm1](../scripts/_m14_l04_transport_seam.psm1) - Cross-version process, stream, hashing, argument, and atomic raw-capture seam.
* [scripts/m14_l04_remote_transport.ps1](../scripts/m14_l04_remote_transport.ps1) - Cross-version payload hashing.
* [tests/test_m14_l04_remote_transport.py](../tests/test_m14_l04_remote_transport.py) - Native PowerShell build-only/fake-process and argument-quoting regressions.
* [docs/sprint-plans/sprint-79.md](../docs/sprint-plans/sprint-79.md) - Marked the compatibility atomic task complete.
* [CHANGELOG.md](../CHANGELOG.md) - Recorded the transport compatibility correction.
* [.agents/memory/lessons-learned.md](../.agents/memory/lessons-learned.md) - Recorded native PowerShell API differences and safe fallbacks.

## Testing

* **Test File:** [tests/test_m14_l04_remote_transport.py](../tests/test_m14_l04_remote_transport.py)
* **Status:** Focused transport suite passed (`51 passed`); the new timeout and
  child-argv compatibility cases passed (`5 passed`) across native WinPS 5.1
  and pwsh 7. The related v2/postprocess suite passed (`127 passed`).
* **Execution Command:** `uv run pytest tests/test_m14_l04_remote_transport.py -q`

Additional compatibility probes executed production build-only and the private
fake process seam under native Windows PowerShell 5.1 without network/SSH;
pwsh 7 behavior and existing-target atomic replacement were also verified.

## Additional Notes

No real CUDA, network, SSH, model, dataset, or holdout operation was run.
The worktree was clean before this task; the source/tests and documentation
are being finalized in separate commits after the recorded Graphify update.
