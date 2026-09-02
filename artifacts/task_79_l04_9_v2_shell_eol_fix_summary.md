# Task Summary: Sprint 79 L04.9 v2 shell payload EOL fix

**Sprint:** Sprint 79
**Task:** Preserve exact L04 remote shell payload bytes across Windows Git clones

## Summary of Work

Tracked Bash scripts now declare `*.sh text eol=lf`, preventing effective
`core.autocrlf=true` configurations from converting remote transport payloads
to CRLF. Regression coverage enumerates every tracked `.sh`, checks explicit
`text: set`/`eol: lf` attributes, and clones the payload with
`core.autocrlf=true` before exercising BuildOnly through pwsh and native
Windows PowerShell when available. The clone payload bytes, size, and SHA-256
must remain identical to the source payload. Existing L04.9 v2 `-text`
evidence rules and bytes remain unchanged.

## Files Modified

* `.gitattributes` — pins tracked shell scripts to LF checkout bytes.
* `tests/test_m14_l049_v2_git_integrity.py` — verifies all tracked shell
  scripts have explicit LF attributes.
* `tests/test_m14_l04_remote_transport.py` — verifies BuildOnly metadata after
  a `core.autocrlf=true` clone under both PowerShell variants.
* `CHANGELOG.md` — records the cross-platform payload guarantee.
* `.agents/memory/lessons-learned.md` — records the Windows checkout trap.
* `graphify-out/` — refreshed via `graphify update .`.

## Testing

* `uv run pytest tests/test_m14_l049_v2_git_integrity.py -q` — 5 passed.
* `uv run pytest tests/test_m14_l04_remote_transport.py -q` — 86 passed.
* Full L04/v2 suite — 583 passed, 1 skipped.
* Visualization suite with `--extra viz` — 78 passed.
* `uv run ruff check src tests scripts` — passed.
* `uv run pyright` — 0 errors, 0 warnings, 0 informations.
* `uv run mkdocs build --strict` — passed.
* `uv run python scripts/validate_evidence_ledger.py` — passed.

The repo-wide Ruff format check still reports the unrelated pre-existing
`scripts/_m14_l04_validate_tcav.py`; all files changed by this task pass the
format check. No commit, push, SSH, CUDA run, or evidence/D3 edit was made.
