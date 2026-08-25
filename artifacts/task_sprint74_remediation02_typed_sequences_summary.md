# Sprint 74 remediation 02 — exact typed-result sequences

Status: Complete (2026-08-25)

## Scope

The typed envelope tree previously collapsed tuples into lists, changing
allowlisted CEM/MPPI/profile contracts on restore. Tuple values now use an
explicit private sequence marker and restore as tuples; ordinary lists remain
lists. Regression coverage asserts CEM candidate/history/profile tuples and
MPPI configuration tuple fields.

## Files

- `src/latent_anything/portable_results.py`
- `tests/test_portable_results.py`
- `docs/sprint-plans/sprint-74.md`

## Focused validation

```text
uv run ruff format src/latent_anything/portable_results.py tests/test_portable_results.py
2 files left unchanged
uv run ruff check src/latent_anything/portable_results.py tests/test_portable_results.py
All checks passed!
uv run pyright src/latent_anything/portable_results.py
0 errors, 0 warnings, 0 informations
uv run pytest -q tests/test_portable_results.py
5 passed in 5.73s
```

## Graph refresh

Pending at artifact creation; the required immediate refresh follows this
focused completion.

Refresh completed immediately after the concern:

```text
graphify update .
[graphify watch] Rebuilt: 9359 nodes, 18277 edges, 842 communities
Code graph updated.
```

The known 42 zero-node JSON/source warning and community-label drift warning
were reported; no graphify failure occurred.
