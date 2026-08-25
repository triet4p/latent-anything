# Sprint 76 Remediation 16 — Windows drive-path strings

Status: Complete (2026-08-25)

## Change

MLflow now detects absolute Windows drive-path strings before `urlsplit`, so
`C:/tracking` and `C:\\tracking` normalize consistently with `Path` and
`file:///C:/tracking` on Windows. Drive-relative, encoded, ADS, traversal,
UNC/device, remote, URI-like, and reparse forms remain fail-closed. POSIX
continues to reject drive-qualified forms as ambiguous.

## Files

- `src/latent_anything/integrations/mlflow_recorder.py`
- `tests/test_mlflow_recorder.py`
- `docs/OPTIONAL_INTEGRATIONS.md`
- `docs/sprint-plans/sprint-76.md`

## Focused validation

```text
uv run pytest -q tests/test_mlflow_recorder.py
15 passed in 4.42s
```

## Graph refresh

`graphify update .` completed immediately after this atomic completion:

```text
Rebuilt: 9926 nodes, 19372 edges, 874 communities
Warning: 42 source/JSON files produced zero graph nodes; graphify retried them.
```
