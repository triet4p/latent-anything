# Sprint 76 Remediation 11 — Fail-closed W&B resume provenance

Status: Complete (2026-08-25)

## Change

W&B resume now requires a provider-configured `latent_anything.identity` value
that is a canonical 64-character lowercase SHA-256 identity. Missing, empty,
non-string, malformed, or mismatching values fail the resume and finish the
provider run with a failure exit code. The adapter no longer accepts a legacy
run merely because a tagged fallback happens to be present.

## Focused validation

```text
uv run pytest -q tests/test_wandb_recorder.py
8 passed
```

The test matrix covers missing, empty, malformed, non-string, and changed
identity values.

## Graph refresh

`graphify update .` completed immediately after this atomic completion:

```text
Rebuilt: 9884 nodes, 19302 edges, 870 communities
Warning: 42 source/JSON files produced zero graph nodes; graphify retried them.
```
