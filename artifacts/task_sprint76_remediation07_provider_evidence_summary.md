# Sprint 76 Remediation 07 — Real provider evidence

Status: Complete (2026-08-25)

The focused count below is the atomic-boundary snapshot; later remediation
tasks added tests. The final current totals are recorded in Remediation 09.

## Change

The optional integration lane now independently computes SHA-256, retrieves
real MLflow file-store artifacts, and verifies exact bytes and checksums. It
exercises real MLflow parent/child and resume behavior. W&B offline runs use a
documented adapter-owned validated artifact mirror because the offline SDK
does not expose a portable provider-side read API; the test verifies exact
mirror bytes/checksums, parent linkage, explicit fail-closed resume when the
provider drops adapter provenance, cleanup, and a network-denial sentinel.
W&B child runs request `reinit="create_new"` so the SDK cannot collapse them
into the active parent run.

## Focused validation

```text
uv run --extra tracking pytest -q -m integration tests/test_tracking_parity.py
2 passed, 1 deselected
```

No cloud server, credentials, or provider network was used.

## Graph refresh

`graphify update .` completed immediately after this atomic completion:

```text
Rebuilt: 9855 nodes, 19267 edges, 868 communities
Warning: 42 source/JSON files produced zero graph nodes; graphify retried them.
```
