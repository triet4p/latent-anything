# Sprint 76 Remediation 13 — W&B offline network and resume evidence

Status: Complete (2026-08-25)

## Change

The real W&B lane sets offline mode and its local directory before SDK import,
denies HTTP/URL and non-local socket destinations, and permits only loopback
or IPC connections required by W&B's local service. It verifies distinct
parent/child provider IDs, independently hashes retrieved mirror bytes, waits
for newly created threads to settle after teardown, and rejects offline resume
when W&B has not persisted the adapter identity instead of claiming false
continuation semantics.

## Focused validation

```text
uv run --extra tracking pytest -q -m integration tests/test_tracking_parity.py
2 passed, 1 deselected
```

## Graph refresh

`graphify update .` completed immediately after this atomic completion:

```text
Rebuilt: 9904 nodes, 19334 edges, 889 communities
Warning: 42 source/JSON files produced zero graph nodes; graphify retried them.
```
