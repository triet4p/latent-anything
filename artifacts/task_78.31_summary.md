# Sprint 78.31 — Alias and deprecation ledger

## Summary

Added the authoritative English compatibility ledger at
[`docs/API_COMPATIBILITY.md`](../docs/API_COMPATIBILITY.md), linked from
[`docs/INDEX.md`](../docs/INDEX.md). It records 18 verified public alias
records and two schema/path migration records, including canonical spellings,
beta timing, warning boundaries, identity/behavior guarantees, replacements,
and tests. No alias was removed and package metadata remains `0.1.0b1` under
the planned `0.9.0` compatibility-epoch decision.

Snapshot B was expanded with runtime-observed transition aliases and explicit
schema migrations. Regeneration command:

```text
uv run python scripts/api_freeze_snapshot.py --write
```

Current snapshot SHA-256: `9f2a17d70fb3df5978a186bcfdb6682ceec7dd64373c4929efe9b20d8709a2d3`.
Section B SHA-256: `daae209349031c72b2a04d6b67f8a9ebac5a2354849897345cc4c1a29d6dc923`.

## Owner timing correction

The timing column distinguishes RFC0001's planned `0.2.0` vocabulary window
from actual repository history: `0.2.0` was never published; canonical symbols
were added Unreleased in Sprint78.29, and canonical registry kinds landed in
Sprint31, all under package metadata `0.1.0b1`. Symbol aliases carry a current
Unreleased/Sprint78.31 deprecation notice without import-time warnings; registry
aliases are deprecated at the Unreleased/Sprint31 construction boundary with
one warning. No row calls the unreleased window a `since` or deprecation
release. The planned `0.9.0` removal decision is preserved.

Replay CLI parity was strengthened to execute both `replay-run` and
`replay-run-config` against the same missing record and assert identical
exception type and message, in addition to parser identity and capture output
parity.

## Runtime enforcement

[`tests/test_api_compatibility.py`](../tests/test_api_compatibility.py)
verifies exact canonical symbol identity, one warning per legacy registry
construction, canonical/legacy CLI parser/output parity and identical replay
failure type/message, MPPI Pydantic
aliases, CEM/MPPI/RolloutResult property parity and mutation protection, both
deterministic/stochastic transition prediction aliases, `std`/`nll` metric
aliases, and ledger coverage against snapshot B. The focused alias/API suite
passed **58 tests**.

The result-property and transition property aliases intentionally do not warn:
they have no safe construction boundary and warning on every property read
would make normal numerical use noisy. Registry normalization remains the one
warning-bearing boundary and is asserted at cardinality one.

## Gates

- CI-equivalent environment: `uv sync --locked --extra viz`.
- Full locked-viz pytest: **1560 passed, 36 skipped, 39 warnings**.
- Scoped Ruff and strict Pyright: **PASS** (`0 errors, 0 warnings, 0 informations`).
- MkDocs strict/link scan: **PASS** using `uv sync --locked --extra docs` and an explicit temporary site directory; the directory was removed safely afterward.
- `git diff --check`: **PASS**, with existing LF/CRLF normalization warnings only.
- Graphify final topology: **10,841 nodes / 20,922 edges / 921 communities**
  after `graphify update . --no-cluster` and `graphify cluster-only . --no-viz
  --no-label`.

## Policy outcome

The Sprint 28/31 naming and deprecation comparison is complete. All audited
families are either retained beta aliases with an explicit replacement and
deadline, warning-free aliases where warning is impractical, or separately
classified schema/path migrations. Removal remains a future reviewed action at
the `0.9.0` epoch; this task performs no removal, metadata bump, tag, or
publication.
