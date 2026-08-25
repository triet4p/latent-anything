# Sprint 77 Phase A Task 04 — product budgets and evidence ledger

Status: complete (2026-08-25)

## Delivered

- Added `docs/PERFORMANCE.md` with the fixed workload matrix, environment
  scope, proposed p95 budgets, hard semantic gates, advisory latency/memory
  policy, profile attribution, and an explicit Rust-decision boundary.
- Added the Phase-A contract record to both `docs/EVIDENCE_LEDGER.md` and
  `docs/evidence-ledger.json`.
- Added the page to `docs/INDEX.md` and `mkdocs.yml`; added the user-visible
  Phase-A harness/DTW optimization entry to `CHANGELOG.md`.

## Proposed budgets

For the declared offline fixture only: distance 100 µs, DTW 50 ms, geodesic
150 ms, activation-capture overhead 300 µs, small rollout/planning 5 ms,
portable codec 3 ms, one bounded stream 3 ms, and plugin listing 100 µs. The
filesystem recorder and artifact/cache values remain storage-dependent
advisories, not real-time guarantees. Hard gates are semantic digest,
bounded-input, no-network, and test-contract checks; latency/variance/memory
remain environment-sensitive.

## Exact validation

```text
uv run python -c "import json; json.load(open('docs/evidence-ledger.json', encoding='utf-8'))"
PASS; typed ledger parses
```

## Graphify trace

```text
graphify update .
PASS; 10,003 nodes / 19,505 edges / 885 communities; 46 zero-node JSON/source warnings.
```
