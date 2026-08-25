# Sprint 77 Phase B Task 01 — Rust/PyO3 deferral decision

Status: complete (2026-08-25). This task records the owner-approved decision;
it does not start a Rust implementation or close Sprint 77/Milestone 13.

## Decision

Rust/PyO3 is deferred for the pre-stable framework, not permanently
prohibited. The decision is recorded in
`.agents/memory/decisions.md` and keeps the public NumPy boundary unchanged.

The measured basis is the Phase-A comparison in
`artifacts/sprint77_phase_a_comparison.json`: Euclidean DTW improved from
38,297.3 to 27,478.2 microseconds median (-28.25%) with an unchanged
correctness digest after a low-risk NumPy optimization. The remaining profiled
cost includes dynamic programming/traceback and dependency work. Geodesic
finite differences and Arrow/SQLite/filesystem paths do not currently identify
a narrow, low-risk Rust kernel. The evidence is limited to one Windows 11 CPU
environment, eight repetitions, unavailable RSS, no real-policy performance
lane, and no larger-workload call-frequency study.

## Reconsideration contract

A future proposal must establish larger-workload call frequency and scaling,
isolate removable kernel cost against the current NumPy baseline, preserve
dtype/shape/window/max-step/tie-break/traceback/error semantics, and quantify
cross-platform wheels, toolchain, maintenance, expected benefit, and the
cross-language contract. No Rust dependency, extension, or ADR implementation
plan is added by this task.

## Focused validation

```text
uv run python -c "import json; json.load(open('docs/evidence-ledger.json', encoding='utf-8'))"
PASS; evidence ledger parses.
git diff --check
PASS; only normal Git LF/CRLF conversion warnings.
```

## Graphify trace

```text
graphify update .
PASS; graphify rebuilt 10,017 nodes / 19,517 edges / 908 communities. It
reported 46 recurring zero-node JSON/source warnings and a changed community
set (898 saved labels to 908 communities); graphify backed up the prior
semantic/curated graph and regenerated the aggregated graph view.
```
