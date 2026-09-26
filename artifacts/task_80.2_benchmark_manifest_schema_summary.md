# Task 80.2 — Benchmark-Manifest Schema

## Status

**Complete.** Sprint 80 task 80.2 is marked `[x]`; task 80.1 remains complete and tasks 80.3–80.29 remain pending. Historical Sprint 79 evidence was not changed.

## Summary of Work

Added the single frozen machine-readable schema at
`artifacts/benchmark_manifest_schema_v1.json`. The contract is
architecture-neutral and covers immutable model/revision identity, dataset and
split identity, representation layer/slice axes, injected/known defect or
counterexample declaration, taxonomy-linked metrics, independent seeds,
controls, uncertainty, causal expectation or explicit non-applicability,
predeclared pass/fail thresholds, and a canonical lock digest.

Added `latent_anything._benchmark_manifest`, a private schema loader and
consumer-facing manifest validator. A manifest is accepted only when it is
explicitly `predeclared`, has all exact required sections, binds all controls
and thresholds to declared metrics, declares uncertainty and seeds, and has a
locked SHA-256 commitment over canonical content. Missing fields, unexpected
fields, invalid types, mutable/post-hoc status, changed content after locking,
unsupported causal declarations, and unbound thresholds fail closed.

No concrete model, dataset, benchmark result, workflow API, report schema, or
80.5 manifest was added.

## Files Modified

- `artifacts/benchmark_manifest_schema_v1.json` — canonical versioned manifest schema.
- `src/latent_anything/_benchmark_manifest.py` — schema loader, canonical digest, and fail-closed manifest validator.
- `tests/test_benchmark_manifest.py` — focused valid, incomplete, post-hoc, mutation, and non-applicable causal contract tests.
- `docs/sprint-plans/sprint-80.md` — marked task 80.2 complete only.

## Validation Evidence

Focused validation only (no formatter, linter, type checker, or project-wide suite was run):

```text
uv run pytest tests/test_benchmark_manifest.py -q
8 passed in 3.67s

uv run python -c "from copy import deepcopy; from tests.test_benchmark_manifest import _manifest; from latent_anything._benchmark_manifest import BenchmarkManifestValidationError, validate_manifest; valid=_manifest(); validate_manifest(valid); invalid=deepcopy(valid); invalid['status']='completed'; ..."
PASS valid_manifest_and_post_hoc_rejection

uv run python -m json.tool artifacts/benchmark_manifest_schema_v1.json > NUL
(exit code 0; valid JSON)
```

The final task-level graph refresh was:

```text
[run command: `graphify update .`]
Re-extracting code files in . (no LLM needed)...
  AST extraction: 100/253 uncached files (39%) [8 workers]
  AST extraction: 200/253 uncached files (79%) [8 workers]
  AST extraction: 253/253 uncached files (100%) [8 workers]
  warning: 245 source file(s) produced zero nodes and are absent from the graph: hooks.json, evals.json, evals.json, act_policy_representation_benchmark.json, anisotropy_benchmark.json (+240 more). A re-run will retry them (empties are no longer cached); if it persists, please report the file(s) (#1666).
[graphify watch] community set changed since labeling (1047 saved labels, 1019 communities now; renamed 178 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) -> 2026-09-21/
Graph has 13722 nodes (above 5000 limit). Building aggregated community view...
graph.html written (aggregated: 1019 community nodes, 1225 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt: 13722 nodes, 28358 edges, 1019 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
Exit code: 0 (104.21s)
```

## Affected Claims

- Benchmark declarations now have one versioned, architecture-neutral machine-readable contract.
- The contract carries every field named by task 80.2 and binds thresholds/controls/causal expectations to declared metrics.
- Predeclared lock and canonical digest semantics prevent post-hoc mutation from masquerading as a valid manifest.

## Negative Results and Limitations

- This task freezes manifest declaration and validation semantics only; it does not execute models, datasets, metrics, controls, uncertainty estimators, or interventions.
- Taxonomy family identifiers are carried as opaque strings; taxonomy-reference validation belongs to later report validation (80.4).
- The validator confirms declaration shape, bindings, and commitment integrity, not the scientific correctness of metric values or control outcomes.
- Causal non-applicability is accepted only with an explicit reason and non-promotable empty target set; it is never treated as causal evidence.
- No project-wide validation was run by design. Evidence-review readiness is limited to the focused contract tests, JSON syntax check, and direct valid/post-hoc smoke scenario above.
