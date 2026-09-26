# Task 80.1 — Representation-Problem Taxonomy

## Status

**Complete.** Sprint 80 task 80.1 is marked `[x]`; tasks 80.2–80.29 remain pending. Historical Sprint 79 evidence was not changed.

## Summary of Work

Added the single frozen, versioned machine-readable taxonomy at
`artifacts/representation_problem_taxonomy_v1.json`. It defines stable,
architecture-neutral identifiers and applicability rules for collapse/rank loss,
anisotropy/inactive dimensions, redundancy/superposition, separability/probe
leakage, density/OOD/distribution drift, sparse-feature instability, and
sequence/trajectory drift. Every family declares required evidence, including
measurements, controls/counterexamples, and provenance where applicable.

Added `latent_anything._representation_taxonomy`, a data-driven contract loader
and evaluator. It validates the taxonomy structure and only allows a claim when
applicability is explicitly `applicable` and every required evidence item is
explicitly `observed`. `not_applicable`, `unsupported`, unknown evidence,
invalid evidence status, missing evidence, and schema mutations cannot produce a
passing claim; unsupported and non-applicable cases return
`outcome="unsupported"` with `claim_allowed=False`.

## Files Modified

- `artifacts/representation_problem_taxonomy_v1.json` — canonical versioned taxonomy document.
- `src/latent_anything/_representation_taxonomy.py` — structural validator, loader, and fail-closed claim semantics; no model-specific code.
- `tests/test_representation_taxonomy.py` — focused consumer-observable contract tests.
- `docs/sprint-plans/sprint-80.md` — marked task 80.1 complete only.

## Validation Evidence

Focused validation only (no formatter, linter, type checker, or project-wide suite was run):

```text
uv run pytest tests/test_representation_taxonomy.py -q
6 passed in 12.66s

uv run python -c "from latent_anything._representation_taxonomy import evaluate_claim, load_taxonomy; t=load_taxonomy(); f=next(x for x in t['families'] if x['id']=='sequence_trajectory_drift'); e={x['id']:'observed' for x in f['required_evidence']}; ok=evaluate_claim('sequence_trajectory_drift', applicability='applicable', evidence_status=e, taxonomy=t); blocked=evaluate_claim('sequence_trajectory_drift', applicability='not_applicable', evidence_status=e, taxonomy=t); assert (ok.outcome, ok.claim_allowed)==('supported', True); assert (blocked.outcome, blocked.claim_allowed)==('unsupported', False); print('PASS valid_applicable_and_not_applicable_fail_closed')"
PASS valid_applicable_and_not_applicable_fail_closed

uv run python -m json.tool artifacts/representation_problem_taxonomy_v1.json
(valid JSON; no error)
```

The final task-level graph refresh was:

graphify update .
Re-extracting code files in . (no LLM needed)...
  AST extraction: 100/333 uncached files (30%) [8 workers]
  AST extraction: 200/333 uncached files (60%) [8 workers]
  AST extraction: 300/333 uncached files (90%) [8 workers]
  AST extraction: 333/333 uncached files (100%) [8 workers]
  warning: 244 source file(s) produced zero nodes and are absent from the graph: hooks.json, evals.json, evals.json, act_policy_representation_benchmark.json, anisotropy_benchmark.json (+239 more). A re-run will retry them (empties are no longer cached); if it persists, please report the file(s) (#1666).
[graphify watch] community set changed since labeling (1010 saved labels, 1047 communities now; renamed 1010 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) -> 2026-09-21/
Graph has 13677 nodes (above 5000 limit). Building aggregated community view...
graph.html written (aggregated: 1047 community nodes, 1320 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt: 13677 nodes, 28233 edges, 1047 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
Backgrounded as job bg_4; result completed after 69.79s with exit code 0.
```

## Affected Claims

- The taxonomy contract now covers all seven named representation-problem families.
- Evidence completeness and applicability are explicit machine-readable contract fields.
- Unsupported and explicitly non-applicable cases are non-promotable and cannot masquerade as passing evidence.

## Negative Results and Limitations

- This task freezes contract semantics only; it does not implement detectors, benchmark manifests, report schemas, workflow orchestration, model adapters, or evidence runs.
- The evaluator checks declaration/evidence status semantics, not the scientific validity of a detector's metric values; later detector and report tasks must provide those values and controls.
- Sequence/trajectory drift is explicitly non-applicable when no sequence/time axis is declared; it remains unsupported rather than passing.
- No project-wide validation was run by design; the focused test and direct smoke command above are the complete task-level validation evidence.
