# Task 80.3 — Diagnostic-Report Schema

## Status

**Complete.** Sprint 80 tasks 80.1–80.3 are marked `[x]`; tasks 80.4–80.29 remain pending. Historical Sprint 79 evidence was not changed.

## Summary of Work

Added the single frozen machine-readable schema at
`artifacts/diagnostic_report_schema_v1.json`. It defines capture provenance,
symptoms, localization, hypotheses, statistical evidence, interventions,
comparisons, limitations, next action, and claims. Claims carry explicit
`kind`, `status`, evidence references, control references, causal flag,
claim-allowed flag, and missing-evidence reasons.

The four evidence kinds are intentionally disjoint: `observation`,
`explanation`, `causal_result`, and `unsupported_conclusion`. The schema records
allowed statuses and required semantics for each kind. Explanations cannot be
causal results, causal results must declare `causal=true` and control
references, and unsupported conclusions must be `status=unsupported`,
`claim_allowed=false`, and include missing-evidence reasons.

Added `latent_anything._diagnostic_report`, a shape and evidence-kind semantic
checker scoped to this schema. It does not validate taxonomy references,
artifact hashes, control outcomes, evidence completeness across artifacts, or
conclusion strength; those remain the independent 80.4 validator boundary.
No concrete report or workflow implementation was added.

## Files Modified

- `artifacts/diagnostic_report_schema_v1.json` — canonical versioned report schema.
- `src/latent_anything/_diagnostic_report.py` — schema loader and scoped shape/semantic validator.
- `tests/test_diagnostic_report.py` — focused valid distinction and malformed/conflated-case tests.
- `docs/sprint-plans/sprint-80.md` — marked task 80.3 complete only.

## Validation Evidence

Focused validation only (no formatter, linter, type checker, or project-wide suite was run):

```text
uv run pytest tests/test_diagnostic_report.py -q
7 passed in 2.98s

uv run python -c "from copy import deepcopy; from tests.test_diagnostic_report import _report; from latent_anything._diagnostic_report import DiagnosticReportValidationError, validate_report_shape; valid=_report(); validate_report_shape(valid); invalid=deepcopy(valid); invalid['claims'][1]['causal']=True; ..."
PASS distinct_kinds_and_explanation_conflation_rejection

uv run python -m json.tool artifacts/diagnostic_report_schema_v1.json > NUL
(exit code 0; valid JSON)
```

The final task-level graph refresh was:

```text
[run command: `graphify update .`]
Re-extracting code files in . (no LLM needed)...
  AST extraction: 100/251 uncached files (39%) [8 workers]
  AST extraction: 200/251 uncached files (79%) [8 workers]
  AST extraction: 251/251 uncached files (100%) [8 workers]
  warning: 246 source file(s) produced zero nodes and are absent from the graph: hooks.json, evals.json, evals.json, act_policy_representation_benchmark.json, anisotropy_benchmark.json (+241 more). A re-run will retry them (empties are no longer cached); if it persists, please report the file(s) (#1666).
[graphify watch] community set changed since labeling (1036 saved labels, 1038 communities now; renamed 23 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) -> 2026-09-21/
Graph has 13757 nodes (above 5000 limit). Building aggregated community view...
graph.html written (aggregated: 1038 community nodes, 1206 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt: 13757 nodes, 28437 edges, 1038 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
Exit code: 0 (48.92s)
```

## Affected Claims

- A diagnostic report now has one versioned, architecture-neutral machine-readable contract with every named report section.
- Consumers can distinguish direct observation, non-causal explanation, causal result, and unsupported conclusion without relying on prose.
- Unsupported conclusions are explicitly non-promotable and cannot be represented as supported evidence.

## Negative Results and Limitations

- This task freezes report shape and evidence-kind semantics only; it does not render reports, execute workflow stages, or validate scientific evidence.
- Cross-artifact references are intentionally opaque strings. Taxonomy linkage, evidence completeness, control outcomes, artifact hashes, and conclusion-strength validation remain 80.4 responsibilities.
- A falsified causal result is a valid causal outcome record, while an unsupported causal claim must use `unsupported_conclusion`; the schema does not reinterpret either outcome.
- No project-wide validation was run by design. Evidence-review readiness is limited to the focused contract tests, JSON syntax check, and direct valid/conflated smoke scenario above.
