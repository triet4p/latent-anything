# Task Summary: 80.20 — Aligned checkpoint/run comparison

**Sprint:** Sprint 80 — Stable Depth
**Task:** 80.20

## Summary of Work

Added `src/latent_anything/_run_comparison.py`: one architecture-neutral
private executor for the workflow `compare` stage
(`make_compare_executor` plus the `ComparisonSpec`/`RunSide` carriers).
Each side declares immutable `run_id`/`checkpoint_id` identities and an
`alignment` mapping over the exact14-field `ALIGNMENT_FIELDS` set using
the frozen80.14 vocabulary (`dataset_slice_id`, `dataset_configuration`,
`preprocessing_identity`, `layer_module_identity`, `model_identity`,
`representation_identity`, `seeds`, `diagnostic_config`,
`manifest_identity`, `taxonomy_identity`, `schema_identity`, `axes`,
`representation_metric`, `task_metric`). Both sides must declare
identical alignment (differing/missing/extra fields reject at
construction), and the shared values are cross-checked before any
callback against the bound prior evidence and frozen manifest: dataset
configuration vs manifest split identity, dataset slice vs a prior
localize selection, layer/module vs a localize selection or the captured
representation, model/representation/axes vs the bound capture
provenance, seeds vs `manifest_seed_identity`, diagnostic config vs
`detect_config_identity` of the prior detect payload (SHA-256 of its
canonical config — invented config identities reject), manifest/schema vs
the manifest, taxonomy vs the prior localize family, and the metric
fields vs the spec's own metrics. The representation metric must equal
the prior localize metric, be carried by a non-omitted prior explain row,
and both metrics must be request-declared, manifest-declared, and present
in prior detect metric evidence (invented metric identities reject).
Request-declared comparisons, sides, and metric sets must match the
declared specs exactly; prior capture/detect/localize/explain/intervene
records must be present and structurally intact (missing intervene trials
or conclusions reject). Contract errors raise `StageContractError`
before callbacks; declaration and measurement errors raise
`ComparisonError`.

For each side and each role (representation, task) the caller's callback
receives a `ComparisonApplication` and must return a
`ComparisonMeasurement` echoing the side `run_id` (wrong side rejects),
the declared metric id, the shared captured-inputs digest, and a finite
value; each point has an identity-derived seed and each side/role gets a
manifest-repetition bootstrap interval through the central
`run_bootstrap`/`derive_stream_seed` primitives. Per metric the executor
records signed and absolute deltas (`candidate − baseline`, the M14
`build_comparison_report` convention), the conservative delta interval
`[cand.lower − base.upper, cand.upper − base.lower]`, the declared
tolerance, and the manifest metric definition. The bounded
classification never infers one metric from the other: any prior stage
outcome `unsupported` → `unsupported` (recorded, not promoted); any
metric whose exceedance coincides with a zero-spanning interval →
`inconclusive`; otherwise the independent change flags → `both` /
`representation_only` / `task_only` / `neither`. Every record carries
both side identities, the shared alignment, per-metric
baseline/candidate measurements with uncertainty, deltas, tolerances,
metric definitions, the classification, its reason, and bound provenance
(capture identity, detect-config identity, manifest/workflow digests,
upstream blockers). `comparison_report_items` renders bounded report
comparison rows (exact frozen-schema fields; change classifications map
to `observed`, inconclusive/unsupported stay non-promoted). Deterministic
canonical replay reproduces the payload and stage digest exactly. The
coordinator stays method-agnostic; the public API is unchanged.

## Files Modified

* [src/latent_anything/_run_comparison.py](src/latent_anything/_run_comparison.py) - New compare-stage executor: alignment/run-side carriers, identity helpers, pre-callback binding against all five prior records, per-side measurement with uncertainty, bounded classification, report rows, factory.
* [tests/test_run_comparison.py](tests/test_run_comparison.py) -9 behavioral tests covering the both-change record with full provenance, representation-only/task-only/neither separation, inconclusive delta interval, upstream-unsupported blocking, declaration validation, alignment/prior/request rejection before callbacks, measurement-contract failures, deterministic replay, and a four-classification workflow run with `validate_report_shape`.
* [scripts/sprint80_task80_20_smoke.py](scripts/sprint80_task80_20_smoke.py) - Committed smoke running the real seven-stage workflow (real capture binding, real detect executor, real explain executor, real intervene executor, compare executor) with five aligned comparisons plus the shared80.18 causal claim.
* [docs/sprint-plans/sprint-80.md](docs/sprint-plans/sprint-80.md) - Marked 80.20 `[x]`.

## Testing

* **Test File:** [tests/test_run_comparison.py](tests/test_run_comparison.py) (plus regression suites)
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_run_comparison.py tests/test_intervention_trials.py tests/test_steering_trials.py -q`
* **Result:** `28 passed in 10.53s` (9 comparison +10 intervention +9 steering)
* **Smoke (80.20):** `uv run python scripts/sprint80_task80_20_smoke.py` → exit 0. Exact output:

```
PASS workflow path: both/representation-only/task-only/neither/inconclusive recorded
PASS separation: representation-only and task-only never infer the other metric
PASS shared inputs/metric: every side/role measurement echoes one capture_identity
PASS fail-closed: alignment mismatch and prior tamper rejected before callbacks
PASS determinism: replay reproduces the identical compare payload and digest
PASS report path: bounded comparison rows +80.18 causal claim shape-validate
public surface: 9/9 80.6 names intact, no comparison symbols leaked
PASS frozen manifest: read-only, unchanged by the smoke run
```

* **Regression smokes:** `uv run python scripts/sprint80_task80_18_smoke.py` → exit 0, all nine PASS lines; `uv run python scripts/sprint80_task80_19_smoke.py` → exit 0, all ten PASS lines (shared intervention paths intact).
* **Graph:** `graphify update .` after graph-visible writes → `AST extraction:255/255 uncached files (100%)`, `Rebuilt:15266 nodes, 34052 edges, 1036 communities`, `graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out`. A final confirming `graphify update .` ran after this artifact was written; its output is reported in the task handoff.
* **Scope note:** no formatters, linters, or project-wide suites were run, per task constraints.

## Additional Notes

* Reuse: signed-delta convention and two-side record vocabulary from the
  M14 `build_comparison_report`/`RunComparisonReport` utilities; the
  alignment field names from `CheckpointAxisDeclaration`/`CheckpointCell`
  and `TokenAxisDeclaration` (80.14); central
  `derive_stream_seed`/`run_bootstrap` (80.15); frozen report-schema
  comparisons section (80.3) for the report rows; `ComparisonRequest`
  from80.6. `RunRecord`-backed `build_comparison_report` was intentionally
  NOT coupled into the stage (it belongs to the persistence lane,80.21).
* The smoke binds `diagnostic_config` through `detect_config_identity`
  computed from a real `evaluate_detection` payload evaluated before the
  workflow; the workflow's detect stage re-evaluates the identical
  frozen configuration (same identity).
* The frozen encoder manifest's two declared metrics serve as the
  representation (`bottleneck-effective-rank`, also the localize metric)
  and task (`bottleneck-singular-spread`) metrics in tests/smoke; a true
  task metric arrives with the core proofs (80.23–80.24).
* Localize payload remains a contract-shaped stand-in (the frozen encoder
  manifest declares no layer axis); capture/detect/explain/intervene run
  their real executors.
* Downstream: 80.21 persistence, 80.22 report rendering.
