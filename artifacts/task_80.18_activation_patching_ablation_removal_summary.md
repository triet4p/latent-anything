# Task Summary: 80.18 — Activation patching, ablation, and removal trials

**Sprint:** Sprint 80 — Stable Depth
**Task:** 80.18

## Summary of Work

Added one architecture-neutral private executor for declared
patch/ablate/remove causal trials at the workflow `intervene` stage
(`src/latent_anything/_intervention_trials.py`). The executor binds every
trial and all four required control classes (identity/zero-strength,
random, shuffled, off-target) to the exact prior capture/localize/explain
payload identities, the shared captured-inputs digest
(`BoundCapture.capture_identity` from 80.7), and the single
manifest-declared downstream metric (a causal-expectation target metric
with a positive predeclared threshold tolerance). All binding, request,
metric, provenance, and control-declaration checks run in full before any
measurement callback fires and raise `StageContractError`; declared
non-finite values and malformed trial declarations raise
`InterventionError` at construction time. Measurements are taken through
one caller-supplied callback that must echo the bound capture digest and
metric id and must be finite — violations fail the stage closed.

Baseline, intervened, and all four control measurements plus a200-draw
bootstrap uncertainty interval run through the central
`_statistical_controls` executor (`ControlSpec`/`ControlPlan`/
`execute_plan`/`run_bootstrap`/`derive_stream_seed`) under
identity-derived streams, recording intervention provenance (method,
carrier, manifest/request, capture/localize identity), seeds
(evaluation/controls plus per-role stream seeds), target,
strength/removal semantics, baseline/intervened/control measurements,
control outcomes and thresholds, uncertainty, and a bounded
supported/falsified/inconclusive/unsupported conclusion from a
predeclared decision table (failed zero-strength control blocks with
`unsupported`; failed specificity controls falsify target specificity;
inert or wrong-direction effects falsify; baseline inside the intervened
interval is inconclusive). `intervention_report_items` renders bounded,
shape-compatible `causal_result` claim fragments at the same Phase-IV
boundary as 80.16 (claims allow only supported/falsified; blocked records
name their reason). The `DiagnosticWorkflow` coordinator was not modified;
no method-specific logic entered it.

## Files Modified

* [src/latent_anything/_intervention_trials.py](src/latent_anything/_intervention_trials.py) - New private intervene-stage executor: trial/control declaration carriers, pre-callback identity binding, central control scheduling, uncertainty, bounded conclusions, report claim fragments, `make_intervene_executor` factory.
* [tests/test_intervention_trials.py](tests/test_intervention_trials.py) -10 contract-level tests covering positive patch/ablate/remove cases, shared-input/metric controls, failed-control blocking, inert/wrong-direction falsification, inconclusive uncertainty, all four conclusions in one real workflow run with `validate_report_shape`, pre-callback rejection of tampered prior/stage identity, declaration validation, measurement-contract failures, and deterministic replay.
* [scripts/sprint80_task80_18_smoke.py](scripts/sprint80_task80_18_smoke.py) - Committed smoke running the real seven-stage `DiagnosticWorkflow` with real capture binding (80.7), real detect executor (80.9), real explain executor (80.16), and the new intervene executor over the frozen encoder manifest.
* [docs/sprint-plans/sprint-80.md](docs/sprint-plans/sprint-80.md) - Marked 80.18 `[x]`.

## Testing

* **Test File:** [tests/test_intervention_trials.py](tests/test_intervention_trials.py)
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_intervention_trials.py -q`
* **Result:** `10 passed in 3.93s`
* **Smoke:** `uv run python scripts/sprint80_task80_18_smoke.py` → exit0. Exact output:

```
PASS workflow path: patch/ablate/remove supported, inert ablate falsified
PASS shared inputs/metric: every measurement echoes one capture_identity and metric
PASS falsifying case: declared ablate intervention is inert beyond noise: |effect| 0.0 does no...
PASS determinism: replay reproduces the identical intervene payload and digest
PASS fail-closed: tampered capture identity rejected before callbacks
PASS failed control: identity/zero-strength failure blocks causal support
PASS report path: causal_result claims shape-validate with bounded conclusions
public surface: 9/9 80.6 names intact, no intervention symbols leaked
PASS frozen manifest: read-only, unchanged by the smoke run
```

* **Graph:** `graphify update .` after graph-visible writes → `AST extraction:254/254 uncached files (100%)`, `Rebuilt:15057 nodes, 33191 edges, 1066 communities`, `graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out`. A final confirming `graphify update .` ran after this artifact was written; its output is reported in the task handoff.
* **Scope note:** no formatters, linters, or project-wide suites were run, per task constraints.

## Additional Notes

* Conclusion semantics are predeclared in the module docstring decision
  table; the specificity gate compares control effects against
  `max(|intervened - baseline|, manifest threshold tolerance)` with strict
  inequality, so a control that reproduces the intervention effect fails
  and can never promote causal support.
* The smoke's localize payload is a contract-shaped stand-in: the frozen
  encoder manifest declares no layer axis, so the80.13 layer/slice
  fixture cannot run against it; every identity field the explain and
  intervene executors bind (manifest, representation, family, metric,
  slice selection, verdict) is present and real, and detect/explain/
  capture/intervene all execute their real executors.
* Trial control ids are request+trial-declared, not manifest controls:
  `detection_config_from_manifest` (80.9) rejects request controls absent
  from the frozen manifest, and the frozen manifests predeclare only
  detection-side statistical controls. The smoke therefore binds real
  detect evidence through a detect-side request whose manifest identity
  matches the run, while the main request declares the intervention
  controls.
* Downstream tasks:80.19 adds steering/dose-response trials beside this
  seam;80.21/80.22 register the provisional
  `intervention-<id>-record` evidence refs when assembling the persisted
  report.
