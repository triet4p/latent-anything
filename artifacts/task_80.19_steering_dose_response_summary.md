# Task Summary: 80.19 — Steering and dose-response trials

**Sprint:** Sprint 80 — Stable Depth
**Task:** 80.19

## Summary of Work

Extended `src/latent_anything/_intervention_trials.py` with a second
architecture-neutral private executor for steering dose-response trials at
the workflow `intervene` stage (`make_steering_intervene_executor` plus the
`SteeringTrialSpec` carrier), sharing the80.18 binding, measurement,
central-control, uncertainty, and report machinery. Each steering trial
binds, before any measurement callback fires, the exact prior
capture/localize/explain identities, request/manifest identities, the exact
downstream metric and target (the target must be a prior localize
selection), and an immutable unit direction fingerprinted into a SHA-256
`direction_digest` with required direction provenance (`method`,
`carrier`, `source` — the80.19 smoke learns directions through the real
`methods/steering.py` `SteeringVector.fit` prototype contrast). The
predeclared dose plan must include the zero/identity strength and at least
two nonzero doses; declarations with missing/unique-violating/non-finite
strengths, non-unit/non-1-D/non-finite directions, missing provenance, or
wrong control classes fail at construction with `InterventionError`.

Execution measures the zero/identity dose as baseline, every nonzero dose
with its own identity-derived stream and per-dose bootstrap interval
(`run_bootstrap`, manifest repetitions/confidence), then runs the three
required controls — zero-identity, random-direction (rng-drawn direction on
the same captured inputs/scope/metric at the strongest declared dose), and
off-target (declared direction on a different selection) — through the
central `_statistical_controls` executor (`ControlSpec`/`ControlPlan`/
`execute_plan`/`derive_stream_seed`). Control rules: zero-strength must
reproduce the baseline within the manifest threshold tolerance;
random-direction fails only when its effect carries the declared sign AND
reaches `max(|endpoint effect|, tolerance)` (an rng direction that moves
the metric differently is recorded, never hidden); off-target fails
sign-agnostically on that bound, enforcing selectivity. The response is
classified (`monotonic_increase` / `monotonic_decrease` /
`non_monotonic` / `inert` / `unsupported`) from consecutive material dose
steps, and a predeclared decision table yields a bounded
supported/falsified/inconclusive/unsupported conclusion: failed identity
control blocks with `unsupported`; failed random-direction or off-target
control falsifies specificity/selectivity; inert or non-monotonic
responses and wrong-direction monotonic responses falsify the predeclared
dose expectation; an endpoint interval spanning the baseline is
inconclusive; support additionally requires every control, the monotonic
declared-direction response, material task-metric change, and uncertainty
excluding the baseline — a directional correlation never promotes without
passing all gates. Every record carries strengths, per-dose measurements
and uncertainty, direction provenance/digest, task-metric change,
random-direction and off-target measurements/outcomes, seeds, and the
bounded conclusion; `intervention_report_items` (widened to accept the
`steer` kind) renders bounded `causal_result` claim fragments.
`_bind_declaration`/`_bind_stage` were extracted so80.18 and80.19 share one
binding implementation; the coordinator remains method-agnostic and the
public `latent_anything` API is unchanged.

## Files Modified

* [src/latent_anything/_intervention_trials.py](src/latent_anything/_intervention_trials.py) - Added `SteeringTrialSpec`, `STEERING_KIND`, `STEERING_CONTROL_CLASSES`, response classification/decision table, per-dose execution with central controls, `make_steering_intervene_executor`; extracted shared `_bind_stage`/`_bind_declaration`/`_run_central_controls`/`_intervene_output`; widened control-class/kind/report validation additively.
* [tests/test_steering_trials.py](tests/test_steering_trials.py) -9 behavioral tests: supported dose response with full recording, non-monotonic/inert/wrong-direction falsification, identity/random/off-target blocking, uncertainty-inconclusive, pre-callback declaration and identity rejection, measurement-contract failures, deterministic replay, real `SteeringVector` reuse, and a four-conclusion workflow run with `validate_report_shape`.
* [scripts/sprint80_task80_19_smoke.py](scripts/sprint80_task80_19_smoke.py) - Committed smoke running the real seven-stage workflow (real capture binding, real detect executor, real explain executor, steering intervene executor) with four geometry-grounded trials on the frozen encoder manifest.
* [docs/sprint-plans/sprint-80.md](docs/sprint-plans/sprint-80.md) - Marked 80.19 `[x]`.

## Testing

* **Test Files:** [tests/test_steering_trials.py](tests/test_steering_trials.py) (9 tests) and [tests/test_intervention_trials.py](tests/test_intervention_trials.py) (10 tests, regression after the shared-binding refactor)
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_steering_trials.py tests/test_intervention_trials.py -q`
* **Result:** `19 passed in 4.19s`
* **Smoke (80.19):** `uv run python scripts/sprint80_task80_19_smoke.py` → exit 0. Exact output:

```
PASS workflow path: monotonic dose response supported; three falsifying trials blocked
PASS supported response: every strength, measurement, and uncertainty recorded
PASS falsifying/blocked: non-monotonic, off-target selectivity, random-direction specificity
PASS shared inputs/metric + direction provenance: one capture_identity, digests recorded
PASS determinism: replay reproduces the identical intervene payload and digest
PASS fail-closed: tampered identity and non-localized target rejected before callbacks
PASS failed control: identity/zero-strength failure blocks causal support
PASS report path: steer causal_result claims shape-validate with bounded conclusions
public surface: 9/9 80.6 names intact, no steering symbols leaked
PASS frozen manifest: read-only, unchanged by the smoke run
```

* **Regression smoke (80.18):** `uv run python scripts/sprint80_task80_18_smoke.py` → exit 0, all nine PASS lines (workflow/shared-inputs/falsifying/determinism/fail-closed/failed-control/report/surface/manifest).
* **Graph:** `graphify update .` after graph-visible writes → `AST extraction:255/255 uncached files (100%)`, `Rebuilt:15153 nodes, 33570 edges, 1061 communities`, `graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out`. A final confirming `graphify update .` ran after this artifact was written; its output is reported in the task handoff.
* **Scope note:** no formatters, linters, or project-wide suites were run, per task constraints.

## Additional Notes

* Smoke trial outcomes (measured on the real captured batch, participation
  ratio via `compute_latent_health`, tolerance0.1): `steer-on-half-e0`
  monotonic decrease (endpoint −0.5075) → supported; `steer-gap-nonmonotonic`
  (rise then fall) → falsified by classification; `steer-offselect-e0`
  (on-tail endpoint −0.2132 vs off-half −0.4548) → falsified by
  off-target selectivity; `steer-nonspecific-direction` (uniform direction,
  endpoint +0.1299 vs rng-drawn +0.7170) → falsified by random-direction
  specificity. The rng draws are fixed by the predeclared control-id
  streams; the gate itself is untouched.
* Dose doses execute per-dose bootstrap intervals (doses × manifest
  repetitions callback draws per trial); with the smoke's deterministic
  callback the intervals are degenerate and reproducible, and stochastic
  callbacks produce real intervals (exercised by the inconclusive test).
* Trial control ids remain request+trial-declared (not manifest controls)
  exactly as in80.18; the localize stand-in declares the slice selections
  the steering targets must bind to.
* Downstream: 80.20 comparison, 80.21 persistence, 80.22 report rendering.
