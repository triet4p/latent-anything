# Task 79 L04.10 Phase A Summary

## Scope

Implemented the private, network-gated additive steering runtime for the
frozen M14 L04.10 contract. The runtime targets the pinned
`TransformerLMIntegration` GPT-2 boundary and keeps additive hidden-state
intervention separate from true interchange activation patching. No real
model, CUDA, remote execution, or evidence promotion was performed.

## Changes

- Added `scripts/_m14_l04_steering.py` with layer 6/native hidden-state index 7
  hook capture and additive `hidden + strength * direction` intervention.
- Fit the normalized clean-minus-corrupted direction from train pairs only and
  evaluate the four holdout groups with strengths `0`, `0.25`, `0.5`, and `1`.
- Added five deterministic seeds, 2,000-replicate group bootstrap metrics,
  zero-strength identity, randomized-direction, shuffled-label,
  matched-norm, and previous-valid-token off-target controls.
- Added fail-closed target-token, split/pair, finite-value, CUDA/network,
  resource-budget, no-mutation, and hook-cleanup handling without returning
  prompts, token arrays, raw model outputs, weights, or filesystem paths.
- Corrected off-target locality to aggregate `mean(abs(clean-corrupted
  effect per causal pair))`, preventing opposite endpoint changes from
  cancelling. Cleanup failures now return an explicit failed/D0 result, and
  successful runs retain equal model-parameter digests before/after.
- Phase A exposes `semantic_candidate` and its diagnostic criteria only;
  `evidence_eligible`, `acceptance`, and `evidence_level` remain hard-coded to
  `false`, `false`, and `D0`. Artifact assembly preserves that boundary with a
  strict AdditiveSteering sanitizer and generic D0 validator: only allowlisted
  metrics, controls, raw summaries, provenance, and bounded resource peaks are
  retained; handler flags, unknown fields, unsafe values, and digest/linkage
  mismatches fail closed.
- The validator links every retained additive field across the top-level
  artifact, active execution, and active record, including all provenance
  fields and resource tuples. Pre-CUDA envelopes validate only their strict
  sparse schema; they never pass through the completed-result sanitizer.
- `completed_real_cuda_d0` is accepted as technical completion in artifact,
  run, and failure status/linkage checks while remaining D0, non-eligible, and
  non-accepted. Invalid handler mapping keys/constants and recursive unknown or
  sensitive fields fail closed; malformed handler results become conservative
  failed D0 triads.
- Randomized, shuffled-label, matched-norm, off-target, and zero-strength
  controls are emitted with per-seed metric/pass fields. They are diagnostics,
  not repository acceptance claims; final promotion remains deferred.
- Wired the `AdditiveSteering` real-run dispatch branch and added focused
  injectable fake-integration tests.

## Files Modified

- `scripts/_m14_l04_steering.py` — private additive steering runtime.
- `scripts/m14_l04_explanations.py` — network-gated dispatcher mapping.
- `scripts/_m14_l04_artifact.py` and `scripts/_m14_l04_validate.py` — preserve
  the Phase A D0 boundary and validate diagnostic real-CUDA provenance.
- `scripts/_m14_l04_envelope.py` — link completed D0 status to complete
  run/failure stages.
- `tests/test_m14_l04_steering.py` — deterministic hook, paired off-target,
  cleanup failure, digest, artifact integration, sanitization, malformed-input,
  cross-envelope linkage, and dispatcher coverage.
- `docs/sprint-plans/sprint-79.md` — marked L04.10 Phase A complete.
- `CHANGELOG.md` — recorded the user-visible runtime addition.

## Verification

- `uv run pytest tests/test_m14_l04_steering.py -q` — 29 passed.
- Related L04 regression selection (`uv run pytest tests -k m14_l04 -q`) —
  605 passed, 1 skipped, 1673 deselected.
- Focused steering/runner/explanations/remote-postprocess selection — 115
  passed.
- Ruff format/check and strict Pyright for changed runtime/tests — passed.
- `uv run mkdocs build --strict` — completed successfully.
- `graphify update .` — completed; existing graphify extraction warnings for
  zero-node JSON files were reported by the tool.
- L04.9 D3 regression/integrity selection (`test_m14_l049_v2*.py`) — 224
  passed; no L04.9 D3 artifact bytes were changed.

## Phase A integrity follow-up

- Holdout sanitization now requires the exact `g09`--`g12` clean/corrupted
  pairs, unique row IDs, canonical pair linkage, and no train overlap.
- Fixture metadata is independently rehashed against the authored fixture and
  accepts only lowercase 64-hex digests; self-rehashed substitutions fail.
- Artifact assembly cross-links per-seed metrics, raw-summary derivations,
  control effects, and matched-norm diagnostics before retaining any field.
- Post-CUDA handler failures preserve their stage/backend/network provenance;
  partial failures may retain the canonical `resource_peak: "not measured"`
  envelope while remaining failed D0 diagnostics.
- Every real Additive handler result now publishes an
  `execution_result_digest` inside provenance before assembly; the artifact,
  run record, and failure envelope retain and validate the exact device,
  network, backend, stage, attempted, cleanup, and resource-peak tuple.
- Cleanup-stage mutation retains the original causal `failure_stage`, so an
  attempted CUDA execution cannot be rewritten as a preflight failure.
- Shuffled-label control now permutes balanced train-example labels
  deterministically per seed, independently of fitted labels, and records only
  safe policy/cardinality/actual-assignment-digest provenance; unchanged label
  assignments are rejected even when the sampled index permutation is not the
  identity permutation.
- Additive metric and control threshold/comparator/unit fields are checked
  against the frozen plan declarations after result self-rehashing, so a
  self-rehashed threshold mutation still fails closed.
- Additional adversarial and synthetic scoring-failure regressions cover these
  boundaries; no model, CUDA, remote execution, or promotion was performed.

## Notes

The handler returns a sanitized execution result for the dispatcher. Task 2
D3 promotion wiring remains intentionally deferred; Phase A only accepts a
technically complete `completed_real_cuda_d0` diagnostic status. L04.9 D3
evidence bytes and related historical artifacts were not modified.
