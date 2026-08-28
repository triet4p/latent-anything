# Task Summary: Sprint 79 L04.7 — Tuned Logit Lens Phase A

Implemented the concrete tuned-lens runner and its offline contract seams.
The runtime validates and re-acquires only the exact pinned WikiText-2
manifest selections, streams model batches, fits independent dense affine
translators for native states 0..11 with the frozen tokenwise KL objective,
and keeps native state 12 as the identity terminal control. Fit and evaluation
combine paired source/target rows so each corpus batch performs one model
forward, then reuse detached hidden states/logits for every translator and
control under bounded memory. The deterministic schedule is seed 79, AdamW,
one epoch, batch size 4, learning rate 1e-3,
weight decay 1e-4, and gradient clipping 1.0. No authored fixture or MSE
substitute is used.

Holdout rows average token KL independently. The acceptance statistic is the
macro-average direct-minus-tuned improvement across layers 0..11, with strict
point and conservative minimum-lower-bound gates over bootstrap seeds
17/29/41/53/67 (2,000 percentile replicates each). A deterministic seed-79
shuffled-target translator is retained as a diagnostic control. Artifacts
retain only selected row indices, row text hashes, scalar layer metrics,
translator digests, global terminal-parity maxima, and provenance; corpus text
and hidden/logit tensors are not retained. The validator recomputes all
metric/control pass flags from numeric evidence and independently binds exact
manifest file/content/split digests and row order.

The accepted execution provenance also binds before/after model parameter
digests, explicit no-mutation evidence, CUDA device identity, and numeric
elapsed/GPU/RSS peak measurements. Validation recomputes the frozen
30-minute, 6-GB-GPU, and 4-GB-RSS budget gate and rejects missing, non-finite,
boolean, string, negative, or oversized evidence.
An accepted D3 result also fails closed whenever recomputed resource values
actually exceed any frozen cap, even if a serialized `budget_pass: false`
flag is internally consistent.

The implementation records real elapsed/resource measurements and enforces the
frozen caps only when the CUDA run is performed; offline tests do not claim the
30-minute target. The first authenticated real-CUDA attempt is recorded below;
the corrected commit still requires a new owner-authorized run to establish
semantic acceptance and resource evidence.

## Files

- `scripts/_m14_l04_execution_common.py`
- `scripts/_m14_l04_tuned_lens_metrics.py`
- `scripts/_m14_l04_wikitext_runtime.py`
- `scripts/_m14_l04_tuned_lens.py`
- `scripts/_m14_l04_validate_tuned_lens.py`
- `scripts/_m14_l04_artifact.py`
- `scripts/_m14_l04_validate.py`
- `scripts/m14_l04_explanations.py`
- `tests/test_m14_l04_tuned_lens.py`
- `artifacts/m14/l04-explanations.TunedLogitLens.attempt1.{partial,run,failure}.json` — retained semantic-failure evidence.
- `artifacts/m14/l04-explanations.ssh.TunedLogitLens.*.recovery.{audit,exit}.json` — sanitized recovery transport audit.

## Verification

- `uv run pytest tests/test_m14_l04_tuned_lens.py tests/test_m14_l04_runner.py -q` — 62 passed.
- Existing L04 regression suite — 136 passed after the correction.
- Full suite — 1768 passed, 36 skipped, 39 warnings.
- Ruff and strict Pyright for all touched Python files — passed.
- `graphify update .` — run after this change.

The first owner-authorized exact-SHA CUDA execution reached the real
`TransformerLMIntegration`/WikiText handler and retained a semantic D0 failure:
`tuned-lens macro metric requires exactly fitted native layers 0..11`. The
production boundary correction now filters the 13-key evaluator output to
fitted layers before macro aggregation, while native layer 12 remains parity
only. Failed real attempts now retain truthful explicit execution/backend
markers and `resource_peak: "not measured"`; injected and dispatcher-only
artifacts remain non-promotable. The original setup-failure and semantic-failure
audits are preserved. A corrected SHA requires owner review and a new explicit
authorization before any CUDA rerun.

The retained attempt-1 partial artifact is immutable, sanitized, and
self-digest-valid. It intentionally fails the current artifact validator
because it predates the explicit `execution_attempted`/`execution_backend`
provenance markers and the current resource schema; its run and failure
envelopes independently pass validation. No historical evidence was migrated
or rewritten.
