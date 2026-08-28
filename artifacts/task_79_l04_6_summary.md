# Sprint 79 L04.6 — Direct Logit Lens Phase A

## Outcome

Implemented the offline-ready direct logit-lens execution path for the single
frozen L04.6 use case. The handler uses the concrete
`TransformerLMIntegration.generate()` boundary, captures native indices 0–12,
checks terminal LM-head parity under the integration's post-`ln_f` semantics,
and records target/non-target probabilities without prompt leakage. Held-out
values are aggregated by independent causal group; five frozen seeds provide
reproducible summaries and bootstrap intervals, with shuffled-label,
randomized-token, finite, and terminal-parity controls.

The dispatcher writes atomic partial/run/failure triads. Dependency-injected
handlers are marked `injected_offline_non_eligible`. Direct lens is explicitly
support-only and remains D0; this task performed no model download, CUDA,
network, SSH, commit, push, or evidence promotion.

## Files

- `scripts/_m14_l04_direct_lens.py`
- `scripts/_m14_l04_direct_lens_runtime.py`
- `scripts/_m14_l04_validate_direct_lens.py`
- `scripts/m14_l04_explanations.py`
- `scripts/_m14_l04_artifact.py`
- `scripts/_m14_l04_validate.py`
- `tests/test_m14_l04_direct_lens_handler.py`
- `docs/sprint-plans/sprint-79.md`
- `CHANGELOG.md`

## Gates

- `uv run pytest tests/test_m14_l04_direct_lens_handler.py -q` — 4 passed.
- `uv run pytest tests/test_m14_l04_runner.py tests/test_m14_l04_integrated_gradients_handler.py tests/test_m14_l04_tcav_handler.py -q` — 51 passed.
- Ruff check/format passed for all changed Python files.

The full strict review and repository-wide gates remain with the owner. Real
CUDA execution is intentionally deferred to the owner-approved remote lane;
the earlier logit-lens CUDA parity evidence remains a separate historical
semantic verification and does not satisfy this L04 artifact contract.
