# Sprint 79 L04.9 Phase A — True Activation Patching

## Scope

Implemented the offline-tested, network-gated true clean/corrupted activation
interchange path for `TrueActivationPatching`. The implementation targets the
pinned `openai-community/gpt2` revision through the concrete
`TransformerLMIntegration` boundary; `ModelAdapter` remains `N/A` by design.

This is Phase A only. No model was downloaded, no CUDA execution was run, no
remote capture was attempted, and no D3 evidence was promoted.

## Contract

- Clean hidden activation replaces the corrupted activation at layer 6 / native
  hidden-state index 7 and target position.
- Separate controls patch adjacent layer 5 at the target token and layer 6 at
  the independently resolved previous valid token; both clean and corrupted
  positions are resolved separately.
- The compatibility control key is `shuffled_direction`; its metadata explicitly
  identifies a deterministic split-preserving non-self shuffled donor
  activation.
- Strength diagnostics are `[0.0, 0.25, 0.5, 1.0]`; strength `1.0` is the true
  interchange acceptance point.
- Margins always use fixed GPT-2 classes `" true" - " false"`.
- Recovery aggregates causal pairs/groups before deterministic 2,000-replicate
  bootstrap. The clean/corrupted denominator is finite and strictly greater
  than `1e-12` in absolute value.
- Train and holdout shuffled donors are independently deranged within their
  split, with exact domain/range/non-self checks; layer and token off-target
  metrics are retained separately before the combined worst-case gate.
- Acceptance remains fail-closed: recovery CI lower bound `> 0.10`, combined
  worst-case held-out off-target absolute effect `<= 0.10`, zero-strength error
  `<= 1e-6`, finite controls, no mutation, and resource budget pass.

## Changed implementation

- Added private handler `scripts/_m14_l04_activation_patching.py`.
- Added fail-closed validator `scripts/_m14_l04_validate_activation_patching.py`.
- Wired dispatcher and artifact promotion for the exact record ID
  `THY-T05-ACTIVATION-PATCHING`.
- Added offline contract tests in `tests/test_m14_l04_activation_patching.py`.

The public VAE `ActivationPatch`, public transformer integration API, and frozen
L04 plan were not changed.

## Verification

Focused L04.9 and runner tests pass. Ruff and Pyright pass for the changed
implementation. Real CUDA verification remains a separate owner-authorized
phase after review, commit/push, and the current retention protocol.
