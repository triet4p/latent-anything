# Review remediation: world-model and adapter evidence

## Scope

Fixed JEPA variance regularization, JEPA checkpoint typing, adapter keyword signatures, VQ-VAE collapse acceptance, tokenizer checkpoint binding, and Sprint 72 benchmark inputs.

## Evidence

- JEPA variance control is computed from trainable context representations and records its source in fit metadata.
- JEPA checkpoint arrays are NumPy-only at the serialization boundary and pass strict Pyright.
- VQ-VAE/tokenized/JEPA public methods accept the frozen Protocol keyword names.
- VQ-VAE acceptance uses strict `perplexity > 1.0`; the reproduced collapsed run is rejected.
- Tokenized dynamics bind to a deterministic fitted-checkpoint/schema digest and reject tokenizer mutation before fitting, prediction, evaluation, and rollout.
- Sprint 72 now fits from raw digit image trajectories encoded by the fitted tokenizer, measures that provenance, and records the collapsed one-code result as D1 rather than treating perfect constant-token metrics as D2 evidence.

## Verification

VQ-VAE, JEPA, tokenized-world-model, and both reproduction-script tests pass.
