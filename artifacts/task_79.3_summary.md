# Task 79.3 — M14 L01 core contract evidence

## Outcome

M14 L01 was executed as one bounded local CPU lane using the existing
`ConvVAE` adapter, `AnalysisPipeline` with the existing `PCA` method, and the
`PipelineResult`, `LatentSpace`, and `LatentValue` contracts. The real
`sklearn.datasets.load_digits` dataset was split once with
`numpy.default_rng(42).permutation`: 1,437 train samples and 360 held-out
samples, with disjoint index digests recorded in the artifact.

All predeclared acceptance criteria passed. The authoritative ledger promotes
only `THY-T01-METRIC-SPACE-VA-VECTOR-SPACE` from D1 to D2. The historical
Sprint 78.38 gap map remains unchanged as a record of the starting state.

## Exact result

- Held-out reconstruction MSE: `0.17170413275226673`.
- All-zero held-out baseline MSE: `0.23585069444444445`.
- Train-pixel-mean diagnostic baseline MSE: `0.0731402344479359`.
- Improvement over all-zero baseline: `0.27197953282807774` (27.20%).
- The ConvVAE reconstruction is worse than the train-pixel-mean baseline;
  this is intentionally retained as a diagnostic, not hidden as a quality
  win. The zero-baseline threshold is only a non-degenerate sanity check.
- D2 scope is limited to real held-out metric/vector/pipeline contract
  evidence; it is not a claim of model-quality or reconstruction superiority.
- First held-out latent pair Euclidean distance: `0.2031720156080903`.
- Train latent utilization: `0.0045607807114720345`.
- Artifact SHA-256: `edf9ebe10ef8e2c8132e2074cee213fb8115bfd2826ddf68aad39f3b65bd4fac`.
- Run record references the same artifact digest and records cleanup as
  completed; no temporary files or network access were used.

The model is the existing `latent_anything.adapters.conv_vae.ConvVAE`, seeded
with 42, latent dimension 4, and 8 epochs. It fits only the train partition.
PCA is also fit only on train latents; held-out latents are transformed after
fit. Inputs remained byte-identical, outputs were float64, shapes were
validated, all measured values were finite, and `LatentValue` arithmetic
returned a new value without mutating either operand.

## Evidence and gates

- Artifact: [`m14/l01-core.json`](m14/l01-core.json)
- Immutable run/provenance record: [`m14/l01-core.run.json`](m14/l01-core.run.json)
- Runner: [`m14_l01_core.py`](../scripts/m14_l01_core.py)
- Focused tests: [`test_m14_l01_core.py`](../tests/test_m14_l01_core.py)
- Ledger: [`evidence-ledger.json`](../docs/evidence-ledger.json), exactly one
  row promoted to D2.
- Validator coverage after promotion: 26/63 core and 26/65 overall; no
  threshold waiver or unrelated row change.
- No commit or push was performed in this lane.
