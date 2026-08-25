# Sprint 34 Carryover Task C34.2 — Held-Out ConvVAE Benchmark

**Status:** Complete

## Summary

Added a deterministic, train-only ConvVAE integration benchmark over the
bundled sklearn digits dataset. The benchmark splits 1,797 normalized 8x8
images into 1,437 train and 360 held-out samples using `default_rng(42)`, fits
the ConvVAE/PCA/SAE/steering direction only on train data, and applies all
learned transforms to held-out values.

## Acceptance result

- Held-out reconstruction MSE: `0.17170413275226673`.
- All-zero baseline MSE: `0.23585069444444445`.
- Improvement over all-zero baseline: `27.197953282807774%` (threshold: 10%).
- Train-pixel-mean diagnostic MSE: `0.0731402344479359`; the benchmark keeps
  this stronger baseline visible and does not claim to beat it.
- Train latent utilization: `0.0045607807114720345` (threshold: `0.001`).
- Steering direction norm: `0.9999999999999999`; decoded held-out steering
  mean absolute delta: `0.01362483808124024`.
- Runtime: `1.6414667000062764` seconds on Windows CPU, under the 30-second
  advisory budget.
- All hard acceptance checks passed; evidence level is D2 for this compact
  trained CPU integration lane.

## Files

- `scripts/conv_vae_heldout_benchmark.py` — benchmark, split, thresholds, and
  provenance.
- `tests/test_conv_vae_heldout_benchmark.py` — deterministic split and
  independent artifact/threshold regression checks.
- `artifacts/conv_vae_heldout_benchmark.json` — measured evidence.
- `artifacts/conv_vae_heldout_benchmark_config.json` — reproducible config.

## Validation

- `uv run pytest tests/test_conv_vae_heldout_benchmark.py -q` — **2 passed**.
- `uv run python scripts/conv_vae_heldout_benchmark.py` — **passed**, wrote
  the two artifacts above.
- `graphify update .` — completed with no code-graph topology change; the
  existing graph outputs remain at 10,046 nodes, 19,541 edges, and 884
  communities. Graphify reported 48 zero-node JSON/source warnings (known
  `#1666` behavior) and left outputs untouched.

## Graph refresh

- Command: `graphify update .`
- Result: exit 0; no code-graph topology changes detected; 48 known zero-node
  JSON/source warnings.
