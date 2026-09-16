# Sprint 79 Queue Position 35 — SAE Reconciliation

## Selected row and ordering

Selected `THY-T05-SPARSE-AUTOENCODER-SAE-ANTHROPIC-2023`, queue position 35, as the earliest executable row after position 33. Position 34 TunedLogitLens is already qualifying at D3 and was skipped. Positions 36 (Disentanglement), 37 (Activation Patching), and 38 (Steering) remain gated by the failed TCAV dependency and real-model causal contracts; position 39 Dictionary Learning lacks an implementation; position 40 OpenVLA lacks checkpoint/license/access and adapter scope.

## Status

The row remains D1. The retained real-model evidence is non-qualifying because cross-seed stability fails its frozen gates. No D3 promotion or threshold change occurred.

## Provenance and evidence

The exact-source run used source `005954c636ae7a45ac3072c69e2c118db044682b`, pinned `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8`, layer 6 hidden states, expanded `prompt-grid-v2-1024` deterministic fixture, and seeds `[0, 1, 2]`. The run used the real `TransformerLMIntegration` boundary and retained no model weights, prompt text, or temporary corpus/checkpoint files.

Target-level config: `artifacts/m14/l06-sae.config.json` (SHA-256 `72ba92dfad42df720ec812e2918ffea3b6a3d76d5df6d2f15a0642c1caa44ba4`). Reconciliation receipt: `artifacts/m14/l06-sae-reconciliation.run.json`. Genuine evidence:

- `artifacts/m14/l06-sae.json` — SHA-256 `c2c4538b42e968663efaba5092f3b4385c95bdf4ee0900f87a428cbae32e3918`.
- `artifacts/m14/l06-sae-run.json` — SHA-256 `68f2d57b77b27fd928af906e7e7a60d7fd1e1a3de1aee899caa1d60812140c22`.
- `artifacts/m14/l06-feature-atlas.json` — SHA-256 `868307f44f5bd9312546bc0df28c6f3fd00a11a9f57b8554ad7bf18d20f68e7d`.

## Metrics and failed gate

The run captured `10,752` tokens with reconstruction MSE `0.7104635182257194`, zero dead features, mean matched cosine `0.8467253367037664` as a diagnostic, minimum matched cosine `0.7387987235614891` against the frozen strict `> 0.85` gate (failed), and alignment quality `0.75` against strict `> 0.7` (passed). Because cross-seed stability requires the minimum-cosine and alignment gates, the row remains D1.

## Validation and integrity

- `uv run pytest tests/test_sae_evaluation.py -q` — **21 passed**.
- `uv run python scripts/validate_evidence_ledger.py --json` — **errors: []**, **40/63 core (63.492063%)**, **40/65 overall (61.538462%)**.
- JSON map, machine-readable ledger override, human ledger, plan table, config, receipt, and summary agree on D1 retention and failed stability gates.
- Previously documented blockers and TCAV failure remain unchanged. Sprint 79 line 595 remains unchecked, and no later plan item was changed.
