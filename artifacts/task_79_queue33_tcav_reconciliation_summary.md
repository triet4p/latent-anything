# Sprint 79 Queue Position 33 — TCAV Reconciliation

## Selected row and ordering

Selected `THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018`, queue position 33, as the earliest dependency-order row after position 27 with its L03 prerequisite, owner authorization, pinned fixture, and remote CUDA runtime available. Positions 28 and 32 (linear and nonlinear probing) were already qualifying. Positions 29 (V-JEPA), 30 (Genie), and 31 (policy-gradient) remain blocked by missing implementation/checkpoint/access prerequisites.

## Status

The row remains D0. This is a retained semantic failure, not a promotion. `accepted_gap_ids=[]`, `accepted_record_ids=[]`, `evidence_eligible=false`, and `acceptance=false` remain truthful.

## Provenance and evidence

The owner-reviewed exact-SHA recovery used source `5c38b63f01d280939790e415de699ab285a228de`, pinned `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8`, and concrete `TransformerLMIntegration` (ModelAdapter is N/A), at transformer layer 6/native hidden-state index 7. The immutable evidence triplet is:

- `artifacts/m14/l04-explanations.TCAV.attempt1.partial.json` — SHA-256 `adbb6ac51b9222767c9c15e0c18ae716b3d0a5f94eb9de73e0bd6bda3de2d523`.
- `artifacts/m14/l04-explanations.TCAV.attempt1.run.json` — SHA-256 `123240b06bfbc7578c9b0264a903d2c729322e9ed31028a0519fa98d1aa12fd1`.
- `artifacts/m14/l04-explanations.TCAV.attempt1.failure.json` — SHA-256 `1f00beb47b842b77b36f386f1cf9ecefb1e2a28fbb7dcab4c87cdcf6021dec6b`.
- `artifacts/m14/l04-explanations.ssh.TCAV.attempt3.audit.json` — SHA-256 `ad4fc26c87ea89e18f6ef57342f6469b072eca24f409d453468624a3dacf0472`.

The target-level config is `artifacts/m14/l04-tcav.config.json` (SHA-256 `850cf327341dc7b3da405538121279d090f5c7d74aa0018b25dc822525fcb28a`). The reconciliation receipt is `artifacts/m14/l04-tcav.run.json`.

## Metrics and failed gates

Held-out accuracy was `0.875`; Wilson lower bound was `0.5291118177871466` against strict `> 0.55` (failed); bootstrap lower was `1.0`; corrected empirical p was `0.24` against `<= 0.05` (failed); intervention agreement was `1.0`. Shuffled-label, random-direction, matched-norm, off-target-token, and zero-strength controls all passed. The remote device was an NVIDIA GeForce RTX 4060 Ti; transport, bundle, and cleanup audit gates passed. No rerun is authorized.

## Validation and integrity

- `uv run pytest tests/test_m14_l04_tcav_handler.py tests/test_m14_l04_explanations.py -q` — **27 passed**.
- `uv run python scripts/validate_evidence_ledger.py --json` — **errors: []**, **40/63 core (63.492063%)**, **40/65 overall (61.538462%)**.
- The map snapshot, JSON ledger override, human ledger, plan table, and queue-position record now agree on D0 retention and the failed gates.
- Queue positions 2–4, 6–9, and 12 remain blocked as previously documented. Line 595 stays unchecked; later Sprint 79 plan items are unchanged.
