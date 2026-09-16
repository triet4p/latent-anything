# Sprint 79 queue position 38 — AdditiveSteering reconciliation

## Selection and scope

Queue position 38, `THY-T05-STEERING-VECTORS-ZOU-ET-AL-2023-REPRESENTATION-ENGINEERING`, was the earliest next runnable row after the authoritative evidence review: queue positions 36 (Disentanglement) and 37 (Activation Patching) already satisfy their bounded target contracts. This row reconciles the retained owner-authorized real-model run only. It does not change Sprint 79 line 595, queue positions 39–40, or later plan items.

## Target-level provenance

- Source checkpoint: `7ad6648a9bc5d22793b63b764cb9f93990a247e8`
- Model: `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8` (MIT)
- Runtime: `TransformerLMIntegration`, CUDA `cuda:0`, layer 6/native hidden-state index 7
- Target tokens: `true=2081`, `false=3991`
- Seeds: `[17, 29, 41, 53, 67]`; strengths: `[0, 0.25, 0.5, 1]`
- Fixture: `artifacts/m14/l04-prompt-factor-fixture.jsonl`
- Target configuration and receipt: `artifacts/m14/l04-steering.config.json`, `artifacts/m14/l04-steering-reconciliation.run.json`
- Immutable evidence: `artifacts/m14/l04-explanations.AdditiveSteering.attempt1.partial.json`, `.run.json`, `.failure.json`, and the sanitized SSH retention audit

## Outcome

The run remains a completed real-CUDA D0 diagnostic and the theory row remains D1. Target effect was `0.059151649475097656` against strict `> 0.05`; selectivity was `0.05431842803955078` against strict `> 0.05`; and off-target token effect was `0.004833221435546875` against `<= 0.1`. Zero-strength identity, shuffled-label, matched-norm, no-mutation, and resource-budget controls passed.

The required randomized-direction control failed for seeds 29 (`0.32630348205566406`) and 67 (`0.10096931457519531`) against `<= 0.1`. This frozen negative control blocks causal D3 promotion. The failure is recorded without threshold relaxation, relabeling, or unsupported claims; a corrected owner-authorized rerun is required before promotion.

## Verification

- `uv run pytest tests/test_m14_l04_steering.py tests/test_latent_anything/test_steering.py -q` — **63 passed**
- Evidence ledger validator — required to remain `errors: []`, with coverage unchanged at **40/63 core** and **40/65 overall**

The machine-readable gap map and both ledgers preserve D1 status, the failed gate, exact evidence paths, and the blockers for queue positions 39 (no dedicated dictionary-learning implementation) and 40 (OpenVLA checkpoint/license/access/adapter scope absent).
 
