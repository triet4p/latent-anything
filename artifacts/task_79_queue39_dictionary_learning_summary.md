# Sprint 79 queue position 39 — Dictionary Learning promotion

## Selection and ordering

Queue position 39, `THY-T05-DICTIONARY-LEARNING`, was selected after position 38 AdditiveSteering. Position 39 was the earliest dependency-order row after confirming position 39 had no dedicated implementation and position 40 OpenVLA remained blocked on checkpoint/license/access and adapter scope. Position 40 and Sprint 79 line 595 were not modified.

## Target-level evidence

The dedicated implementation is `src/latent_anything/dictionary_learning.py`, tested by `tests/test_dictionary_learning.py`, and executed by `scripts/m14_l06_dictionary_learning.py`. The predeclared contract is `artifacts/m14/l06-dictionary-learning.config.json`; genuine evidence and immutable receipt are `artifacts/m14/l06-dictionary-learning.json` and `artifacts/m14/l06-dictionary-learning.run.json`.

The offline deterministic fixture uses seed `79`, 600 samples × 12 features, an 80/20 train/held-out split, and scikit-learn `DictionaryLearning` with 8 components and two-sparse OMP codes. Dataset digest: `f13c3abefe009a306ae434df0fd45578aa2fa5f4a381330db428ae2489f0a1f5`. The source checkout used by the measured run was `806efa78b7d2b9f96b086f933ec5c7c6bd19968b`.

## Outcome and metrics

The row is promoted from D0 to D2. Held-out reconstruction MSE is `0.013416282559909032` versus the train-mean baseline `0.06497977490954669`; the strict ratio gate `< 0.5` passes (`0.2064685908589988`). Held-out mean L0 is `2.0` against `<= 2.0`. Train reconstruction MSE is `0.010704513402240056`; dictionary shape is `[8, 12]`. Finite metrics, disjoint train/held-out indices, no-input-mutation, and dictionary-shape controls all pass. Artifact payload digest is `dd768b79a1f5a83123ddda4bb743e720f0bdc9e349836d42c419dc5689df19a4`.

This is bounded D2 algorithm evidence only. It does not promote the separate real-GPT-2 SAE row or make a D3 named-model claim.

## Verification

- `uv run pytest tests/test_dictionary_learning.py -q` — **3 passed**.
- `uv run python scripts/m14_l06_dictionary_learning.py` — accepted `true`; deterministic rerun reproduced the same artifact digest and metrics.
- `uv run python scripts/validate_evidence_ledger.py --json` — `errors: []`; **41/63 core (65.079365%)**, **41/65 overall (63.076923%)**.

The gap map, queue, evidence ledger, plan table, configuration, artifact, and receipt agree on D2. Position 40 OpenVLA remains D0 with its checkpoint/license/access and adapter-scope blocker. Sprint 79 line 595 remains unchecked.
