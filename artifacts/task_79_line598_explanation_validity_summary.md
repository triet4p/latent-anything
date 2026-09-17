# Task Summary: Sprint 79 line 598 — explanation-validity controls and coverage gate

**Sprint:** Sprint 79
**Task:** line 598 (Execute explanation-validity controls and confirm the theory ledger meets 95% core / 90% overall D2-or-D3 thresholds)

## Summary of Work

Mapped the exact explanation-validity control set via graphify-first traversal plus the authoritative
artifacts (`docs/evidence-ledger.json`, `scripts/validate_evidence_ledger.py`,
`artifacts/m14/l04-explanations.plan.json`, `docs/VAE_EXPLANATION_BENCHMARK.md`,
`src/latent_anything/evaluation.py`), then ran every applicable independently reachable control with
production code. No real control/validator blocker was found locally, so no source, threshold, or
ledger change was made. Recomputed coverage from authoritative validator output: **41/63 core
(65.079365%)** and **41/65 overall (63.076923%)**, i.e. **19 core and 18 overall qualifiers short** of
the `ceil(0.95 x 63) = 60` / `ceil(0.90 x 65) = 59` gates. All retained D0/D1 failures were preserved;
no denominator change, exclusion, synthetic promotion, duplicate evidence, or rounding was applied.
Line 598 therefore stays **unchecked** with the exact shortfall and blocker inventory below, while all
reachable work is complete.

## Files Modified

* [artifacts/task_79_line598_explanation_validity_summary.md](task_79_line598_explanation_validity_summary.md) - new reproducible summary for this task (this file); no source, ledger, map, queue, or plan file changed

## Testing (exact commands and results)

* `uv run python scripts/validate_evidence_ledger.py --json` — `errors: []`, core `[41, 63,
  0.6507936507936508]`, overall `[41, 65, 0.6307692307692307]`. Human-readable form: `Inventory: 107
  capabilities / core: 41/63 (65.1%) / overall: 41/65 (63.1%)`.
* `uv run python -m scripts.m14_l04_explanations --check` — EXIT 0; `plan_sha256:
  f3c315e356af0ee54d4196cc365ee22bd997b069d18a3e72c6b479f94e0b3e1a`, `content_sha256:
  f5c66f6d947c23f25d41e6aaf8982481feabc92bbff600bd929d27772fb62c0f`, `split_sha256:
  7d788c18212bb1d7e345528c68af6f2bf3e0f745ca77e2d115d74ac3e964121b`, `pair_sha256:
  7225e73c1238b23f6521718c8401331e59653a90499f4b2d75f32dddfe6c1c9c` (frozen plan/fixture intact).
* `uv run python scripts/vae_explanation_benchmark.py` — EXIT 0; rerun reproduces the honest verdict
  `accepts_explanation: false` (reconstruction MSE `0.1885 > 0.1`; input-feature probe beats the latent
  probe, so the input-baseline gate fails). Probe values differ quantitatively from the committed bytes
  under the current sklearn build, so the working-tree overwrite was reverted (`git checkout --`) and
  the committed artifact is unchanged.
* `uv run pytest tests/test_evaluation.py -q` — **5 passed** (explanation-validity accept/reject gates).
* Full explanation-validity focused set (`test_evaluation`, `test_dictionary_learning`,
  `test_m14_l04_tcav_handler`, `test_m14_l04_disentanglement`, `test_m14_l04_activation_patching`,
  `test_m14_l04_steering`, `test_m14_l04_tuned_lens`, `test_m14_l04_direct_lens_handler`,
  `test_m14_l04_explanations`, `test_sae_evaluation`) — **221 passed, 1 skipped**.
* Accepted-control set (`test_m14_l04_disentanglement`, `test_m14_l04_tuned_lens`,
  `test_m14_l04_remote_postprocess`, `test_dictionary_learning`) — **149 passed, 1 skipped**.
* `uv run pytest tests/test_m14_validation_contract.py tests/test_m14_l04_remote_postprocess.py -q`
  — **55 passed** (24 unique authority lanes, referenced paths valid).
* No remote CUDA run was started: every remaining explanation row needs an owner-authorized exact-SHA
  remote rerun, and canonical OpenVLA BF16 is infeasible on the authorized 16 GB host (not retried, per
  standing constraint).

## Control outcomes and confidence intervals (all retained, none relaxed)

* TCAV (`THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018`, D0): held-out accuracy `0.875`
  (95% CI `[0.625, 1.0]`, pass `> 0.6`); Wilson lower `0.5291118177871466` (fail strict `> 0.55`);
  bootstrap lower `1.0` (pass); corrected empirical p `0.24` (fail `<= 0.05`); intervention agreement
  `1.0` (pass). All five frozen controls pass; the two failed semantic gates block promotion.
* AdditiveSteering (`THY-T05-STEERING-VECTORS-...`, D1): target effect `0.05915` (pass `> 0.05`);
  selectivity `0.05432` (pass `> 0.05`); off-target `0.00483` (pass `<= 0.1`); zero-strength,
  shuffled-label, matched-norm, no-mutation, budget controls pass. Required randomized-direction
  control fails (seed 29: `0.32630`, seed 67: `0.10097` vs `<= 0.1`); D3 blocked.
* SAE (`THY-T05-SPARSE-AUTOENCODER-SAE-ANTHROPIC-2023`, D1): `10,752` tokens, MSE `0.7104635182257194`
  finite, zero dead features, mean matched cosine `0.8467253367037664` (diagnostic), minimum matched
  cosine `0.7387987235614891` (fail strict `> 0.85`), alignment quality `0.75` (pass strict `> 0.7`).
  Cross-seed stability fails; D1 retained.
* Accepted explanation rows (not rerun; immutable receipts verified): Disentanglement D2 (exact SHA
  `4d3a4b65...`, `passed_real_cuda`), TunedLogitLens D3 (exact SHA `278a9f76...`, fit seed 79,
  `passed_real_cuda`), ActivationPatching D3 (L049V2 Stage B, source `6af20749...`), DictionaryLearning
  D2 (seed 79, held-out MSE `0.00448003417620913` vs train-mean baseline `0.06497977490954669`, L0
  `2.0`).
* VAE compact benchmark stays D1 (`accepts_explanation: false`); it is a control, not a promotion.

## Coverage arithmetic (validator-backed, no rounding)

* Core: `41/63 = 65.079365%`; gate `ceil(0.95 x 63) = ceil(59.85) = 60`; **shortfall 19**.
* Overall: `41/65 = 63.076923%`; gate `ceil(0.90 x 65) = ceil(58.5) = 59`; **shortfall 18**.
* Binding gate is core. Even promoting all three reachable explanation rows (TCAV, SAE, Steering)
  would reach only 44/63, still 16 short; the gate is unreachable without the non-explanation rows
  below, so no promotion was forced.

## Blocker inventory (22 core + 2 non-core non-qualifying rows; minimal unblocking evidence each)

Core D1 (needs new passing run, same frozen thresholds): `THY-T01-MANIFOLD-HYPOTHESIS` (L02 held-out
ranking AUC `>= 0.55` and latent-vs-raw delta `>= -0.05`; current `0.4560546875` / `-0.4124755859375`);
`THY-T03B-GAUSSIAN-PARAMETERS-LA-LATENT-VARIABLE` (named real-3DGS evidence, L17);
`THY-T05-SPARSE-AUTOENCODER-SAE-ANTHROPIC-2023` (min matched cosine `> 0.85`, L06 real-model rerun);
`THY-T05-STEERING-VECTORS-...` (randomized-direction `<= 0.1` on all seeds, owner-authorized L04 rerun).
Core D0 TCAV: owner-authorized exact-SHA remote CUDA rerun passing Wilson lower `> 0.55` and corrected
p `<= 0.05`. Remaining core D0 rows need owner-approved implementations plus pinned
checkpoints/licenses/CUDA per the gap-map lanes and cannot be closed by explanation controls:
`THY-T02-VAE-HIGGINS-ET-AL-2017` (L08 beta-VAE lane); `THY-T02-VQGAN-ESSER-ET-AL-2021` (L13 approved
VQGAN lane); `THY-T03-NORMALIZING-FLOWS`, `THY-T04-DENSITY-ESTIMATION-TRONG-LATENT`,
`THY-T04-OPTIMAL-TRANSPORT-TRONG-LATENT` (L05 flow/density/transport lanes);
`THY-T03B-DYNAMIC-3DGS` (named dynamic-3DGS checkpoint, L17); `THY-T06-RSSM-...` (real/trained
temporal checkpoint, L15); `THY-T07-POLICY-GRADIENT-...`, `THY-T07-VALUE-EQUIVALENCE-MUZERO`,
`THY-T07-MCTS-TRONG-LATENT` (L16 policy/value/search lanes); `THY-T08-I-JEPA-...` (named
`facebook/ijepa_vith14_1k@be440b1...` checkpoint + license/access, L12); `THY-T08-V-JEPA-...` (named
V-JEPA checkpoint/dataset, L12); `THY-T09-EMA-CODEBOOK-UPDATE`, `THY-T09-RESIDUAL-VQ-...`,
`THY-T09-FINITE-SCALAR-QUANTIZATION-FSQ` (L13 EMA/residual/FSQ lanes); `THY-T09-GAIA-1-...`,
`THY-T09-GENIE-...` (named checkpoints/access, L14). Non-core: `THY-X01-LEWM-...` D1 (named LeWM
checkpoint/access, L12); `THY-X01-OPENVLA` D0 (Linux NVIDIA host with `>= 24` GiB VRAM, real
Transformers adapter, 500 LIBERO-Spatial trials; canonical BF16 measured infeasible on the authorized
RTX 4060 Ti 16 GB and not retried).

## Additional Notes

* Authority agreement: ledger validator (`errors: []`, 41/63, 41/65), gap-map snapshot (41/63,
  41/65), queue blockers, retained L04/SAE/steering/V AE failure receipts, and this summary agree;
  nothing was re-pinned or re-run to alter immutable accepted evidence.
* Sprint 79 line 598 remains `[ ]` unchecked: the gate (`>= 60/63` core and `>= 59/65` overall) is not
  met and the deficit is concrete external evidence, not a reachable local fix.
* Line 599 (exports/registry/entry-points/profiles/CLI/schema/security/sync-async/composition/plugin/
  cache/streaming/tracking verification) is independently actionable and may proceed next; it does not
  depend on the coverage gate.
* Branch left clean; graphify query performed first (`explanation validity TCAV steering
  disentanglement activation patching tuned lens`); no code changed so no graph rebuild was required.
