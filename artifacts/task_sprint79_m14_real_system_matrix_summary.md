# Sprint 79 M14 real-system matrix summary

## Scope and accounting

This artifact covers only the current Sprint 79 task at `docs/sprint-plans/sprint-79.md:594`. The authority is `docs/M14_REAL_SYSTEM_VALIDATION.md`; its acceptance semantics and immutable predecessor evidence were not changed. All 24 authority rows are implementation-applicable to this repository. No row is inapplicable or silently omitted.

Lane receipts use `source_sha` as the exact checkout HEAD used for their command. The code-content baseline is unchanged across filing-only commits: L04/L10/L12/L14/L15/L17-L22/L24 receipts executed at `fdd769f842d6fb228270c662db39b261a62bdf40`; L11/L23 executed at `4a159bd9bae0d044109e35b16bbd060160c436fa`; L08/L13/L16 executed at `e8e65b96d7b2d67e7468b2c33bf04fd13e160e7d`; L05/L06 executed at `99effa0eac5f41240c47320cf650218bc268ec43`; L07 executed at `5e06fc9483b2da7bf105c875bc998f47ba94fe4e`; L09 measured artifact rerun executed at `89d2cd4b7ef25c730f2aee768817263f9036c35b`. These later revisions changed only evidence/docs filing plus reproducibility runners, not existing lane implementations. The final filing commit is reported at handoff.

- **Completed/accepted:** 8/24 (L01, L03, L05, L07, L08, L09, L13, L23).
- **Partial:** 2/24 (L02, L04).
- **Pending/observed:** 6/24 (L06, L11, L14, L15, L16, L22).
- **Externally/prerequisite blocked:** 8/24 (L10, L12, L17, L18, L19, L20, L21, L24).
- **Applicable:** 24/24; **hard blocked:** 8/24; **row-wide completion:** 8/24.

The Sprint 79 checkbox therefore remains `[~]`, not `[x]`. The release evidence ledger remains below its independent release thresholds; no percentage waiver or skipped network lane was promoted.

## Immutable accepted and partial evidence

| Row | Result and exact evidence |
|---|---|
| L01 | **Accepted D2.** Existing `artifacts/m14/l01-core.json` and `l01-core.run.json` are immutable accepted evidence. The recorded runner command is `uv run python scripts/m14_l01_core.py`; status `accepted`; sklearn digits held-out contract, finite schema/digest, no mutation, and zero-baseline improvement passed. Artifact SHA-256: `edf9ebe10ef8e2c8132e2074cee213fb8115bfd2826ddf68aad39f3b65bd4fac`. |
| L02 | **Partial D2 (4/6 records).** Existing immutable `artifacts/m14/l02-geometry.json` and `l02-geometry.run.json` record accepted SLERP/LERP/Riemannian/latent-operation records and honest failures for the manifold and DTW records. Recorded command: `uv run python -m scripts.m14_l02_geometry`; artifact SHA-256 `97d26f5fb1d12dc00658ff9cfec12a91b080bb4a1bb3cd96e7bc2ed70f9e5a58`. No stable Frechet or physical-trajectory claim is made. |
| L03 | **Accepted D2 forward-only.** Existing immutable `artifacts/m14/l03-analysis.json`, `l03-analysis.run.json`, and `l03-analysis.attempt4.capture-audit.json` record three accepted GPT-2 hidden-state records at `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8`, exact execution command `uv run python -m scripts.m14_l03_analysis --run-real`, source SHA `bb0da6fdc4fb00950ce4cb574ec83e8a9344db8`, and artifact SHA-256 `60bda13a4bbf68bbb6c9308cc813913fa653c37fba368fe1e4ea7a1f898ce06b`. Remote evidence recorded NVIDIA GeForce RTX 4060 Ti/CUDA 12.8, wrapper exit 0, captured output, removed disposable checkout `/tmp/remote-cuda-test.F0de65/repo`, and removed isolated caches (`uv-cache`, `torch-extensions`, `cuda-cache`, `hf-cache`). |
| L04 | **Partial.** Capability receipt `artifacts/m14/l04-explanations.json` points to immutable accepted TunedLogitLens D3 promotion and Disentanglement D2 records. The remaining IntegratedGradients, TCAV, DirectLogitLens, TrueActivationPatching, and AdditiveSteering use cases are not complete; no row-wide promotion is claimed. Existing L04 evidence uses the authority's direct authenticated PowerShell `ssh.exe` transport exception, not the generic remote-cuda wrapper. |

## Current-checkpoint lane executions and blockers

Each row below has an independently verifiable receipt under `artifacts/m14/`. A passing focused test is reported as implementation evidence only; it is not promoted to the row's real-system D2/D3 acceptance without the authority's artifact contract.

| Row | Status | Exact command/result | Receipt and unresolved requirement |
|---|---|---|---|
| L05 | Accepted | `uv run --locked --extra transformers --with huggingface-hub==0.35.3 python scripts/m14_l05_density.py` persisted real GPT-2 held-out density and geodesic evidence: mean AUROC `0.9750000000000001` across seeds `(0,1,2)`, mean AUPRC `0.975609756097561`, path finite with endpoints preserved and converged. | [`l05-density.json`](m14/l05-density.json) and [`l05-density-run.json`](m14/l05-density-run.json), artifact SHA-256 `04a6b8313a90514d63021cebaad7135c1430792ba0858e7a1395eae42379b4e9`. Pinned GPT-2 revision/weights, exhaustive 80/40/20/40 prompt split digests, Python/NumPy/Torch/Transformers/hub versions, CPU RSS peak `1387851776`, HF cache-only policy, cleanup, predeclared AUROC minimum `0.90`, and finite exact-endpoint path threshold are recorded. |
| L06 | Observed/pending | `env LATENT_ANYTHING_RUN_NETWORK=1 uv run --locked --extra transformers --with huggingface-hub==0.35.3 python scripts/m14_l06_sae.py` persisted real GPT-2 layer-6 SAE/atlas evidence: `234` tokens, reconstruction MSE `0.9514871142121589`, atlas entries `64`; cross-seed matched cosine mean/min and alignment quality were `0.0/0.0/0.0` across seeds `(0,1,2)`. | [`l06-sae.json`](m14/l06-sae.json), [`l06-sae-run.json`](m14/l06-sae-run.json) SHA-256 `08ccc689f05602924a425f31b96d8462e1af9c2a053795ded87a13aa2b5fa3d3`, and atlas [`l06-feature-atlas.json`](m14/l06-feature-atlas.json) SHA-256 `bed0ca1c913006fb124cc2a87c8fab1a18d5b0d44cb8af383ff16fbe69bb224d`. Real GPT-2/corpus revisions, prompt digest, versions, CPU RSS `1314619392`, cache-only policy, cleanup, and predeclared stability thresholds (cosine `0.85`, alignment `0.7`) are recorded; no promotion because cross-seed stability failed. |
| L07 | Accepted | `env LATENT_ANYTHING_RUN_NETWORK=1 uv run --locked --extra transformers --with huggingface-hub==0.35.3 python scripts/m14_l07_interventions.py` persisted paired real GPT-2 hidden-state and VAE-latent controls: GPT-2 steering effect `0.5000000000002008`, VAE activation-patch effect `20.72073172096069`; `SubspaceProjection`, LERP, steering, and activation-patch shape, finite, reversibility, and no-mutation gates passed. | [`l07-interventions.json`](m14/l07-interventions.json) and [`l07-interventions-run.json`](m14/l07-interventions-run.json), artifact SHA-256 `1bd8481d9aae4f99fd16205393acdb3ab5cf9150aa9ee92b87edbf868c7267e4`. Pinned GPT-2 revision/weights, sklearn digits revision/license, seeds, CPU RSS `1092190208`, cache-only policy, and cleanup are recorded. |
| L08 | Accepted | `uv run pytest tests/test_latent_anything/test_conv_vae.py tests/test_latent_anything/test_conv_vae_evidence.py -q` — **3 passed in 8.93s**; `uv run python -c "from pathlib import Path; from scripts.conv_vae_heldout_benchmark import main; print(main(Path('artifacts/m14')))"` produced held-out D2 metrics: reconstruction MSE `0.17170413275226673`, zero-baseline improvement `0.27197953282807774`, latent utilization `0.0045607807114720345`, steering decode delta `0.01362483808124024`, disjoint split and composition checks passed. | [`l08-convvae.json`](m14/l08-convvae.json) plus [`conv_vae_heldout_benchmark.json`](m14/conv_vae_heldout_benchmark.json). Pinned seed `42`, scikit-learn `1.9.0`, BSD-3-Clause bundled digits data, offline CPU run; artifact SHA is recorded in the receipt. |
| L09 | Accepted | `env LATENT_ANYTHING_RUN_NETWORK=1 uv run pytest tests/test_diffusers_vae.py tests/test_diffusers_vae_network.py -m network -q` — **1 passed, 5 deselected, 1 warning in 14.75s**; `env LATENT_ANYTHING_RUN_NETWORK=1 uv run --locked --extra diffusers --with huggingface-hub==0.35.3 python scripts/m14_l09_diffusers_vae.py` persisted parity/sampling/shape metrics. | [`l09-diffusers-vae.json`](m14/l09-diffusers-vae.json) and [`l09-diffusers-vae-run.json`](m14/l09-diffusers-vae-run.json), artifact SHA-256 `a867896227e67659f4ab50ea7d68dcdd268b28ee9d8c29efe32225e6d40d016a`. Model `stabilityai/sd-vae-ft-mse@31f26fdeee1355a5c34592e401dd41e45d25a493`; weights SHA `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`, MIT; CPU/Windows-11-10.0.26200-SP0; Python 3.13.3; NumPy 2.4.6; Torch 2.10.0; Diffusers 0.39.0; huggingface-hub 0.35.3; RSS peak 921952256 bytes; input seed 7, sample seeds 123/123/124; final rerun HF-cache-only policy. |
| L10 | Blocked | `uv run pytest tests/test_diffusers_conditional_network.py -m network -q` — **4 skipped in 3.45s**. | [`l10-diffusion.json`](m14/l10-diffusion.json). High-VRAM model/network/license-card access and remote CUDA owner run are required. |
| L11 | Observed/pending | `env LATENT_ANYTHING_RUN_NETWORK=1 uv run pytest tests/test_transformer_lm_network.py -m network -q` — **8 passed, 5 deselected in 59.55s**; vocab/hidden-state shape, lens parity, intervention, masking, and cleanup checks passed. | [`l11-gpt2.json`](m14/l11-gpt2.json). Complete lane artifact/capture contract remains; no row-wide promotion. |
| L12 | Blocked | `uv run pytest tests/test_latent_anything/test_jepa_checkpoint.py -m network -q` — **1 skipped in 0.23s**. | [`l12-ijepa.json`](m14/l12-ijepa.json). Model-card license/access and checkpoint provisioning are missing. |
| L13 | Accepted | `uv run pytest tests/test_latent_anything/test_vq_vae.py -q` — **10 passed in 5.61s**; `uv run python scripts/vq_vae_digits_evidence.py` generated pinned CPU D2 evidence: codebook perplexity `13.090496630645841`, dead-code rate `0.0`, discrete reconstruction MSE `0.17885987018827126`, finite encode/decode checks passed. | [`l13-vq.json`](m14/l13-vq.json) plus [`../vq_vae_digits_evidence.json`](../vq_vae_digits_evidence.json). Seed `42`, `compact-vq-vae-v1`, scikit-learn digits `1.9.0`, BSD-3-Clause bundled data, offline; artifact/config hashes are recorded in the receipt and generated-codebook cleanup is documented. |
| L14 | Pending | `uv run pytest tests/test_latent_anything/test_tokenized_world_model.py -q` — **9 passed in 8.66s**. | [`l14-tokenized.json`](m14/l14-tokenized.json). Bounded rollout artifact and preserved early-failure-or-validated-fix evidence are required. |
| L15 | Pending | `uv run pytest tests/test_latent_anything/test_transition.py -q` — **26 passed in 31.64s**. | [`l15-transitions.json`](m14/l15-transitions.json). Real temporal-model gap and complete seeded rollout/state-carry artifact remain. |
| L16 | Pending | `uv run pytest tests/test_reward_value.py tests/test_cem.py tests/test_cem_rollout.py tests/test_mppi.py tests/test_mppi_rollout.py -q` — **32 passed in 6.27s**. | [`l16-planning.json`](m14/l16-planning.json). Policy-grounded D3 gap and declared return/regret trace artifact remain. |
| L17 | Blocked | Not run: `uv run pytest tests/test_latent_anything/test_gaussian_3d_renderer_network.py -m network -q` requires the missing prerequisite. | [`l17-3dgs.json`](m14/l17-3dgs.json). No named checkpoint, access/license metadata, or pinned hash; no unnamed 3DGS substitution. |
| L18 | Blocked | `uv run python scripts/lerobot_dataset_inspection.py lerobot/aloha_sim_insertion_human --revision cc571a3c661df81b566dbfde3d5c1e85fcdf7884 --output artifacts/m14/l18-dataset.json` — **exit 1: `ModuleNotFoundError: No module named 'lerobot'`**. | [`l18-dataset.json`](m14/l18-dataset.json). LeRobot is not installed locally; Linux/LeRobot-capable upstream dataset/license/schema capture is required. |
| L19 | Blocked | `uv run pytest tests/test_lerobot_act.py::test_pinned_public_act_checkpoint_pair_loads_through_lerobot_factories -m network -q` — **1 skipped in 3.08s**. | [`l19-act.json`](m14/l19-act.json). Linux CUDA host and model-card license/access capture unavailable. |
| L20 | Blocked | `uv run pytest tests/test_lerobot_diffusion.py::test_pinned_public_diffusion_checkpoint_pair_loads_through_lerobot_factories -m network -q` — **1 skipped in 3.19s**. | [`l20-diffusion-policy.json`](m14/l20-diffusion-policy.json). Upstream model/data access, license capture, and remote CUDA/Linux execution unavailable. |
| L21 | Blocked | `uv run pytest tests/test_lerobot_smolvla.py::test_smolvla_gpu_checkpoint_intervention_lane -m network -q` — **1 skipped in 3.27s**. | [`l21-smolvla.json`](m14/l21-smolvla.json). Required Linux GPU (~16 GB), model/data access, and license capture unavailable. |
| L22 | Pending | `uv run pytest tests/test_portable.py tests/test_portable_results.py tests/test_artifact_store.py tests/test_latent_anything/test_cache.py tests/test_sprint75_streaming.py tests/test_run_record.py tests/test_run_record_portable.py tests/test_experiment_recorder.py tests/test_mlflow_recorder.py tests/test_wandb_recorder.py -q` — **116 passed in 15.86s**. | [`l22-runtime.json`](m14/l22-runtime.json). Complete M14 runtime artifact and optional tracking-account evidence remain. |
| L23 | Accepted | Full contract command — **115 passed in 15.91s, 19 warnings**; built-in/group probe returned `L23_BUILTINS_32_GROUPS_5_OK`; installation test exercises separately installed hello-world plugin; existing profile artifact records 39/39 base/profile lanes across Python 3.12/3.13/3.14 Windows and ubuntu-latest. | [`l23-contract.json`](m14/l23-contract.json). Local D1 contract requirements complete; no external remainder for this row. |
| L24 | Blocked | Existing accepted evidence is referenced, not rewritten: [`clean environment summary`](../task_sprint79_clean_environment_matrix_summary.md) and [`release audit summary`](../task_sprint79_release_audits_summary.md). | [`l24-rc.json`](m14/l24-rc.json). External Actions account and unresolved M14 evidence thresholds remain. |

## Matrix contract verification

- `uv run pytest tests/test_m14_validation_contract.py -q` — **3 passed in 0.24s**. This verifies the authority table has 24 unique lanes and all referenced repository paths remain valid.
- Existing accepted Sprint 79 artifacts were not rewritten: `task_sprint79_local_gate_remediation_summary.md`, `task_sprint79_clean_environment_matrix_summary.md`, and `task_sprint79_release_audits_summary.md`.
- No remote CUDA run was started for a blocked lane: the authority requires exact model/checkpoint/access prerequisites before remote execution, and the generic `remote-cuda-test` invariant forbids substituting a skipped local lane. L03's retained remote cleanup evidence is recorded above; L04 follows its explicit direct-PowerShell transport exception.
- No later Sprint 79 task was started or marked complete.

## Receipt hashes

The newly created row receipts are immutable once committed. Their SHA-256 values at creation are:

```text
l04-explanations.json       5d896a9b8603c340fe0203e303870f78109a69721ac8729924c1ce384ac1989e
l05-density.json            47a2de5dece3473a7f07a098601d75a8249bea0fcf28c2c6cbf0246a137fb450
l06-sae.json                aa1681a2b72cdd9030b324d4c32fc4e17f7e80bc7fd1bbda7136c779f943ba5b
l07-interventions.json      086d585833eb881fe9c8309990133e204041012f9600857272cf91465478d499
l08-convvae.json            589b18984def9c9289f529f0f2c09705016488b8ed6696e825620c0b929c7b8d
l09-diffusers-vae.json      d89f05e30f8aa240b6e04b2151983e2c836c5ddff66dd462b234f0f537f83575
l10-diffusion.json          6f06d68e7c6294921be261dc667a61e962eb2daee733f60058a9f28c4e5cdc44
l11-gpt2.json               4962aec35b4852333ba4c58e3af755c65164f4d7f2e83a15a98ec7e14c3c13e7
l12-ijepa.json              76f9df6fc31ac7b6bd0d094b12d644c63c754eb53e3e2c49313de303b03f6313
l13-vq.json                 2ec6243202cf0ff67b88fd790d8d15ae7e0e2fb3e6479305e393060c3dbcbe45
l14-tokenized.json          01c4a14451fb5775d05076929a6e00d715a5112b51ef7d12c5f411ead3337cef
l15-transitions.json        945bb9560393d1f4acfcb5903f32291bdedceb1372f289bbdc709d5993ab5dae
l16-planning.json           47b5f75fbfac4d8355504010c471fe5fe4d690144a6c8ee4bd3e2d42f8b4687d
l17-3dgs.json               1190fbb3645a7488105009a62cf9b237f7d6250d1c383fb8163d33482f75667a
l18-dataset.json            996f1c460a5ace4a80cc6b1c740fc6f5c28e3a460d648d1e7a4711fb2ec4bbc4
l19-act.json                9bf5867f01548e61de544aa01d0ca646e08287441316e72be459519eafeff6bf
l20-diffusion-policy.json   35497393c099d1bfc97cdc827c376bd354ba665294513a85c391921f41972472
l21-smolvla.json            f2386c71913f51ff55f7ea41cec6743da4cae9abcde9a90ddf175587d0d8de50
l22-runtime.json            97cedb1e30716d25ba9cbb9dd1b9d39be180d96788623567d18e8c1569845e21
l23-contract.json           1001c2a8ec6635eaf4a5161ae66fbaf8424686f9ddb5a2a857c26322923da95b
l24-rc.json                 8317d98efb259cf45dc11e2edfae464fdb0b4b7a4beee38a4ecf2f6ac9d88b34

```

## Plan status

`docs/sprint-plans/sprint-79.md:594` is updated to `[~]` with this artifact as the row-level account. The exact blockers above remain explicit; no threshold was weakened, no failed/skipped lane was promoted, and lines 595 onward remain untouched.

## Delivery

- Evidence delivery was pushed on `origin/sprint79-local-gate-remediation`; the exact final revision is reported at handoff.
- `origin/sprint79-local-gate-remediation` was pushed and matched the final local commit.
- The working tree was clean after push.
