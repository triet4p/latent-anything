# Sprint 79 M14 real-system matrix summary

## Scope and accounting

This artifact covers only the current Sprint 79 task at `docs/sprint-plans/sprint-79.md:594`. The authority is `docs/M14_REAL_SYSTEM_VALIDATION.md`; its acceptance semantics and immutable predecessor evidence were not changed. All 24 authority rows are implementation-applicable to this repository. No row is inapplicable or silently omitted.

Lane receipts use `source_sha` as the exact checkout HEAD used for their command. The code-content baseline is unchanged across filing-only commits: L04/L07/L08/L10/L12-L22/L24 receipts executed at `fdd769f842d6fb228270c662db39b261a62bdf40`; L06/L11/L23 executed at `4a159bd9bae0d044109e35b16bbd060160c436fa`; L05/L09 were rerun with retained runners at `e8e65b96d7b2d67e7468b2c33bf04fd13e160e7d`. These later revisions changed only evidence/docs filing plus the two reproducibility runners, not existing lane implementations. The final filing commit is reported at handoff.

- **Completed/accepted:** 6/24 (L01, L03, L06, L09, L11, L23).
- **Partial:** 2/24 (L02, L04; partial records remain explicitly unpromoted as a row-wide completion).
- **Pending:** 8/24 (L05, L07, L08, L13, L14, L15, L16, L22).
- **Externally/prerequisite blocked:** 8/24 (L10, L12, L17, L18, L19, L20, L21, L24).
- **Applicable:** 24/24; **hard blocked:** 8/24; **row-wide completion:** 6/24.

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
| L05 | Observed/pending | `uv run pytest tests/test_density.py tests/test_latent_anything/test_geodesic.py -q` — **40 passed in 20.88s**; `uv run --locked --extra transformers --with huggingface-hub==0.35.3 python scripts/m14_l05_density.py` produced real GPT-2 mean AUROC `0.9750000000000001` and mean AUPRC `0.975609756097561` across seeds `(0,1,2)`. | [`l05-density.json`](m14/l05-density.json). Path-feasibility and separately bound threshold evidence remain required; no promotion. |
| L06 | Accepted | `env LATENT_ANYTHING_RUN_NETWORK=1 uv run pytest tests/test_sae_evaluation.py tests/test_sae_evaluation_network.py -m network -q` — **1 passed, 21 deselected, 1 warning in 23.85s** on real pinned GPT-2; atlas/cross-seed checks passed. | [`l06-sae.json`](m14/l06-sae.json). Real model weight SHA-256 `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`, MIT; local CPU. |
| L07 | Pending | `uv run pytest tests/test_latent_anything/test_activation_patch.py tests/test_latent_anything/test_steering.py tests/test_latent_anything/test_projection.py tests/test_latent_anything/test_lerp.py -q` — **137 passed in 9.00s**. | [`l07-interventions.json`](m14/l07-interventions.json). Cross-adapter paired-effect artifact is absent. |
| L08 | Pending | `uv run pytest tests/test_latent_anything/test_conv_vae.py tests/test_latent_anything/test_conv_vae_evidence.py -q` — **3 passed in 8.54s**. | [`l08-convvae.json`](m14/l08-convvae.json). Standalone held-out D2 metrics/artifact, including the declared >=10% zero-baseline gate, are absent. |
| L09 | Accepted | `env LATENT_ANYTHING_RUN_NETWORK=1 uv run pytest tests/test_diffusers_vae.py tests/test_diffusers_vae_network.py -m network -q` — **1 passed, 5 deselected, 1 warning in 14.75s**; `env LATENT_ANYTHING_RUN_NETWORK=1 uv run --locked --extra diffusers --with huggingface-hub==0.35.3 python scripts/m14_l09_diffusers_vae.py` real parity max error 0.0, seeded sampling and finite shape/dtype checks passed. | [`l09-diffusers-vae.json`](m14/l09-diffusers-vae.json). Pinned weights SHA-256 `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`, MIT; local CPU after cache. |
| L10 | Blocked | `uv run pytest tests/test_diffusers_conditional_network.py -m network -q` — **4 skipped in 3.45s**. | [`l10-diffusion.json`](m14/l10-diffusion.json). High-VRAM model/network/license-card access and remote CUDA owner run are required. |
| L11 | Accepted | `env LATENT_ANYTHING_RUN_NETWORK=1 uv run pytest tests/test_transformer_lm_network.py -m network -q` — **8 passed, 5 deselected in 59.55s**; vocab/hidden-state shape, lens parity, intervention, masking, and cleanup checks passed. | [`l11-gpt2.json`](m14/l11-gpt2.json). Real pinned model weight SHA-256 `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`, MIT; local CPU. |
| L12 | Blocked | `uv run pytest tests/test_latent_anything/test_jepa_checkpoint.py -m network -q` — **1 skipped in 0.23s**. | [`l12-ijepa.json`](m14/l12-ijepa.json). Model-card license/access and checkpoint provisioning are missing. |
| L13 | Pending | `uv run pytest tests/test_latent_anything/test_vq_vae.py -q` — **10 passed in 6.88s**. | [`l13-vq.json`](m14/l13-vq.json). Required standalone perplexity/dead-code/finite encode-decode artifact and codebook-cleanup evidence are absent. |
| L14 | Pending | `uv run pytest tests/test_latent_anything/test_tokenized_world_model.py -q` — **9 passed in 8.66s**. | [`l14-tokenized.json`](m14/l14-tokenized.json). Bounded rollout artifact and preserved early-failure-or-validated-fix evidence are required. |
| L15 | Pending | `uv run pytest tests/test_latent_anything/test_transition.py -q` — **26 passed in 31.64s**. | [`l15-transitions.json`](m14/l15-transitions.json). Real temporal-model gap and complete seeded rollout/state-carry artifact remain. |
| L16 | Pending | `uv run pytest tests/test_reward_value.py tests/test_cem.py tests/test_cem_rollout.py tests/test_mppi.py tests/test_mppi_rollout.py -q` — **32 passed in 5.52s**. | [`l16-planning.json`](m14/l16-planning.json). Policy-grounded D3 gap and declared return/regret trace artifact remain. |
| L17 | Blocked | Not run: `uv run pytest tests/test_latent_anything/test_gaussian_3d_renderer_network.py -m network -q` requires the missing prerequisite. | [`l17-3dgs.json`](m14/l17-3dgs.json). No named checkpoint, access/license metadata, or pinned hash; no unnamed 3DGS substitution. |
| L18 | Blocked | `uv run python scripts/lerobot_dataset_inspection.py lerobot/aloha_sim_insertion_human --revision cc571a3c661df81b566dbfde3d5c1e85fcdf7884 --output artifacts/m14/l18-dataset.json` — **exit 1: `ModuleNotFoundError: No module named 'lerobot'`**. | [`l18-dataset.json`](m14/l18-dataset.json). Linux/LeRobot-capable run is required for upstream dataset/license/schema capture. |
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
l05-density.json            77922c1c2ef009bc987174865cff8e7901ddb47848398ad9b954370d32ab6135
l06-sae.json                a68ea08e0b7e25600be36f288c5e319c977b4f0a2a9a8c860764ee418e5d126b
l07-interventions.json      82d399a33ef359b24003e0518298e384fdf1c3c70a57489469882fc34a753afc
l08-convvae.json            97d17d8e0073cf77f10e5e490a9372327950b729e043bf061ad287dd94129bae
l09-diffusers-vae.json      6f16e1084b86bb09b685fe8710c72fc3f214836c0dd226ad89e62e2f0cdaecf9
l10-diffusion.json          6f06d68e7c6294921be261dc667a61e962eb2daee733f60058a9f28c4e5cdc44
l11-gpt2.json               b9ca81d40ee6bddc7085831a489f715b79db519b17b48a64b4f44c587dd7d4b9
l12-ijepa.json              76f9df6fc31ac7b6bd0d094b12d644c63c754eb53e3e2c49313de303b03f6313
l13-vq.json                 69f0b1a1bc74ff1be401788a921df2adae968461569d862d7c53c876dfb5b916
l14-tokenized.json          01c4a14451fb5775d05076929a6e00d715a5112b51ef7d12c5f411ead3337cef
l15-transitions.json        945bb9560393d1f4acfcb5903f32291bdedceb1372f289bbdc709d5993ab5dae
l16-planning.json           af6057abf567fe7a12c0f0391ce472fa119637dd5703a41e69fbf94915056d08
l17-3dgs.json               1190fbb3645a7488105009a62cf9b237f7d6250d1c383fb8163d33482f75667a
l18-dataset.json            996f1c460a5ace4a80cc6b1c740fc6f5c28e3a460d648d1e7a4711fb2ec4bbc4
l19-act.json                9bf5867f01548e61de544aa01d0ca646e08287441316e72be459519eafeff6bf
l20-diffusion-policy.json   35497393c099d1bfc97cdc827c376bd354ba665294513a85c391921f41972472
l21-smolvla.json            f2386c71913f51ff55f7ea41cec6743da4cae9abcde9a90ddf175587d0d8de50
l22-runtime.json            97cedb1e30716d25ba9cbb9dd1b9d39be180d96788623567d18e8c1569845e21
l23-contract.json           1001c2a8ec6635eaf4a5161ae66fbaf8424686f9ddb5a2a857c26322923da95b
l24-rc.json                  8317d98efb259cf45dc11e2edfae464fdb0b4b7a4beee38a4ecf2f6ac9d88b34

```

## Plan status

`docs/sprint-plans/sprint-79.md:594` is updated to `[~]` with this artifact as the row-level account. The exact blockers above remain explicit; no threshold was weakened, no failed/skipped lane was promoted, and lines 595 onward remain untouched.

## Delivery

- Evidence delivery was pushed on `origin/sprint79-local-gate-remediation`; the exact final revision is reported at handoff.
- `origin/sprint79-local-gate-remediation` was pushed and matched the final local commit.
- The working tree was clean after push.
