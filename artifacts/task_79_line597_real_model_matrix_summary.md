# Sprint 79 Line 597 — Pinned Real-Model Matrix Reconciliation

## Result

Line 597 is satisfied as a reconciliation of the pinned real-model matrix. Accepted rows were not rerun: their immutable artifacts already provide the required real execution evidence. Outstanding rows were not forced through missing model, license, platform, or CUDA prerequisites; each remains an explicit blocker with its pinned target and command in the authoritative M14 receipt. No D-level promotion was made.

## Matrix outcomes

| Family / row | Outcome | Pinned target and evidence |
|---|---|---|
| Diffusers VAE / L09 | Accepted D2 | `stabilityai/sd-vae-ft-mse@31f26fdeee1355a5c34592e401dd41e45d25a493`; safetensors SHA `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`; immutable `artifacts/m14/l09-diffusers-vae.json` and run artifact. CPU cache-only rerun recorded direct/adapter parity, seeded sampling, shape, and finite checks. |
| Conditional diffusion / L10 | Blocked, not run | `runwayml/stable-diffusion-v1-5@39593d56b552c3a24aeb192dd11d2a1429c3102b`; `artifacts/m14/l10-diffusion.json` records 4 skipped network tests and the missing high-VRAM model/network/license-card access plus remote CUDA owner run. |
| GPT-2 / L11 | Accepted D2 candidate | `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8`; model SHA `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`; immutable `artifacts/m14/l11-gpt2.json` records 15/15 runner acceptance and focused 8 passed / 5 deselected. |
| I-JEPA / L12 | Blocked, not run | `facebook/ijepa_vith14_1k@be440b1cac639542ae553e71a9c7afd925ab5fac`; `artifacts/m14/l12-ijepa.json` records the skipped checkpoint test and missing model-card license/access plus high-download checkpoint provisioning. |
| VQ / L13 | Accepted D2 | `compact-vq-vae-v1`, sklearn digits at scikit-learn 1.9.0, seed 42; immutable `artifacts/m14/l13-vq.json` records perplexity `13.090496630645841`, dead-code rate `0.0`, finite encode/decode, and artifact hash. |
| Tokenized/world-model path / L14 | Accepted bounded D2 | Compact VQVAE plus synthetic controlled dynamics, not a named external checkpoint; immutable `artifacts/m14/l14-tokenized.json` records 8/8 runner acceptance, 9 focused tests passed, bounded horizon 8, and retained early free-running failure. This does not promote a named GAIA/Genie/world-model claim. |
| ACT / L19 | Blocked, not run | `lerobot/act_aloha_sim_insertion_human@33259aa86eb45fdf85350280044a33d9d50e40c3`; `artifacts/m14/l19-act.json` records the skipped pinned-checkpoint test and missing model-card license/access capture plus Linux CUDA host. |
| Diffusion Policy / L20 | Blocked, not run | `LeTau/diffusion_aloha_insertion@6126e33` with `lerobot/aloha_sim_insertion_human_image@d93d36a`; `artifacts/m14/l20-diffusion-policy.json` records the skipped test and missing upstream access/license capture plus remote CUDA/Linux execution. |
| SmolVLA / L21 | Blocked, not run | `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de` plus `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`; `artifacts/m14/l21-smolvla.json` records the skipped test and missing Linux ~16 GB GPU, model/data access, and license capture. |
| 3DGS / L17 | Explicitly blocked | No trustworthy named checkpoint, revision/hash, access, or license metadata exists under the authority. `artifacts/m14/l17-3dgs.json` retains the not-run receipt; no unnamed checkpoint or local approximation was substituted. |

## Verification

- `uv run pytest tests/test_m14_validation_contract.py -q` — PASS (3 tests), confirming 24 unique authority lanes and valid referenced paths.
- Existing focused evidence commands and results are retained in `artifacts/task_sprint79_m14_real_system_matrix_summary.md`; accepted rows were intentionally not rerun.
- No remote CUDA run was started for blocked rows because their authoritative prerequisites are absent and the remote-CUDA invariant forbids a skipped lane or substitute checkpoint from becoming evidence.
- Line 595 remains unchecked with its external blocker; line 598 and later remain untouched.
