# Task 79.4C — M14 L02 canonical execution

## Outcome

The direct-file invocation from the prior attempt failed before execution with
`ModuleNotFoundError: No module named 'scripts'`. The supported package-aware
command is now `uv run python -m scripts.m14_l02_geometry`; its side-effect-free
check is `uv run python -m scripts.m14_l02_geometry --check`.

Fix commit `9f1bcbe362ae76015a9db752f9b80e3c81022ed7` was pushed without force
and matched `origin/main` exactly before either real run. The plan digest is
`e2bcbbb361f08eefa429ca48dd708ac00c094fe98959c17d7c88b261d73f869f`.

## Runs and determinism

Both real runs completed successfully from git SHA
`9f1bcbe362ae76015a9db752f9b80e3c81022ed7` using the canonical module
command. Run 1 timestamp was `2026-08-27T03:41:30.480919Z`; run 2 timestamp
was `2026-08-27T03:42:42.722813Z`. The artifact payloads were byte-identical,
including deterministic metrics, verdicts, provenance, and self-digest:
`97d26f5fb1d12dc00658ff9cfec12a91b080bb4a1bb3cd96e7bc2ed70f9e5a58`.
The run records were identical except for `timestamp_utc`. Both artifact and
run-record validators returned no errors.

## Data, model, backend, and provenance

- Dataset: `sklearn.datasets.load_digits`, BSD-3-Clause; 1,797 samples with
  `/16.0` normalization, RNG seed 42 permutation, 1,437 train and 360 held
  out. Train index digest: `f9c880cf9066db68aa666972fe306b6c9afcb789a959f6bcd5a76115e068b2c4`;
  held-out index digest:
  `2153b4535ef4d3bfd092731276601deeab61004b817ee2b54daf4fa1f7b84d59`;
  dataset content digest:
  `4bc179b311d025a12b852eec8fb2ddeaa3099a2a97c99bb40d78bf33d2f59d4a`.
- Model: `latent_anything.adapters.conv_vae.ConvVAE`, latent dimension 4,
  eight epochs, random state 42; fit scope train images only. Metrics:
  reconstruction MSE `0.18588410317897797`, posterior KL
  `0.006323094479739666`, latent utilization `0.0045607807114720345`.
- Density: Gaussian mixture with 10 components, `n_init=2`, random state 42;
  fit scope train latents only; state digest
  `f1b5a6674cdae0d5e1e6e29412b16f83e2d4aed2c3094de7e5d1fe5b435bc1ec`.
- Backend: Python 3.13.3, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.0,
  Torch 2.10.0, latent-anything 0.1.0b1, Windows-11-10.0.26200-SP0;
  network offline and no credentials used.
- Source digests: runner
  `30cfd70c3cf474d0de9740fe5e20efc00c7d11b6cc7c541335dc149bbb6d0226`;
  contract
  `530e20e02dd1cb247ceb8100b6f2bcdc7f5de5d2aafa40b7d4e3a35d6225925f`;
  combined implementation
  `675ff3d646b293fcc85a7b1c201727ce95392a491c951c897c7c9f1a7c3749ad`.
- Input mutation controls were identical before and after for all train/held-
  out images and labels.

## Independent record results

| Record | Verdict | Accepted | Exact deterministic metrics |
|---|---|---:|---|
| `manifold_hypothesis` | failed | no | `pair_count=128`; `real_pair_auc=0.4560546875`; `shuffled_label_auc=0.46142578125`; `raw_pixel_auc=0.8685302734375`; `latent_vs_raw_auc_delta=-0.4124755859375`; `finite=true`; `train_only_density_fit=true` |
| `slerp_spherical` | accepted | yes | `endpoint_error=0.0`; `norm_error=2.220446049250313e-16`; `angular_additivity_error=2.220446049250313e-16`; `finite=true`; `no_input_mutation=true` |
| `lerp_euclidean` | accepted | yes | `endpoint_error=0.0`; `coefficient_error=0.0`; `finite=true`; `no_input_mutation=true` |
| `riemannian_density_geodesic` | accepted | yes | `converged=true`; `iterations=2`; `endpoint_error=0.0`; `mean_log_density_delta=0.012367905751419883`; `path_length=0.13895596335260527`; `finite=true`; `train_only_density_fit=true` |
| `slerp_latent_operation` | accepted | yes | `endpoint_error=0.0`; `norm_error=2.220446049250313e-16`; `angular_additivity_error=2.220446049250313e-16`; `finite=true`; `no_input_mutation=true` |
| `trajectory_similarity_dtw` | failed | no | `independent_pair_trials=128`; `ranking_auc=1.0`; `median_self_to_indexwise_ratio=17.015624999997637`; `median_self_to_unrelated_ratio=0.010693904158763163`; `finite=true`; `unequal_lengths=true`; `no_self_mapping=true`; `no_input_mutation=true`; pair-path digest `027a5a891b0b1a8efee88ecdc814e5d37570fdd5401313a4d1b196313ae26652`; positive digest `73bf28914f08d555f65c3d4e29b531daaf8ad5855b22e359eee64e609ad702ea`; negative digest `a320b983de5ac00dab082251b7e4873a27ae68f7418474f38f0b4c0ce5a92b3e`; derangement digest `b96e25cbb68ab8dd9afc4a3f310c7a596785cfd4aae40b71e101b55139414600` |

The artifact is partial (`D2`) with accepted records
`slerp_spherical`, `lerp_euclidean`, `riemannian_density_geodesic`, and
`slerp_latent_operation`. The evidence ledger promotes only the four mapped
gaps above; the manifold and trajectory-similarity gaps remain unchanged and
failed. The trajectory results are model-induced latent
sequences from held-out real digits, not recorded physical trajectories. No
Fréchet metric or claim is made because no stable project API exists.

The execution result was initially result-only and uncommitted; this summary,
artifact, and run record are now the source evidence for the partial ledger
promotion. L03 was not started.
