# VAE Explanation Validity Benchmark

Sprint 36 makes a latent explanation claim pass four independent checks:
held-out centroid-probe factor prediction, shuffled-label control, raw-input baseline, and a
decoded intervention with target/off-target/degradation measurements. The
compact evidence run uses sklearn digits and three ConvVAE seeds. PCA, SAE, and
steering are reported with the same held-out probe definition; only steering is
allowed to make a decoded intervention claim.

The acceptance flag is deliberately conservative: a latent probe must beat the
shuffled-label control by 0.05, not lose to input pixels, increase its target
score, keep mean off-target change at or below 0.1, keep reconstruction and
decoded-intervention MSE at or below 0.1, and keep the 95% seed confidence
interval at or below 0.05. A readable projection
that fails those controls remains D1.

Run `uv run python scripts/vae_explanation_benchmark.py` to overwrite the
small JSON artifact. A run earns D2 only when its acceptance flag passes; a
failed control remains D1 evidence, even though the negative result is retained
for reproducibility.
