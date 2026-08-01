# Task Summary: Sprint 46 Task 07

**Sprint:** Sprint 46
**Task:** Regression thresholds and marked full-model evaluation tests

Added offline regression-threshold tests (reconstruction below variance,
decoder alignment > 0.9, dead-feature detection, sparse L0, cross-seed
stability, ranking/steering agreement, atlas round-trip) and a marked
`test_sae_evaluation_network.py` that runs the full pipeline on pinned GPT-2
layer-6 activations.

**Testing:** Offline suite green; network test gated by `LATENT_ANYTHING_RUN_NETWORK=1`.
