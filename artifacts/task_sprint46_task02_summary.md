# Task Summary: Sprint 46 Task 02

**Sprint:** Sprint 46
**Task:** Train/validation separation and checkpoint serialization

Added deterministic train/validation splitting (or explicit `val_data`) so the
SAE is scored on held-out activations, and portable `.npz` checkpoint
serialization (`SAE.state_dict`/`load_state_dict`/`save_checkpoint`/
`load_checkpoint`). Also fixed the SAE L1 penalty to be per-element normalized
so the sparsity term is comparable to reconstruction loss.

**Testing:** `TestCheckpointSerialization` + existing `test_sae.py` passed.
