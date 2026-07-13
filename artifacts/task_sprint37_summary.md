# Sprint 37 — Conditional Diffusion Integration

## Summary

Added a concrete, revision-pinned conditional diffusion integration that records
scheduler latent states and selected denoiser activations without forcing
generative execution into the frozen `ModelAdapter.encode()` contract.

## Files Created

| File | Purpose |
|---|---|
| `src/latent_anything/integrations/diffusers_conditional.py` | Main integration class, typed request/result dataclasses, `LatentSpace` descriptors |
| `tests/test_diffusers_conditional.py` | 21 offline tests (dataclass validation, constructor, LatentValue wrappers, FakeBackend generation) |
| `tests/test_diffusers_conditional_network.py` | 3 network-gated tests (output shape, denoiser capture, seed determinism) |
| `scripts/diffusers_conditional_timestep_trajectory.py` | Artifact script producing trajectory visualisation and summary |

## Modified Files

| File | Change |
|---|---|
| `pyproject.toml` | Added `diffusers-full` optional extra, pyright include entries for new files |

## Key Design Decisions

1. **Not a ModelAdapter** — generation lifecycle differs fundamentally from
   `encode()/decode()/latent_space`. ADR recorded in `decisions.md`
   ([2026-07-13]).

2. **No generative protocol** — first integration proves the shape; shared
   extraction waits for ≥3 differing integrations (Rule of Three).

3. **Scheduler states via callback_on_step_end** — backend-native callback
   avoids hook interference. Captured as non-writable NumPy arrays.

4. **Denoiser activations via ActivationCaptureSession** — hooks reserved for
   internal UNet activations. UNet `mid_block` captured as default location.

5. **Three separate LatentSpace descriptors** — VAE bottleneck (dim=4),
   scheduler states (dim=4, different role), denoiser activation (dim=1280).

## Test Results

```
21 passed in 8.43s (offline tests, no model download)
```

## Quality Gate

- ruff: All checks passed
- ruff format: All files formatted
- pyright strict: 0 errors on all new files
- Offline tests: 21/21 passed

## References

- Sprint plan: `docs/sprint-plans/sprint-37.md`
- ADR: `.agents/memory/decisions.md` [2026-07-13]
- Artifact: `artifacts/diffusers_conditional_timestep_trajectory.png`
