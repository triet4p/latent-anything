# Sprint 38 — Intervention Experiment Predeclaration

**Date:** 2026-07-13
**Status:** Predeclared (not yet executed)

## 1. Bounded Edit

| Field | Value |
|---|---|
| **Operation** | Additive intervention on DDIM scheduler latents at mid-to-late denoising steps |
| **Intervention** | `latent ← latent + strength × direction` via `callback_on_step_end` |
| **Direction (primary)** | Unit-norm random direction (matched-norm to baseline latent) |
| **Direction (random control)** | Pure Gaussian noise direction (unscaled) |
| **Prompt** | `"a photograph of an astronaut riding a horse"` |
| **Scheduler** | DDIM, 30 inference steps |
| **Seeds** | `{42, 123, 456}` (3 seeds) |
| **Dimensions** | 512 × 512 |
| **Guidance scale** | 7.5 |

One edit, one prompt, one scheduler, three seeds. No denoiser-layer intervention — scheduler latents only.

## 2. Target Metric

| Metric | Definition | Direction |
|---|---|---|
| **Target latent change** | Cosine distance between no-edit and edited latent at the final scheduler step: `1 - cos(latent_baseline, latent_edited)` | Higher = larger intervention effect |
| **CLIP proxy** | Not used in this sprint (no external evaluator dependency) | — |

The target is latent-space change at the concept-bearing spatial locations, aggregated over the final latent.

## 3. Non-Target Preservation Metric

| Metric | Definition | Direction |
|---|---|---|
| **Content preservation (SSIM)** | Structural similarity between no-edit and edited output image (mean over RGB channels) | Higher = better preservation |
| **Pixel MSE** | Mean squared error between no-edit and edited output | Lower = better preservation |

## 4. Quality Proxies

| Proxy | Definition | Threshold |
|---|---|---|
| **Latent norm drift** | Relative change in latent norm: `|norm(edited) - norm(baseline)| / norm(baseline)` | < 0.20 (20 %) |
| **Cosine similarity to baseline** | Cosine similarity between final latents (no-edit vs edited) | Reported per seed |
| **Trajectory deviation** | Mean per-step cosine similarity over all denoising steps | Reported as trajectory |

## 5. Evaluation Set

```
prompts:   ["a photograph of an astronaut riding a horse"]
seeds:     [42, 123, 456]
steps:     30
scheduler: DDIM
controls:  [no-edit, prompt-only, random-direction, matched-norm]
```

Four conditions per seed = 12 runs total.

- **no-edit**: Baseline without any intervention.
- **prompt-only**: Same seed but prompt without "astronaut" → `"a photograph of a horse"`.
- **random-direction**: `direction = N(0,1)`, strength=1.0, steps [15, 25).
- **matched-norm**: `direction = NormalizedRandom × ||latent_at_step_15||`, strength=1.0, steps [15, 25).

## 6. Evidence-Promotion Thresholds

| Condition | Target | Preservation | Quality | Decision |
|---|---|---|---|---|
| Random-direction vs no-edit | Target change > 0.05 | SSIM > 0.7 | Norm drift < 20 % | D2: measurable intervention effect exists |
| Matched-norm vs random | Target change differs with p < 0.10 (3 seeds) | SSIM comparable | Norm drift comparable | D2: direction structure matters beyond norm |
| Prompt-only vs any intervention | Target change > 2× intervention effect | — | — | D1 only (prompt dominates latent edit) |

**Rules:**
- If target change > 0.05 AND SSIM > 0.7 AND norm drift < 20 % across ≥2 seeds → promote to **D2**.
- If all 3 seeds fail any one threshold → retain **D1**, record negative result.
- Prompt-only comparison is informational (qualitative framing, not a direct competitor).
