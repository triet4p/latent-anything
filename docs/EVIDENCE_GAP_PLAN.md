# Theory evidence-gap closure plan

This is the Sprint 78.38 execution plan for the non-qualifying theory rows,
updated with the Sprint 79 L02 partial result. The levels in
[`docs/evidence-ledger.json`](evidence-ledger.json) remain authoritative; the
immutable L02 artifact supports exactly four D2 promotions. The row-level
machine-readable map is
[`artifacts/task_78.38_gap_map.json`](../artifacts/task_78.38_gap_map.json).

## Current gate and arithmetic

The read-only validator reports 107 capabilities, **30/63 core (47.6%)** and
**30/65 overall (46.2%)**. The core denominator contains 63
implementation-applicable or benchmark-only rows in T01–T09/T03B; the overall
denominator adds two applicable X01 rows. There are exactly **35 current D0/D1
rows** in this plan: **33 core** and **2 non-core**. A qualifying row is D2 or
D3, so at least **30 additional core qualifiers** are required to reach
`ceil(0.95 × 63) = 60`; at least **29 additional overall qualifiers** are
required to reach `ceil(0.90 × 65) = 59`. The core gate is therefore the
binding gate. Headline model, causal explanation, and named integration claims
target D3; ordinary algorithm capabilities target D2 unless their row says
otherwise.

### Sprint 79 L02 partial result

The committed, reproducible artifact
[`l02-geometry.json`](../artifacts/m14/l02-geometry.json) passed its artifact
and run-record validators and supports exactly four independent D2 promotions:
`THY-T03-SLERP-SPHERICAL-LINEAR-INTERPOLATION`,
`THY-T04-LERP-LINEAR-INTERPOLATION`,
`THY-T03-RIEMANNIAN-GEOMETRY-CO-BAN`, and `THY-T04-SLERP`. The artifact
self-digest is
`97d26f5fb1d12dc00658ff9cfec12a91b080bb4a1bb3cd96e7bc2ed70f9e5a58`, and the
two canonical runs were deterministic apart from their UTC run-record
timestamps.

The `THY-T01-MANIFOLD-HYPOTHESIS` record remains D1 because its held-out
ranking AUC was `0.4560546875` (threshold `0.55`) and its latent-vs-raw delta
was `-0.4124755859375` (threshold `-0.05`). The
`THY-T06-TRAJECTORY-SIMILARITY-METRICS` record remains D0 because its
self-to-indexwise ratio was `17.015624999997637` (threshold `0.95`), despite
128 finite trials and ranking AUC `1.0`; its self-to-unrelated ratio was
`0.010693904158763163`. These failures remain linked to the retained artifact
and are not promoted. The lane concerns model-induced latent sequences from
held-out sklearn digits, not recorded physical trajectories, and makes no
Fréchet claim.

The validator command is:

```text
uv run python scripts/validate_evidence_ledger.py
```

It must continue to report the honest current 30/63 and 30/65 result while
this plan is executed. No row may be deleted, relabeled, or promoted merely to
improve the percentages.

## Exhaustive row inventory

Each row appears exactly once below. Detailed prerequisites, commands, tests,
acceptance, artifact, cleanup, blocker, and dependency fields are in the JSON
map; lane-level defaults are specified in the next section.

| ID | Current | Core | Headline | Target | M14 lane | Capability / insufficiency |
|---|---:|:---:|:---:|:---:|:---:|---|
| `THY-T01-METRIC-SPACE-VA-VECTOR-SPACE` | D2 | yes | no | D2 | L01 | Existing ConvVAE/AnalysisPipeline held-out benchmark and immutable artifact verified |
| `THY-T01-MANIFOLD-HYPOTHESIS` | D1 (failed L02 record) | yes | no | D2 | L02 | Held-out ranking failed: latent AUC 0.4561 vs 0.55 and raw-pixel delta -0.4125 vs -0.05 |
| `THY-T02-VAE-HIGGINS-ET-AL-2017` | D0 | yes | no | D2 | L08 | Theory-only beta-VAE row |
| `THY-T02-VQGAN-ESSER-ET-AL-2021` | D0 | yes | no | D2 | L13 | Theory-only VQGAN row; no approved implementation |
| `THY-T03-LINEAR-STRUCTURE-TRONG-LATENT` | D1 | yes | no | D2 | L03 | PCA tests lack quantitative held-out benchmark |
| `THY-T03-DISENTANGLEMENT` | D0 | yes | yes | D2 | L04 | Benchmark-only row has no factor/control artifact |
| `THY-T03-RIEMANNIAN-GEOMETRY-CO-BAN` | D2 | yes | no | D2 | L02 | Accepted bounded density-geodesic record in the L02 artifact |
| `THY-T03-SLERP-SPHERICAL-LINEAR-INTERPOLATION` | D2 | yes | no | D2 | L02 | Accepted held-out unit-norm interpolation record in the L02 artifact |
| `THY-T03-NORMALIZING-FLOWS` | D0 | yes | no | D2 | L05 | Theory-only flow row |
| `THY-T03B-GAUSSIAN-PARAMETERS-LA-LATENT-VARIABLE` | D1 | yes | yes | D3 | L17 | Reference renderer evidence is not real 3DGS evidence |
| `THY-T03B-DYNAMIC-3DGS` | D0 | yes | yes | D3 | L17 | Theory-only dynamic-scene claim |
| `THY-T04-LERP-LINEAR-INTERPOLATION` | D2 | yes | no | D2 | L02 | Accepted held-out LERP record in the L02 artifact |
| `THY-T04-SLERP` | D2 | yes | no | D2 | L02 | Accepted independent T04 SLERP record in the L02 artifact |
| `THY-T04-DENSITY-ESTIMATION-TRONG-LATENT` | D0 | yes | no | D2 | L05 | Theory-only density row |
| `THY-T04-OPTIMAL-TRANSPORT-TRONG-LATENT` | D0 | yes | no | D2 | L05 | Theory-only transport row |
| `THY-T05-LINEAR-PROBING` | D0 | yes | yes | D2 | L03 | Existing probe is not linked to a D2 benchmark |
| `THY-T05-NONLINEAR-PROBING` | D0 | yes | yes | D2 | L03 | MLP implementation lacks real held-out evidence |
| `THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018` | D0 | yes | yes | D3 | L04 | Theory-only TCAV row |
| `THY-T05-ACTIVATION-PATCHING` | D1 | yes | yes | D3 | L04 | D1 hook tests lack real causal benchmark |
| `THY-T05-SPARSE-AUTOENCODER-SAE-ANTHROPIC-2023` | D1 | yes | yes | D3 | L06 | Synthetic SAE lacks real-model quality/stability evidence |
| `THY-T05-DICTIONARY-LEARNING` | D0 | yes | no | D2 | L06 | Theory-only dictionary-learning row |
| `THY-T05-STEERING-VECTORS-ZOU-ET-AL-2023-REPRESENTATION-ENGINEERING` | D1 | yes | yes | D3 | L04 | D1 steering tests lack real causal/selective evidence |
| `THY-T05-LOGIT-LENS-TUNED-LENS` | D0 | yes | yes | D3 | L04 | Benchmark-only row lacks layerwise model artifact |
| `THY-T06-STOCHASTIC-TRANSITION` | D0 | yes | no | D2 | L15 | Later implementation evidence is not mapped to this ID |
| `THY-T06-RSSM-RECURRENT-STATE-SPACE-MODEL-DREAMER` | D0 | yes | yes | D3 | L15 | Compact RSSM is synthetic, not named Dreamer evidence |
| `THY-T06-TRAJECTORY-SIMILARITY-METRICS` | D0 (failed L02 record) | yes | no | D2 | L02 | DTW record retained but failed self-to-indexwise ratio gate; no promotion |
| `THY-T07-MODEL-PREDICTIVE-CONTROL-MPC` | D0 | yes | no | D2 | L16 | CEM/MPPI do not automatically prove generic MPC row |
| `THY-T07-POLICY-GRADIENT-TREN-IMAGINED-TRAJECTORY-DREAMER` | D0 | yes | yes | D3 | L16 | No actor-critic/policy-gradient lane |
| `THY-T07-VALUE-EQUIVALENCE-MUZERO` | D0 | yes | yes | D3 | L16 | No value-equivalence representation/training lane |
| `THY-T07-MCTS-TRONG-LATENT` | D0 | yes | no | D2 | L16 | No MCTS implementation/benchmark |
| `THY-T08-REPRESENTATION-COLLAPSE` | D0 | yes | no | D2 | L12 | No standalone collapse-control artifact |
| `THY-T08-I-JEPA-ASSRAN-ET-AL-2023` | D0 | yes | yes | D3 | L12 | Compact JEPA is not the named I-JEPA checkpoint |
| `THY-T08-V-JEPA-BARDES-ET-AL-2024` | D0 | yes | yes | D3 | L12 | No video JEPA implementation/model lane |
| `THY-T09-EMA-CODEBOOK-UPDATE` | D0 | yes | no | D2 | L13 | No dedicated EMA update evidence |
| `THY-T09-RESIDUAL-VQ-SOUNDSTREAM-ENCODEC` | D0 | yes | no | D2 | L13 | No residual-VQ implementation/evidence |
| `THY-T09-FINITE-SCALAR-QUANTIZATION-FSQ` | D0 | yes | no | D2 | L13 | No FSQ implementation/evidence |
| `THY-T09-GAIA-1-WAYVE-2023` | D0 | yes | yes | D3 | L14 | Named model survey only |
| `THY-T09-GENIE-GOOGLE-2024` | D0 | yes | yes | D3 | L14 | Named model survey only |
| `THY-X01-OPENVLA` | D0 | no | yes | D3 | L19 | Named policy survey only; non-core X01 |
| `THY-X01-LEWM-LEWORLDMODEL-2026` | D1 | no | yes | D3 | L12 | Compact JEPA is not real LeWM evidence |

## Execution contract and ordering

Execute in dependency order. First close bounded local D2 rows in L01/L02,
L03–L06, L08, L12–L16, then execute named real-model D3 lanes. A row may be
promoted only when its source, focused tests, benchmark, config, and (for D3)
artifact all exist and the validator passes.

Every lane artifact must predeclare the integer seed, exact sample and held-out
split, metric/control definitions, acceptance thresholds, backend/model/dataset
revision, license/access, environment, network policy, resource peak, cleanup,
and SHA-256. A failed run is retained and linked; metrics or evidence levels
must not be edited to hide failure. Use local isolated temp directories for
CPU lanes. Use the remote-cuda-test workflow for CUDA lanes only, with a
disposable clone and isolated cache. No model/network/CUDA work is performed
by 78.38 itself.

### Lane batches

- **L01/L02 core primitives and geometry:** use the existing sklearn-digits or
  recorded-trajectory substitutes, deterministic held-out splits, finite
  shape/dtype/no-mutation checks, endpoint/metric/DTW/geodesic controls, and
  `artifacts/m14/l01-core.json` / `l02-geometry.json`.
- **L03–L06 analysis and explanations:** pin GPT-2
  `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8` and its
  license/access; use
  train/holdout labels, shuffled-label/raw-input/capacity controls, seed
  confidence intervals, selectivity and intervention metrics; write the L03,
  L04, L05, and L06 artifacts. A real-model requirement is never satisfied by
  a compact fixture.
- **L08/L13–L16 bounded model/planning lanes:** preserve existing compact D2
  negative results, held-out splits, codebook/non-collapse controls, sequence
  masks, rollout drift, action bounds, fixed-zero/random-shooting controls,
  and deterministic digests. New VQGAN, EMA/residual-VQ/FSQ, MCTS, policy
  gradient, MuZero, GAIA-1, or Genie rows need owner-approved implementations
  and separate artifacts; existing nearby rows cannot be silently reused.
- **L12 temporal/JEPA:** use the compact collapse-control D2 lane first. D3
  requires the exact named checkpoint and license/access for I-JEPA or LeWM;
  V-JEPA is a separate implementation/data lane. Preserve open-loop drift and
  other negative results.
- **L17 3DGS:** remains blocked until a named checkpoint, revision,
  license/access, CUDA host, and disposable remote lane exist. The existing
  deterministic/reference renderer must not be promoted to D3.
- **L19/OpenVLA:** remains a separate named-policy D3 lane; do not count it as
  core. ACT/Diffusion/SmolVLA lanes do not prove OpenVLA.

## Explicit blockers and waivers

The 3DGS blocker is concrete: M14 L17 has no named checkpoint and therefore no
real multi-view PSNR/SSIM or intervention artifact. The SmolVLA causal claim
also remains **D2 pending a corrected pinned CUDA rerun**: the historical D3
artifact is unverified and must not be counted. Execute the corrected L21 lane
with the pinned model/dataset revisions, simulator controls, seed, thresholds,
and signed artifact before promotion.

External GitHub Actions access, missing model/data licenses, unavailable
checkpoints, CUDA/VRAM, or missing dataset access are blockers—not reasons to
lower thresholds. An owner waiver must state scope, rationale, expiry, and
whether it affects core or headline coverage; no waiver can hide an applicable
core gap or authorize a stable-release claim. Sprint 79 owns execution and
artifact signing; Sprint 80 remains stop-before-release until the validator is
at least 95% core and 90% overall and all headline D3 claims are satisfied.
