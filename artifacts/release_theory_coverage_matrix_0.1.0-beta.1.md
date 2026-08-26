# Theory Coverage Matrix: 0.1.0-beta.1

**Sprint:** 26
**Task:** 9
**Status:** Complete

> **Historical/superseded snapshot (2026-07-10 beta):** This matrix records the
> beta-era theory boundary and is not a current implementation inventory.
> Current evidence status is governed by [M14](../docs/M14_REAL_SYSTEM_VALIDATION.md),
> [the evidence ledger](../docs/EVIDENCE_LEDGER.md), and
> [the machine-readable ledger](../docs/evidence-ledger.json). Retain the
> original dates, metrics, and beta claims for historical traceability.

This matrix maps `docs/THEORY.md` layers to shipped beta code, demo coverage, docs-only coverage, and future work. It is designed to prevent the release notes from overclaiming theory completeness.

| Theory area | Shipped code in beta | Demo/artifact coverage | Release claim |
| --- | --- | --- | --- |
| Tầng 1 - Space and representation | `LatentSpace`, `Trajectory`; Euclidean, unit-norm, and Gaussian-set geometries | PCA, spherical, Gaussian-set demos | Shipped core primitives and basic geometry-aware operations. |
| Tầng 2 - Representation learning | VAE, RandomProjection, HiddenStateAdapter, SAE | VAE, RandomProjection, hidden-state, SAE demos | Shipped first learned/fixed/hidden-state adapter and sparse representation examples. |
| Tầng 3 - Latent geometry | Geometry-keyed `LatentSpace`; distance/interpolate/normalize dispatch | Spherical and Gaussian-set demos | Shipped selected geometry-aware operations; not a full manifold toolkit. |
| Tầng 3B - 3D representation | Gaussian-set geometry and deterministic Gaussian renderer adapter | Gaussian-set and Gaussian renderer demos/artifacts | Shipped simplified 2D Gaussian splat renderer and structured latent stress test; not a production 3DGS backend. |
| Tầng 4 - Computation in latent space | Lerp, SteeringVector, ActivationPatch, ManipulationPipeline | Lerp, steering, activation patch, manipulation, showcase demos | Shipped first manipulation methods and composition paths. |
| Tầng 5 - Probe and intervention | ActivationPatch covers intervention-style manipulation | ActivationPatch and showcase demos | Intervention is partially shipped; probing, linear probes, nonlinear probes, and TCAV are future work. |
| Tầng 6 - Latent space over time | `Trajectory` primitive; trajectory-level Lerp and SteeringVector operations | Lerp/steering trajectory demos | Basic trajectory container and manipulation are shipped; trajectory similarity, segmentation, rollout, and temporal models are future work. |
| Tầng 7 - Planning in latent space | None | Docs only | Future work. No planning/MPC/CEM release claim. |
| Tầng 8 - Predict in latent without decode | None | Docs only | Future work. No latent transition/prediction release claim. |
| Tầng 9 - Discrete latent space | None | Docs only | Future work. No discrete/tokenized latent adapter release claim. |
| Large-scale World Models and VLA | Adapter Protocols can represent hidden-state/no-explicit-latent mode; no VLA adapter | HiddenStateAdapter demo only | Infrastructure-adjacent evidence only; no VLA/world-model integration release claim. |
| Interpretability and analysis tools | PCA, UMAP, SAE; ActivationPatch | Layer A demos and showcase | Shipped first inspection/manipulation tools; clustering, attribution, probes, TCAV, and interactive analysis are future work. |
| Math foundations | Geometry and interpolation choices grounded by docs/ADRs | Spherical/Gaussian demos | Theory-informed implementation, not a complete math library. |
| Practical 3D models | Simplified Gaussian renderer adapter | Gaussian renderer artifact | Simplified deterministic renderer demonstration only. |
| Tầng 14 - Generative manifold / denoising / latent fate | Adapter taxonomy covers explicit learned latent, no-explicit-latent, and deterministic renderer modes | VAE, HiddenStateAdapter, GaussianRendererAdapter demos | Adapter-mode taxonomy is validated; diffusion/JiT-style model integrations are future work. |

## Release Wording Guidance

Safe wording:

- "Core latent-space framework beta."
- "Includes first concrete adapters, geometry-aware primitives, Layer A/B methods, pipelines, registry/config, and runtime helpers."
- "Theory-informed, not theory-complete."

Avoid:

- "Full latent framework."
- "Implements probing/TCAV."
- "Supports planning/rollout/discrete world models."
- "Interactive visualization platform."
- "Stable public API."
