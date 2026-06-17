# Architecture Decision Records (ADR)

A chronological log of *why* key choices were made in this project.

**Purpose:** Prevents re-litigating settled decisions. Before refactoring, replacing a library, or changing an architectural pattern, the agent must check this file first. If a decision already has a recorded rationale, do not reverse it without explicit user instruction.

**Format for each entry:**

```
## [YYYY-MM-DD] <Short title of the decision>

**Decision:** What was decided.
**Alternatives considered:** What else was on the table.
**Reason:** Why this option was chosen over the alternatives.
**Consequences:** What this decision constrains or enables going forward.
```

---

<!-- Add new entries below, newest at the bottom -->

## [2026-06-16] Key `LatentSpace` on geometry/manifold structure, not container shape

**Decision:** `LatentSpace` is identified by its *geometry hint* (the manifold/metric structure of the representation), not by its tensor container shape. The geometry enum must treat an **unordered, permutation-invariant set** (3D Gaussian set, point cloud) as a first-class geometry alongside Euclidean vector, sequence/grid, discrete code, and manifold/Riemannian. Status: theory-supported position to validate in Giai đoạn 1, not a frozen interface.

**Alternatives considered:** (a) Key `LatentSpace` on shape only (flat vector / sequence / grid), treating Gaussian sets as a special-cased afterthought; (b) anchor the abstraction on "the learned VAE bottleneck" as the canonical latent.

**Reason:** The theory roadmap (tier 1 manifold hypothesis & intrinsic dimension, tier 3 geometry, tier 14 JiT "two meanings of latent") shows the invariant across models is the *low-dimensional manifold*, not any container or any specific VAE. A shape-keyed or VAE-anchored abstraction misclassifies (i) no-explicit-latent models like JiT/LLM hidden states and (iii) unordered Gaussian sets, exactly the cases tier 14 proves are not edge cases but core. Keying on geometry is what makes the same handle span all of them.

**Consequences:** Enables a single `LatentSpace` to represent flat vectors, structured Gaussian sets, and hidden-state activations without redesign — the key flexibility stress-test for the primitive (ARCHITECTURE §4, LeWM adapter). Forces the geometry hint to carry the *metric*, not just dimensionality, which downstream distance/interpolation operations depend on (see geometry-aware dispatch ADR below). Reversing this would re-introduce special-casing for every non-flat model and break the plugin-uniformity goal.

## [2026-06-16] `ModelAdapter` must classify three latent-exposure modes

**Decision:** The `ModelAdapter` interface (`encode`/`decode`/`latent_space`) must support three general classes of how a model exposes latent, not assume a single canonical one: (i) **explicit learned latent** — VAE/VQGAN, `decode` is a learned decoder; (ii) **no-explicit-latent** — JiT/ViT/LLM, the latent *is* the hidden-state activations, there is no externalized bottleneck; (iii) **explicit non-latent structured representation** — 3DGS, where `decode` is a deterministic non-learned renderer (rasterizer). Status: theory-supported position to validate in Giai đoạn 1.

**Alternatives considered:** (a) Assume every model has an explicit learned encoder/decoder pair (the VAE shape) and bolt on exceptions; (b) treat the 3DGS/renderer-decode case as a one-off special note rather than a general class.

**Reason:** Tier 14 (JiT, two meanings of latent) generalizes what ARCHITECTURE §4 already flagged only for LeWM ("decode is a deterministic rasterizer") into one of three recurring classes. A VAE-shaped-only interface would force `ModelAdapter` to report "no latent to load" for JiT/LLM (wrong — the activations are the latent) and would special-case the deterministic-renderer decode instead of designing for it.

**Reason continued — why these three and not more:** they are the distinct combinations actually observed across the project's stress-test adapters (VAE, VLA, LeWM/3DGS, diffusion, LLM); more classes would be speculative until a real adapter demands them.

**Consequences:** `decode` cannot be assumed to be a learned, invertible map — it may be a deterministic renderer or absent. `latent_space` metadata must describe which mode applies so Layer A/B methods can branch. Makes the LeWM Gaussian-set adapter a planned class member, not a hack. Reversing this would re-collapse the interface to the VAE assumption and break JiT-like and 3DGS-like adapters.

## [2026-06-16] Distance/interpolation operations dispatch on geometry, not default to Euclidean

**Decision:** `Trajectory` operations that depend on a metric (interpolate, compare, distance) dispatch on `LatentSpace.geometry` rather than hardcoding Euclidean lerp: slerp for unit-norm/spherical, Mahalanobis for anisotropic, log/exp-map geodesics for manifold-constrained, and SO(3)/SE(3) operations for pose-valued latent. Euclidean lerp is one case, not the default. Status: theory-supported position to validate when the third Layer-B method lands.

**Alternatives considered:** (a) Default every interpolation to Euclidean lerp and let callers opt into slerp manually; (b) expose only lerp and slerp as separate named methods with no geometry coupling.

**Reason:** Tiers 3 (isotropy/anisotropy, Riemannian, slerp), 4 (latent computation), and 12 (Lie groups SO(3)/SE(3)) establish that Euclidean interpolation is *wrong* for the common cases — unit-norm latents need geodesics on the sphere, anisotropic latents need Mahalanobis, pose-valued latents live on a curved group where element-wise averaging produces invalid rotations. Since the geometry is already carried by `LatentSpace` (first ADR above), dispatching on it is the natural and correct coupling.

**Consequences:** Operation correctness depends on the geometry hint being accurate, raising the importance of the first ADR. Requires deciding where dispatch lives (`LatentSpace`, `Trajectory`, or `Method`) — left open until a third method forces it, per the extract-from-working-code principle. Reversing this (Euclidean-default) would silently corrupt interpolation/averaging for spherical, anisotropic, and pose-valued latents.

## [2026-06-16] Develop `src` incrementally with a Rule-of-Three generalization gate; treat existing ADRs as hypotheses

**Decision:** `src` is built increment-by-increment, each increment adding exactly one concrete instance that runs end-to-end, with interfaces *extracted from working code* rather than designed upfront. Generalization is gated by the **Rule of Three**: 1 instance → stay hardcoded; 2 instances → a tentative *unstable* shared shape, not public; ≥3 instances *of differing philosophy* → freeze the interface and migrate all prior call-sites in the same commit. The three 2026-06-16 ADRs (geometry-keyed `LatentSpace`, 3-mode `ModelAdapter`, geometry-dispatch) are explicitly **hypotheses to validate/refute by code**, not a build-everything-now spec; first increments may implement far narrower. When code contradicts a pending ADR, a new reversing/amending ADR must be appended (never a silent divergence). Full process and the concrete Giai đoạn 1→2 increment plan live in [docs/INCREMENTAL.md](../../docs/INCREMENTAL.md).

**Alternatives considered:** (a) Generalize the interface after *every* increment (abstract eagerly); (b) build the three ADRs' full interfaces upfront as the starting spec, then fill in implementations; (c) keep the incremental intent informal in IDEA/ARCHITECTURE without a written, enforceable rule.

**Reason:** (a) over-abstracts — two near-identical linear methods do not stress-test an interface, so abstracting at instance 2 bakes in the wrong shape; Rule of Three matches the project's own "evidence from ≥2 real use cases" and ARCHITECTURE §2's pre-chosen verify points (PCA→UMAP→**SAE**, lerp→steering→**activation patching**). (b) is exactly the "design from imagination" that IDEA §7 forbids — and the ADRs themselves are dated as positions "to validate in Giai đoạn 1," not frozen interfaces. (c) leaves the most reversal-prone area (interface contract with plugin authors, flagged as a top risk in IDEA §9) without a concrete gate, so a future agent could either freeze too early or drift from an ADR unnoticed.

**Consequences:** Constrains every increment to end with the §4 generalization check and an ADR-reconciliation step (validated vs. a new reversing ADR), and forbids introducing Protocols/ABCs before the 3rd differing instance. Enables the three pending geometry/adapter ADRs to be confirmed or overturned by real code without re-litigation, and keeps the public surface (`ModelAdapter`, `Method`, `Pipeline`) minimal until ≥2 use cases demand promotion. Reversing this (e.g. front-loading the full interface) would re-introduce the imagination-designed interface risk and decouple the ADR log from what the code actually proves.

## [2026-06-17] Root-level `src/latent_anything/` package with root-level pyproject.toml

**Decision:** The `latent_anything` main framework package lives at `src/latent_anything/` (src-layout) with the root-level `pyproject.toml`, separate from the `latent-anything-theory/` sub-project which remains a standalone uv project for theory research and notebooks.

**Alternatives considered:** (a) Place the package as a sub-project inside a dedicated folder (like `latent-anything-theory/`); (b) use a flat-layout with the package at the repo root without a `src/` layer.

**Reason:** The root-level location makes `import latent_anything` natural and discoverable in a standard Python project. The src-layout prevents import confusion between the repo root and installed package. The theory sub-project (`latent-anything-theory/`) has a different purpose (research + notebooks) and a separate Python version requirement (3.13 vs ≥3.11), so keeping them as separate uv projects is cleaner than nesting them in a monorepo.

**Consequences:** All CI, tooling, and docs must reference `src/latent_anything/` as the package root. The theory sub-project remains independently installable and maintainable. Reversing this would require relocating all source files and updating every path in pyproject.toml, CI, and documentation.

## [2026-06-17] Sprint 4 Round 1 — ADR reconciliation: all three pending ADRs remain pending

**Decision:** The three 2026-06-16 ADRs (geometry-keyed `LatentSpace`, 3-mode `ModelAdapter`, geometry-dispatch) all remain **`pending`** after Sprint 4 Round 1. This increment implemented the simplest euclidean flat case only (Method #1, LatentSpace #1) and did not exercise geometry keying, metric dispatch, or the `ModelAdapter` interface at all. No ADR is confirmed or refuted yet.

**Evidence considered:** The Sprint 4 code — `LatentSpace` with hardcoded `geometry="euclidean"`, `Trajectory` as immutable numpy sequence, PCA wrapping sklearn. None of these touch the geometry-variant logic, metric dispatch, or model adapter concerns that the ADRs describe.

**Status update:** `pending` (no change). Expected validation trigger:
- `LatentSpace` geometry-keyed ADR: Sprint 9 (geometry case #2, unit-norm/spherical).
- `ModelAdapter` 3-mode ADR: Sprint 7 (VAE adapter #1).
- Geometry-dispatch ADR: Sprint 9–12 (when third Layer-B method lands).

**Consequences:** No reversal or migration needed. Next re-evaluation at Sprint 5 (UMAP addition).
