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

## [2026-06-17] Sprint 5 Round 2 — ADR reconciliation: all three pending ADRs remain pending

**Decision:** The three 2026-06-16 ADRs (geometry-keyed `LatentSpace`, 3-mode `ModelAdapter`, geometry-dispatch) all remain **`pending`** after Sprint 5 Round 2. This increment added UMAP (Method #2, nonlinear/stochastic/stateful) and sketched the internal `_MethodBase` shape, but touched no geometry-keying, metric dispatch, or model adapter concerns.

**Evidence considered:** The Sprint 5 code — UMAP wrapping `umap-learn`, `_MethodBase` as minimal internal base with only `fit`/`transform`/`fit_transform`, PCA migrated to `_MethodBase`. None of these exercise the geometry-variant logic, metric dispatch, or model adapter concerns that the ADRs describe.

**Status update:** `pending` (no change). Expected validation trigger:
- `LatentSpace` geometry-keyed ADR: Sprint 9 (geometry case #2, unit-norm/spherical).
- `ModelAdapter` 3-mode ADR: Sprint 7 (VAE adapter #1).
- Geometry-dispatch ADR: Sprint 9–12 (when third Layer-B method lands).

**Consequences:** No reversal or migration needed. Next re-evaluation at Sprint 6 (SAE addition).

## [2026-06-17] Sprint 6 Round 3 — ADR reconciliation: all three pending ADRs remain pending

**Decision:** The three 2026-06-16 ADRs (geometry-keyed `LatentSpace`, 3-mode `ModelAdapter`, geometry-dispatch) all remain **`pending`** after Sprint 6 Round 3. This increment added SAE (Method #3, neural/trained, encoder/decoder with L1 sparsity), froze the `Method` Protocol, migrated PCA/UMAP docstrings to note conformance, and promoted `Method` to the public surface — but touched no geometry-keying, metric dispatch, or model adapter concerns.

**Evidence considered:** The Sprint 6 code — `Method` Protocol in `protocols.py`, SAE torch-based implementation, PCA/UMAP docstring updates, `_MethodBase` docstring update from UNSTABLE to frozen-backing. None of these exercise the geometry-variant logic, metric dispatch, or model adapter concerns that the ADRs describe.

**Status update:** `pending` (no change). Expected validation trigger:
- `LatentSpace` geometry-keyed ADR: Sprint 9 (geometry case #2, unit-norm/spherical).
- `ModelAdapter` 3-mode ADR: Sprint 7 (VAE adapter #1).
- Geometry-dispatch ADR: Sprint 9–12 (when third Layer-B method lands).

**Consequences:** No reversal or migration needed. Next re-evaluation at Sprint 7 (VAE adapter).

## [2026-06-17] Sprint 7 Round 4 — ADR reconciliation: VAE confirms mode (i) of 3-mode `ModelAdapter` ADR, ADR remains pending

**Decision:** The VAE adapter (ModelAdapter #1, explicit learned latent, mode i) confirms that the 3-mode `ModelAdapter` ADR's mode (i) — explicit learned latent with `encode`/`decode`/`latent_space` — is real and useful. The ADR stays **`pending`** (not `validated`) because modes (ii) no-explicit-latent (JiT/LLM hidden states) and (iii) deterministic-renderer (3DGS/LeWM) remain untested. The other two ADRs (geometry-keyed `LatentSpace`, geometry-dispatch) were not exercised by this increment.

**Evidence considered:** The Sprint 7 code — VAE adapter with `encode` (returns latent mean), `decode` (sigmoid reconstruction), `latent_space` property returning `LatentSpace(dim=..., source_model="vae")`. The VAE cleanly exercises mode (i): a learned encoder maps data to a bottleneck, a learned decoder maps back, and the latent space is an explicit lower-dimensional representation. This is the simplest case among the three modes.

**Status update:**
- `ModelAdapter` 3-mode ADR: `pending` → `pending` (mode i confirmed, 2 of 3 modes untested). Expected full validation: Sprint 8 (VLA — may exercise mode ii) + future sprint (3DGS — mode iii).
- `LatentSpace` geometry-keyed ADR: `pending` (no change). Expected validation: Sprint 9 (geometry case #2, unit-norm/spherical).
- Geometry-dispatch ADR: `pending` (no change). Expected validation: Sprint 9–12 (when third Layer-B method lands).

**Consequences:** VAE reinforces that `encode`/`decode`/`latent_space` is a viable shape for the `ModelAdapter` interface, but the ADR is not yet confirmed general. No Protocol/ABC should be created until at least 2 modes are proven (per Rule of Three). Next re-evaluation at Sprint 8 (VLA adapter).

## [2026-06-17] Sprint 8 Round 5 — ADR reconciliation: all three pending ADRs remain pending; `_ModelAdapterBase` shape sketched UNSTABLE

**Decision:** The three 2026-06-16 ADRs (geometry-keyed `LatentSpace`, 3-mode `ModelAdapter`, geometry-dispatch) all remain **`pending`** after Sprint 8 Round 5. This increment added RandomProjection (ModelAdapter #2, stateless/fixed-weight) and sketched the internal `_ModelAdapterBase` shape (marked UNSTABLE), but touched no geometry-keying, metric dispatch, or new model-adapter modes.

**Rule of Three §4a outcome:** Instance #2 → sketch shared shape (`_ModelAdapterBase` with `encode`, `decode`, `latent_space`), marked UNSTABLE, NOT public. `fit` deliberately excluded (VAE-specific). Not promoted to public surface.

**Evidence considered:** The Sprint 8 code — RandomProjection with pure numpy fixed-weight projection matrix, no `fit` method, `encode` (data @ W.T), `decode` (latent @ W), `latent_space` returning `LatentSpace(dim=latent_dim, source_model="random_projection")`. `_ModelAdapterBase` as internal ABC with only `encode`/`decode`/`latent_space`. None of these exercise geometry-variant logic, metric dispatch, or new model-adapter modes (ii) or (iii).

**Status update:**
- `ModelAdapter` 3-mode ADR: `pending` → `pending` (mode i confirmed by VAE, modes ii and iii untested). The RandomProjection confirms the stateless/pretrained pattern exists but is still mode i-like (encode projects to explicit latent, decode projects back). Expected full validation: future sprint with mode ii (JiT/LLM hidden states) + mode iii (3DGS/LeWM deterministic renderer).
- `LatentSpace` geometry-keyed ADR: `pending` (no change). Expected validation: Sprint 9 (geometry case #2, unit-norm/spherical).
- Geometry-dispatch ADR: `pending` (no change). Expected validation: Sprint 9–12 (when third Layer-B method lands).

**Consequences:** RandomProjection proves that a `ModelAdapter` can work with pure numpy (no torch dependency) and without a `fit` step, reinforcing the shared `encode`/`decode`/`latent_space` surface while proving `fit` is genuinely VAE-specific. The internal `_ModelAdapterBase` shape is deliberately unstable and not public. The 3-mode ADR stays pending because only mode (i) is confirmed. Next re-evaluation at Sprint 9 (geometry case #2).

## [2026-06-17] Sprint 9 Round 6 — ADR reconciliation: LatentSpace geometry-keyed ADR → **validated**; geometry-dispatch ADR → **validated**

**Decision:** Two of the three 2026-06-16 ADRs move from `pending` → `validated` after Sprint 9. This increment added `unit_norm` (spherical) as geometry case #2 to `LatentSpace`, proving both geometry-keying and geometry-dispatch with real code across two distinct metrics.

**Evidence considered:**

1. **`LatentSpace` geometry-keyed ADR**: The `geometry` parameter is now instance-level (moved from class-level), validated against `{"euclidean", "unit_norm"}` at construction. Two `LatentSpace` instances with different geometries (`LatentSpace(dim=3)` vs `LatentSpace(dim=3, geometry="unit_norm")`) coexist, each carrying its own geometry key. The key drives `validate_point`, `distance`, `interpolate`, and `normalize` behavior. Code: `src/latent_anything/latent_space.py`.

2. **Geometry-dispatch ADR**: `distance()` dispatches on `self.geometry` — Euclidean (`||a-b||`) vs angular (`arccos(clip(a·b, -1, 1))`). `interpolate()` dispatches — lerp for Euclidean vs slerp (proper geodesic on sphere) for unit_norm, with edge case handling (`sin(ω) ≈ 0`). `validate_point()` dispatches — shape-only for Euclidean, shape + norm ≈ 1 for unit_norm. 35 new tests prove metric correctness across both geometries.

3. **Rule of Three §4a**: Instance #2 confirms dispatch stays inline (`if/elif`), no `GeometryProtocol` extracted. Instance #3 (sequence/grid, Gaussian set, or discrete code) would trigger extraction.

**Status update:**
- `LatentSpace` geometry-keyed ADR: `pending` → **`validated`** (2 geometry cases prove enum structure works).
- Geometry-dispatch ADR: `pending` → **`validated`** (distance/interpolate/validate dispatch on geometry proven with 2 metrics).
- `ModelAdapter` 3-mode ADR: `pending` (no change — mode i confirmed by VAE, modes ii and iii untested).

**Consequences:** Both geometry-related ADRs are now considered settled design principles. Future geometry additions must follow the validated dispatch pattern. The `ModelAdapter` 3-mode ADR remains the last pending ADR, awaiting mode ii (JiT/LLM hidden states) or mode iii (3DGS/LeWM deterministic renderer). This concludes Giai đoạn 1 of the project plan — core primitives with two geometry cases and three ModelAdapter instances.

## [2026-06-18] Sprint 10 Round 7 — ADR reconciliation: both validated ADRs exercised; no change to ADR statuses

**Decision:** Both validated ADRs (geometry-keyed `LatentSpace`, geometry-dispatch) are exercised by the Lerp B-Method — Lerp delegates to `LatentSpace.interpolate()` for geometry-aware dispatch, proving the geometry-dispatch pattern is consumed by Layer B methods. No ADR status changes. The `ModelAdapter` 3-mode ADR remains `pending` (no change — not touched by this Layer B increment).

**Evidence considered:**

1. **`LatentSpace` geometry-keyed ADR (validated)**: Lerp accepts an optional `LatentSpace` at construction via `Lerp(space=space)`. When provided, `__call__` delegates to `space.interpolate(a, b, t)`, consuming the geometry key. When `None`, defaults to Euclidean `(1-t)*a + t*b`. This is the first time a Layer B method exercises the geometry-keyed `LatentSpace` from a consumer perspective — the architecture works as designed.

2. **Geometry-dispatch ADR (validated)**: `Lerp.__call__` with `Lerp(space=LatentSpace(dim=3, geometry="unit_norm"))` produces correct slerp (stays on sphere, midpoint at 45°), while `Lerp()` (no space) produces Euclidean lerp (departs sphere). This confirms the dispatch pattern is usable from Layer B methods without modification.

3. **Rule of Three §4a**: B-Method #1 (Lerp, stateless, pure transform) → stay hardcoded. No `Method` Protocol modification. The existing `Method` Protocol has `fit`/`transform` — stateless methods don't fit this yet. Interface expansion happens when B-Method #3 (activation patching, Sprint 12) reveals the full stateless+stateful spectrum.

**Status update:**
- `LatentSpace` geometry-keyed ADR: `validated` (no change — exercised by Lerp consumer).
- Geometry-dispatch ADR: `validated` (no change — exercised by Lerp consumer).
- `ModelAdapter` 3-mode ADR: `pending` (no change — not touched by this Layer B increment).

**Consequences:** This sprint confirms that the two validated ADRs are consumed correctly by Layer B methods, validating the architecture from the consumer side. The `ModelAdapter` 3-mode ADR remains the last pending ADR. Giai đoạn 2 is now underway — Layer B foundation with Lerp as B-Method #1. Next re-evaluation at Sprint 11 (steering vector, B-Method #2).

## [2026-06-18] Sprint 11 Round 8 — ADR reconciliation: both validated ADRs exercised by SteeringVector; no change to ADR statuses

**Decision:** Both validated ADRs (geometry-keyed `LatentSpace`, geometry-dispatch) are exercised by the `SteeringVector` B-Method #2 — SteeringVector optionally accepts a `LatentSpace` and uses `space.normalize()` for geometry-aware post-steer normalization (e.g. project back to sphere). No ADR status changes. The `ModelAdapter` 3-mode ADR remains `pending` (not touched by this Layer B increment).

**Rule of Three §4a outcome:** B-Method #2 (SteeringVector, stateful, `fit(positives, negatives)` from contrast) → sketch internal `_BMethodBase`, mark UNSTABLE. Lerp (stateless, no `fit`) and SteeringVector (stateful, has `fit`) now show two distinct B-Method patterns. `_BMethodBase` captures `__call__` + `space` + `apply_trajectory`. Neither B-Method inherits from it — structural duck-typing only. The existing `Method` Protocol (`fit`/`transform`/`fit_transform`) was designed for stateful Layer A dimensionality-reduction methods and does NOT fit either B-Method.

**Evidence considered:**

1. **`LatentSpace` geometry-keyed ADR (validated)**: SteeringVector accepts an optional `LatentSpace` at construction via `SteeringVector(space=space)`. When provided and `space.geometry == "unit_norm"`, `__call__` applies `space.normalize()` after steering to keep points on the sphere. When `space=None` (default), Euclidean steering with no normalization.

2. **Geometry-dispatch ADR (validated)**: `SteeringVector(space=LatentSpace(dim=3, geometry="unit_norm"))` triggers `space.normalize()` after each `__call__`, projecting steered points back onto the unit sphere. Tests confirm steered points have `||point|| ≈ 1` at all strengths, while Euclidean steering (no space or Euclidean space) departs from the sphere.

3. **Rule of Three §4a**: Instance #2 → `_BMethodBase` sketched at `methods/_b_base.py`, marked UNSTABLE. Covers `__call__` + `space` + `apply_trajectory`. B-Method freeze at instance #3 (activation patching, Sprint 12) when stateless + stateful + hook-based patterns are all proven.

**Status update:**
- `LatentSpace` geometry-keyed ADR: `validated` (no change — exercised by SteeringVector consumer).
- Geometry-dispatch ADR: `validated` (no change — exercised by SteeringVector consumer).
- `ModelAdapter` 3-mode ADR: `pending` (no change — not touched by this Layer B increment).

**Consequences:** This sprint confirms that stateful B-Methods (with `fit`) can consume the two validated ADRs correctly. The `_BMethodBase` internal sketch follows the established pattern from `_MethodBase` (Sprint 5) and `_ModelAdapterBase` (Sprint 8). The `ModelAdapter` 3-mode ADR remains the last pending ADR. Next re-evaluation at Sprint 12 (activation patching, B-Method #3 — freeze trigger).

## [2026-06-18] Sprint 12 Round 9 — ADR reconciliation: BMethod Protocol frozen; ModelAdapter 3-mode ADR gains consumer-side evidence, stays pending

**Decision:** Both validated ADRs (geometry-keyed `LatentSpace`, geometry-dispatch) continue to be exercised by `ActivationPatch` — it accesses `adapter.latent_space` and uses `.dim` for validation, coupling through the adapter layer. No ADR status changes for the two validated ones. The `ModelAdapter` 3-mode ADR gains **consumer-side evidence** but stays `pending`. The `BMethod` Protocol is now frozen per Rule of Three (§4a).

**Rule of Three §4a outcome:** B-Method #3 (ActivationPatch, model-mediated, data→data via adapter → encode → patch → decode) → **freeze `BMethod` Protocol** in `methods/b_protocols.py`. Remove `_b_base.py` (the UNSTABLE sketch). Migrate Lerp (add `is_fitted` + generic `apply_trajectory(**kwargs)`) and SteeringVector (docstring update). Three distinct B-Method patterns now proven: stateless latent→latent (Lerp), stateful latent→latent (SteeringVector), model-mediated data→data (ActivationPatch).

**Evidence considered:**

1. **`LatentSpace` geometry-keyed ADR (validated, no change)**: `ActivationPatch.space` returns `adapter.latent_space`, which is always a `LatentSpace` instance. Unlike Lerp/SteeringVector (which may return `None`), ActivationPatch always has a space because the adapter is required. The geometry key is consumed through the adapter's `LatentSpace`, not directly — this is a new consumption pattern (adapter-mediated).

2. **Geometry-dispatch ADR (validated, no change)**: Not directly exercised by ActivationPatch — ActivationPatch operates in data space, not latent space. Geometry dispatch is relevant when the `BMethod` is used for latent-space operations (Lerp/SteeringVector), not for model-mediated data→data transformations.

3. **`ModelAdapter` 3-mode ADR (pending → pending)**: ActivationPatch is the first B-Method to consume a `ModelAdapter` directly. It works with VAE (mode i, explicit learned latent, `encode`/`decode` both learned) and RandomProjection (mode i-like, stateless projection). This proves that `ModelAdapter` is consumable from Layer B, but modes (ii) no-explicit-latent (JiT/LLM hidden states) and (iii) deterministic-renderer (3DGS/LeWM) remain untested. The ADR now has **consumer-side evidence** in addition to the producer-side evidence from VAE and RandomProjection, but full validation still requires modes (ii) and (iii).

4. **`BMethod` vs `Method` separation validated**: The frozen `BMethod` Protocol (`space`, `is_fitted`, `apply_trajectory`) is structurally different from the `Method` Protocol (`fit`, `transform`). This confirms ARCHITECTURE.md's prediction that A/B/C methods have different shapes — the aspirational "interface chung cho mọi A/B/C method" is **disproven by code**, replaced by separate fit-for-purpose Protocols.

**Status update:**
- `LatentSpace` geometry-keyed ADR: `validated` (no change — exercised via adapter coupling).
- Geometry-dispatch ADR: `validated` (no change — not directly exercised this sprint).
- `ModelAdapter` 3-mode ADR: `pending` (no change — consumer-side evidence added, but modes ii and iii untested).

**Consequences:** The `BMethod` Protocol freeze is the third Rule of Three freeze in the project (after `Method` at Sprint 6, and geometry dispatch patterns at Sprint 9). The separation of A (`Method`) and B (`BMethod`) Protocols is now an established architectural principle — future Layer C methods will likely produce a `CMethod` Protocol, not be forced into either existing one. The `ModelAdapter` 3-mode ADR remains the last pending ADR with consumer-side evidence but incomplete mode coverage. Next re-evaluation at Sprint 13 (showcase end-to-end).

## [2026-06-19] Sprint 13 Round 10 — ADR reconciliation: composition-only round; no ADR status changes

**Decision:** No ADR status changes for Sprint 13. This is a **composition-only round** that adds no new adapter, method, or geometry instances. All three ADR statuses remain unchanged. The `ModelAdapter` 3-mode ADR gains additional consumer-side evidence (the showcase uses `ActivationPatch` with both VAE and RandomProjection adapters in a composite narrative), but modes (ii) no-explicit-latent and (iii) deterministic-renderer remain untested.

**Rule of Three §4a outcome:** Not applicable — this is a composition round, not an instance-adding round. No new abstraction extracted. No interface frozen. The showcase config is kept as a local artifact in `scripts/showcase_config.py`, deliberately NOT promoted to `src/` or treated as a framework-wide config system.

**Evidence considered:**

1. **`LatentSpace` geometry-keyed ADR (validated, no change)**: VAE's `LatentSpace` (euclidean, dim=3) is consumed by PCA Layer A (projection), ActivationPatch (space property), and Lerp (trajectory blending). All three consumed instances work without modification.

2. **Geometry-dispatch ADR (validated, no change)**: Not directly exercised by this showcase (no spherical/second-geometry needed). The Euclidean-only story is consistent with the VAE adapter's flat latent space.

3. **`ModelAdapter` 3-mode ADR (pending → pending)**: The showcase confirms that `ActivationPatch` (B-Method #3) composes correctly with `VAE` (mode i) for the full encode → patch → decode cycle in a multi-step narrative. This adds consumer-side evidence but does not test modes (ii) or (iii).

**Status update:**
- `LatentSpace` geometry-keyed ADR: `validated` (no change — exercised by showcase).
- Geometry-dispatch ADR: `validated` (no change — not directly exercised this sprint).
- `ModelAdapter` 3-mode ADR: `pending` (no change — consumer-side evidence added, modes ii and iii still untested).

**Consequences:** This sprint proves that the framework's existing primitives compose correctly into an end-to-end story without requiring new abstractions. The `ModelAdapter` 3-mode ADR remains the last pending ADR. Full validation still requires mode ii (JiT/LLM hidden states — e.g. with a real LLM adapter) and mode iii (3DGS/LeWM deterministic renderer). Next re-evaluation at the next sprint that adds a new adapter instance or a new method instance.

## [2026-06-20] Sprint 14 Round 11 — ADR reconciliation: ModelAdapter 3-mode ADR mode (ii) confirmed; ModelAdapter + DecodableAdapter Protocols frozen

**Decision:** The `ModelAdapter` 3-mode ADR remains `pending` after Sprint 14, but mode (ii) is confirmed by running code. This increment added `HiddenStateAdapter` (ModelAdapter #3, mode ii: no-explicit-latent), completing validation of two of the three modes from the 2026-06-16 ADR. The two already-validated ADRs (geometry-keyed `LatentSpace`, geometry-dispatch) remain unchanged.

**Rule of Three §4a outcome:** ModelAdapter #3 (HiddenStateAdapter, no-explicit-latent, encode-only, no decode) → **freeze `ModelAdapter` Protocol** + **`DecodableAdapter` Protocol** in `adapters/protocols.py`. Remove `_base.py` (the UNSTABLE `_ModelAdapterBase` sketch). Three instances with differing philosophies now proven:
- VAE (#1) — explicit learned latent (mode i): conforms to both `ModelAdapter` and `DecodableAdapter`.
- RandomProjection (#2) — fixed explicit projection (mode i-like): conforms to both `ModelAdapter` and `DecodableAdapter`.
- HiddenStateAdapter (#3) — no-explicit-latent (mode ii): conforms to `ModelAdapter` only (no `decode`).

The split into two Protocols (base `ModelAdapter` for `encode` + `latent_space`, extended `DecodableAdapter` for +`decode`) reflects the core evidence: `decode` is NOT universal.

**Evidence considered:**

1. **`ModelAdapter` 3-mode ADR (pending → pending, modes i/ii confirmed)**: Modes (i) and (ii) are now confirmed by running code. Mode (i) was confirmed by VAE (Sprint 7) and RandomProjection (Sprint 8) — both have `encode` + `decode` + `latent_space`. Mode (ii) is confirmed by HiddenStateAdapter (Sprint 14) — `encode` + `latent_space` only, no `decode`, with metadata marking `exposure_mode="hidden_state"`. The ADR's core claim that `decode` cannot be assumed universal is supported, but full validation still requires mode (iii) deterministic renderer.

2. **`LatentSpace` geometry-keyed ADR (validated, no change)**: HiddenStateAdapter's `latent_space` returns a Euclidean `LatentSpace` with metadata marking `exposure_mode="hidden_state"`. The geometry-keyed LatentSpace design handles mode (ii) without modification.

3. **Geometry-dispatch ADR (validated, no change)**: Not directly exercised by this sprint (no spherical/second-geometry needed for the hidden-state case).

4. **Protocol separation validated**: The split into `ModelAdapter` (universal: encode + latent_space) and `DecodableAdapter` (extended: +decode) is the fourth Protocol in the project (after `Method` at Sprint 6, geometry patterns at Sprint 9, and `BMethod` at Sprint 12). The design proves that `ActivationPatch` (which needs `decode`) can be typed against `DecodableAdapter` and will reject a `HiddenStateAdapter` with a clean `TypeError` at construction.

**Status update:**
- `ModelAdapter` 3-mode ADR: `pending` → `pending` (modes i and ii confirmed by code; mode iii pending Sprint 16).
- `LatentSpace` geometry-keyed ADR: `validated` (no change — exercised by HiddenStateAdapter).
- Geometry-dispatch ADR: `validated` (no change — not directly exercised this sprint).

**Consequences:** The `ModelAdapter` 3-mode ADR is not fully validated yet, but its Protocol split (`ModelAdapter` vs `DecodableAdapter`) is established at the Rule-of-Three freeze point. Mode (iii) determination remains provisional until a concrete renderer adapter exists (expected Sprint 16). Next re-evaluation at Sprint 15 (Gaussian set geometry) and Sprint 16 (deterministic renderer adapter for mode iii).

## [2026-06-20] Sprint 15 Round 12 — ADR reconciliation: geometry-keyed and geometry-dispatch ADRs exercised by `gaussian_set` case #3; no status changes

**Decision:** No ADR status changes for Sprint 15. Both validated ADRs (geometry-keyed `LatentSpace`, geometry-dispatch) are exercised by the `gaussian_set` geometry case #3 — the first structured, set-like latent shape. The `ModelAdapter` 3-mode ADR remains `pending` with modes (i) and (ii) confirmed; mode (iii) is not touched by this geometry-only increment.

**Rule of Three §4a outcome:** Geometry #3 (`gaussian_set`) confirms inline `if/elif` dispatch remains acceptable — 3 branches are not yet brittle. No dispatch table extraction was needed. The public `LatentSpace(dim=...)` API is preserved for flat geometries; `n_gaussians` and dimension-parameter fields are additive constructor kwargs.

**Evidence considered:**

1. **`LatentSpace` geometry-keyed ADR (validated, no change)**: The `gaussian_set` geometry case proves that the geometry key can carry structured shape information (`(n_gaussians, param_dim)` instead of `(dim,)`) while preserving flat-geometry ergonomics. The `shape` property now returns geometry-dependent tuples. `n_gaussians` and `param_dim` are exposed as properties. The `gaussian_set_param_layout` metadata entry documents the parameter column layout (position, scale, opacity, color).

2. **Geometry-dispatch ADR (validated, no change)**: `distance()` dispatches to a permutation-aware set distance (sort-by-position lexicographic, then Frobenius norm). `interpolate()` dispatches to a constrained interpolation (log-space for scale, clamp for opacity/color). `normalize()` returns a copy (no geometry constraint beyond what `validate_point` checks). `validate_point()` checks shape `(n_gaussians, param_dim)` plus numeric constraints (scale > 0, opacity/color in [0,1]). All four methods follow the established `if/elif` dispatch pattern.

3. **`ModelAdapter` 3-mode ADR (pending, no change)**: Not touched by this geometry-only sprint. Mode (iii) deterministic-renderer remains pending Sprint 16.

**Status update:**
- `LatentSpace` geometry-keyed ADR: `validated` (no change — exercised by `gaussian_set` case #3).
- Geometry-dispatch ADR: `validated` (no change — exercised by `gaussian_set` distance/interpolate/validate dispatch).
- `ModelAdapter` 3-mode ADR: `pending` (no change — not touched this sprint; modes i and ii confirmed).

**Consequences:** This sprint proves that the geometry-keyed `LatentSpace` design can represent structured, set-like latent shapes without modifying the flat-geometry API. The `if/elif` dispatch pattern survives 3 geometries without abstraction. The sprint prepares the codebase for a deterministic-renderer adapter (Sprint 16) that will consume `gaussian_set` geometry through the `ModelAdapter` interface. Next re-evaluation at Sprint 16 (deterministic renderer adapter for `ModelAdapter` mode iii).

## [2026-06-20] Sprint 16 Round 13 — ADR reconciliation: ModelAdapter mode (iii) confirmed by GaussianRendererAdapter; all three 2026-06-16 ADRs now fully validated with all modes

**Decision:** The `ModelAdapter` 3-mode ADR's mode (iii) — explicit non-latent structured representation with deterministic decode — is now confirmed by running code (`GaussianRendererAdapter`). This closes the last evidence gap for the 2026-06-16 ADR. All three ADRs are now fully validated with all modes confirmed by concrete instances.

**Rule of Three §4a outcome:** Adapter #4 (`GaussianRendererAdapter`) is the first instance in mode (iii). No new Protocol extraction is needed — the existing `ModelAdapter`/`DecodableAdapter` split already handles this case correctly. The `gaussian_set` geometry (geometry #3) survives as inline `if/elif` dispatch — 3 branches remain acceptable.

**Evidence considered:**

1. **`ModelAdapter` 3-mode ADR (pending → validated)**: Mode (iii) is now confirmed by `GaussianRendererAdapter` — a concrete adapter whose `decode` is a deterministic numpy-only 2D Gaussian splat renderer (not a learned neural network). The adapter conforms to both `ModelAdapter` (encode + latent_space) and the shape-generic `DecodableAdapter` (encode + decode + latent_space), with `encode` provided as a heuristic grid-based approximation (the adapter is documented as latent-source-first). It intentionally does not conform to `FlatBatchDecodableAdapter` because its public shapes are image/gaussian-set rather than batch matrices. Three modes are now all confirmed by running code:
   - Mode (i) — explicit learned latent: VAE (Sprint 7), RandomProjection (Sprint 8).
   - Mode (ii) — no-explicit-latent: HiddenStateAdapter (Sprint 14).
   - Mode (iii) — deterministic renderer: GaussianRendererAdapter (Sprint 16).

2. **`LatentSpace` geometry-keyed ADR (validated, no change)**: `GaussianRendererAdapter.latent_space` returns a `gaussian_set` `LatentSpace` with `position_dim=2, scale_dim=2, color_dim=3` (8 columns: pos(2) + scale(2) + opacity(1) + color(3)), and metadata marking `exposure_mode="deterministic_renderer"` plus image dimensions. The geometry key correctly carries structured parameter layout info that the decode method consumes.

3. **Geometry-dispatch ADR (validated, no change)**: Not directly exercised by this sprint — the adapter's `LatentSpace` is `gaussian_set` geometry, but the decode operation is a custom renderer (not a `LatentSpace` distance/interpolate call). The `LatentSpace` geometry dispatch is used only when a caller uses `space.interpolate()` or `space.distance()` on the adapter's latent points.

**Status update:**
- `ModelAdapter` 3-mode ADR: `pending` → **`validated (all 3 modes confirmed)`** — mode (iii) confirmed by GaussianRendererAdapter. This is the final ADR closure from the 2026-06-16 set.
- `LatentSpace` geometry-keyed ADR: `validated` (no change — exercised by GaussianRendererAdapter's gaussian_set latent space).
- Geometry-dispatch ADR: `validated` (no change — not directly exercised this sprint).

**Consequences:** All three 2026-06-16 ADRs are now fully validated with all modes confirmed by running code. The `ModelAdapter` vs `DecodableAdapter` Protocol split is proven correct, and the follow-up `FlatBatchDecodableAdapter` refinement keeps flat-batch Layer B consumers from accidentally accepting structured decoders. The deterministic renderer conforms to `DecodableAdapter` because it has a `decode` method, while `HiddenStateAdapter` (mode ii) correctly does not. The next ADR-relevant sprint will be Sprint 17 (plugin registry) or a future adapter that challenges the existing Protocol shape. No further ADR reconciliation is expected from the 2026-06-16 set.

## [2026-06-21] Sprint 16 review correction — split shape-generic decodability from flat-batch decodability

**Decision:** Keep `DecodableAdapter` as the broad, shape-generic adapter surface (`encode` + `decode` + `latent_space`) and add `FlatBatchDecodableAdapter` for adapters whose public encode/decode contract is specifically flat batch matrices. `ActivationPatch` now requires `FlatBatchDecodableAdapter`.

**Alternatives considered:** (a) Force `GaussianRendererAdapter` to accept and return flat batches; (b) leave `ActivationPatch` typed against the broad `DecodableAdapter`; (c) remove GaussianRenderer from `DecodableAdapter`.

**Reason:** `GaussianRendererAdapter` is a valid mode (iii) `DecodableAdapter`: it decodes a Gaussian-set latent into an image through a deterministic renderer. But `ActivationPatch` computes means over sample batches and broadcasts latent deltas, so accepting a structured image/gaussian-set decoder would fail semantically despite passing runtime Protocol checks. The narrower Protocol records that method-level assumption explicitly without weakening the adapter ADR.

**Consequences:** Structured decoders remain first-class `DecodableAdapter`s, while flat-batch Layer B consumers have an enforceable guard. Future methods must choose the broad or narrow Protocol based on the shapes they actually consume.

## [2026-06-21] Sprint 17 Round 14 — ADR reconciliation: infrastructure-only round; no ADR status changes

**Decision:** No ADR status changes for Sprint 17. This is a **plugin-extraction infrastructure round** that adds no new adapter, method, or geometry instances. All three validated ADRs (geometry-keyed `LatentSpace`, geometry-dispatch, `ModelAdapter` 3-mode) remain unchanged.

**Rule of Three §4a outcome:** Not applicable — this is an infrastructure round, not an instance-adding round. The registry is instance #1 (in-process, no entry points). No new abstraction extracted. No interface frozen.

**Evidence considered:**

1. **`LatentSpace` geometry-keyed ADR (validated, no change)**: Not exercised by this sprint — no new `LatentSpace` instances added.
2. **Geometry-dispatch ADR (validated, no change)**: Not exercised by this sprint — no new geometry or dispatch operations added.
3. **`ModelAdapter` 3-mode ADR (validated, all 3 modes confirmed)**: Not exercised by this sprint — no new adapter instances added.

**Status update:** All three ADRs: `validated` (no change).

**Consequences:** The registry is intentionally in-process with no Python entry points, following the Sprint 17 design constraint. Future rounds (Sprint 18: pydantic config specs, Sprint 19: behavior parity conversion) will exercise the registry from a consumer perspective and may trigger Rule of Three generalization. Next re-evaluation at Sprint 18 (config specs).

## [2026-07-09] Keep Sprint 24 async/profiling runtime surfaces concrete; do not freeze `RuntimeExecutor` yet

**Decision:** Sprint 24 adds async wrappers and profiling hooks directly on existing concrete runtime paths (`AnalysisPipeline`, `ManipulationPipeline`, `BatchExecutor`) and does **not** extract a `RuntimeExecutor` Protocol or shared runtime ABC yet.
**Alternatives considered:** (a) Freeze a new `RuntimeExecutor` Protocol now that batching, cache, and async all exist; (b) introduce a standalone runtime orchestrator object and migrate pipelines to it immediately; (c) keep async support only on one path such as `AnalysisPipeline`.
**Reason:** The current runtime instances still differ more than they agree: `BatchExecutor` batches one array operation, `AnalysisPipeline` composes cache + encode + Layer A fit/transform, and `ManipulationPipeline` has separate latent-only vs adapter-mediated stories with optional decode. Async support in this sprint is thin `asyncio.to_thread` wrapping over working sync code, and profiling is hook-based observability, not a new execution contract. Freezing a shared Protocol here would be "design from imagination" rather than extraction from a third distinct, stress-tested runtime shape.
**Consequences:** Future runtime work should keep composing on the concrete paths until a later sprint produces a genuinely invariant execution surface worth freezing. The public runtime additions are the async methods and profiling data structures, not a new executor abstraction. Re-evaluate only when another runtime story (for example streaming, disk cache orchestration, or a third pipeline execution pattern) forces shared shape by code rather than anticipation.

## [2026-07-09] Keep method-specific arguments outside the frozen `BMethod` invariant

**Decision:** The frozen `BMethod.apply_trajectory` Protocol requires only the shared `trajectory` argument; concrete methods retain their own optional arguments, and generic pipeline dispatch narrows the bound method to a callable at the invocation boundary.
**Alternatives considered:** Require every implementation to accept `**kwargs: object`; make `BMethod` generic over a parameter specification; or split each argument shape into another public Protocol.
**Reason:** Lerp, SteeringVector, and ActivationPatch prove that trajectory application is shared, but their optional arguments are genuinely different. Requiring `**kwargs: object` made the frozen Protocol statically incompatible with two of its three validating implementations, while additional public Protocols would invent abstractions without new concrete instances.
**Consequences:** Static conformance now matches all three validated B-Methods. Generic callers may forward method-specific arguments only at a deliberate callable-dispatch boundary, while direct users retain precise concrete signatures.
