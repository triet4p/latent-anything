# Sprint 12 Plan

## Sprint Goal
Increment thứ chín (Round 9): thêm **ActivationPatch** — B-Method #3, **model-mediated intervention** — method thứ ba của Layer B, khác triết lý hoàn toàn với Lerp (stateless latent→latent) và SteeringVector (stateful latent→latent). ActivationPatch can thiệp **qua ModelAdapter**: encode → patch latent → decode, output là **data space** (không phải latent space). Đây là instance kích hoạt **Rule of Three freeze**: extract và freeze `BMethod` Protocol từ ba pattern đã proven, migrate Lerp + SteeringVector, promote ra public surface.

## Tại sao ActivationPatch khác triết lý

| Layer B Method | Pattern | Input → Output | Model-mediated? | Instance # |
|---|---|---|---|---|
| Lerp (Sprint 10) | Stateless: pure function | latent × latent → latent | No | #1 |
| SteeringVector (Sprint 11) | Stateful: fit → apply | latent → latent | No | #2 |
| **ActivationPatch (this sprint)** | **Stateful: fit → model-mediated** | **data → (encode→patch→decode) → data** | **Yes — qua ModelAdapter** | **#3 (freeze)** |

Khác biệt cốt lõi:
- Lerp và SteeringVector operate trực tiếp trên `np.ndarray` latent points — pure numpy, không cần model.
- ActivationPatch operate **qua** ModelAdapter: nhận input data (ảnh, text, ...), encode thành latent, patch latent, decode trở lại data space. Output là **data space**, không phải latent space.
- ActivationPatch cần reference tới ModelAdapter — đây là B-Method đầu tiên coupling với adapter layer.
- `apply_trajectory` của ActivationPatch decode từng điểm đã patch → output là list of decoded arrays, không phải `Trajectory`.

Đây là stress test thật sự: B-Method có nhất thiết phải là latent→latent transform không? Hay nó có thể là một pipeline encode→manipulate→decode?

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Task 1: Implement `ActivationPatch` concrete class in `src/latent_anything/methods/activation_patch.py`.
  - Constructor: `ActivationPatch(adapter)` — nhận một `ModelAdapter`-like object (duck-typed: cần `encode` và `decode`). Adapter là required (không optional như `space` của Lerp/SteeringVector).
  - `space` property → `LatentSpace` — delegates to `adapter.latent_space`. Always returns a `LatentSpace` (không `None` như Lerp/SteeringVector — vì adapter luôn có latent_space). Return type: `LatentSpace` (not `LatentSpace | None`).
  - `is_fitted` property → `bool`.
  - `fit(source_data: np.ndarray, target_data: np.ndarray) -> None`:
    - Encode source → `source_latent` (n_samples × dim)
    - Encode target → `target_latent` (n_samples × dim)
    - Compute patch delta: `delta = mean(target_latent, axis=0) - mean(source_latent, axis=0)`
    - Store `_delta: np.ndarray` shape `(dim,)`
    - Validate: source_data and target_data must be 2D, non-empty, same feature dim. Dim must match adapter's expected input dim (validate via encode attempt).
  - `__call__(input_data: np.ndarray) -> np.ndarray`:
    - Encode input → latent (n_samples × dim)
    - Patch: `patched_latent = latent + self._delta` (broadcast delta across samples)
    - Decode patched_latent → output_data (n_samples × adapter_output_dim)
    - Returns decoded data (same shape as input_data typically, but validated via decoder output).
  - `delta` property → `np.ndarray` shape `(dim,)` — the learned patch direction in latent space. Raises `RuntimeError` if not fitted.
  - `apply_trajectory(trajectory: Trajectory) -> np.ndarray` — encode each trajectory point (reshape if needed), patch, decode each. Returns stacked decoded outputs. Shape: `(n_points, decoder_output_dim)`. Note: returns `np.ndarray` (decoded data), NOT `Trajectory` — because the output is in data space, not latent space. This is a deliberate difference from Lerp/SteeringVector — captured by `BMethod` Protocol's flexible `apply_trajectory` return type.
  - All I/O: numpy for data, Trajectory for latent sequences. ModelAdapter may use torch internally but public surface is numpy.
- [ ] Task 2: **Freeze `BMethod` Protocol** — extract from `_b_base.py` → promote to public `protocols.py` (or new `b_protocols.py`).
  - Create `src/latent_anything/methods/b_protocols.py` with frozen `BMethod` Protocol:
    ```python
    @runtime_checkable
    class BMethod(Protocol):
        """Structural protocol for Layer B manipulation methods (frozen at B-Method #3)."""
        @property
        def space(self) -> LatentSpace | None: ...
        @property
        def is_fitted(self) -> bool: ...
        def apply_trajectory(self, trajectory: Trajectory, **kwargs: float) -> Trajectory | np.ndarray: ...
    ```
  - `space`: `LatentSpace | None` — Lerp/SteeringVector có thể None, ActivationPatch luôn có (qua adapter).
  - `is_fitted`: `bool` — Lerp always True (stateless, no fit phase), SteeringVector/ActivationPatch True sau `fit`.
  - `apply_trajectory`: primary trajectory-level operation. Return type union `Trajectory | np.ndarray` — Lerp/SteeringVector return `Trajectory` (latent→latent), ActivationPatch returns `np.ndarray` (latent→data). The union captures the genuine diversity proven by 3 instances.
  - `__call__` is deliberately **NOT** in the Protocol — signatures genuinely differ across instances (Lerp: `(a, b, t)`, SteeringVector: `(latent, strength)`, ActivationPatch: `(input_data)`). Forcing a unified `__call__` would be "design from imagination" (INCREMENTAL.md §3 cấm).
  - Mark docstring: "Frozen at B-Method #3 (ActivationPatch, Sprint 12). Validated by 3 instances with differing philosophies: stateless latent→latent (Lerp), stateful latent→latent (SteeringVector), model-mediated data→data (ActivationPatch)."
  - Remove `_b_base.py` (the UNSTABLE sketch is superseded by the frozen Protocol).
- [ ] Task 3: **Migrate Lerp** to note `BMethod` Protocol conformance.
  - Add `is_fitted` property → always returns `True` (stateless methods are always "ready").
  - Add generic `apply_trajectory(trajectory: Trajectory, **kwargs: float) -> Trajectory` method that delegates to existing trajectory ops:
    - If `kwargs` contains `"other"` (a Trajectory) and `"t"` (float): delegate to `between(trajectory, kwargs["other"], kwargs["t"])`.
    - If `kwargs` contains `"n_steps"` (int): delegate to `blend_sequence(trajectory, kwargs["n_steps"])`.
    - Else: raise `ValueError` with helpful message.
  - Update docstring: note conformance to `BMethod` Protocol (structural, duck-typed).
  - Keep existing methods `between`, `blend_sequence`, `__call__` — they are instance-specific and still valid.
- [ ] Task 4: **Migrate SteeringVector** to note `BMethod` Protocol conformance.
  - Already has `space`, `is_fitted`, `apply_trajectory` → conforms structurally.
  - Update docstring: note conformance to `BMethod` Protocol. Note that `fit` is instance-specific (not in Protocol).
  - No code changes needed beyond docstring.
- [ ] Task 5: Export `BMethod` and `ActivationPatch` from `src/latent_anything/methods/__init__.py`. Add both to `__all__`. `BMethod` joins `Method` as the second public Protocol in the methods package.
- [ ] Task 6: End-to-end demo script `scripts/end_to_end_activation_patch_demo.py` — two scenarios:
  - **Scenario A (VAE latent arithmetic)**: Train a tiny VAE on synthetic 2D grid data (e.g. `make_blobs` or simple geometric shapes). Fit `ActivationPatch(adapter=vae)` with source=cluster_A, target=cluster_B. Apply patch to test samples from cluster_A → decode → visualize patched reconstruction vs original reconstruction side-by-side. Show that patched outputs morph toward cluster_B characteristics.
  - **Scenario B (Trajectory patching)**: Create a trajectory of latent points from cluster_A to cluster_B (via Lerp). Apply `ActivationPatch.apply_trajectory(trajectory)` → decode each point → create a grid visualization showing the morphing sequence. Compare with direct latent interpolation (Lerp → Trajectory) to highlight data-space vs latent-space perspective.
  - Use matplotlib: 2×2 grid showing (1) original reconstruction, (2) patched reconstruction, (3) latent space PCA with patch direction arrow, (4) trajectory morphing grid.
- [ ] Task 7: Tests — pytest for `ActivationPatch` and `BMethod` Protocol:
  - `test_activation_patch_construction_with_vae` — construct with VAE adapter
  - `test_activation_patch_construction_with_random_projection` — construct with RandomProjection adapter
  - `test_activation_patch_space_delegates_to_adapter` — space property returns adapter.latent_space
  - `test_activation_patch_is_fitted_initially_false` — not fitted after construction
  - `test_activation_patch_is_fitted_after_fit` — True after fit
  - `test_activation_patch_fit_computes_delta` — fit on synthetic data produces non-zero delta
  - `test_activation_patch_call_moves_toward_target` — applying patch shifts source-like data toward target-like reconstruction
  - `test_activation_patch_call_preserves_input` — input data not mutated
  - `test_activation_patch_call_before_fit_raises` — RuntimeError
  - `test_activation_patch_delta_before_fit_raises` — RuntimeError
  - `test_activation_patch_fit_empty_source_raises` — ValueError
  - `test_activation_patch_fit_mismatched_dim_raises` — ValueError
  - `test_activation_patch_output_shape_matches_input` — decoded output has same shape as input
  - `test_activation_patch_apply_trajectory_returns_ndarray` — returns np.ndarray, not Trajectory
  - `test_activation_patch_apply_trajectory_shape` — output shape (n_points, decoder_dim)
  - `test_bmethod_protocol_is_runtime_checkable` — `isinstance(obj, BMethod)` works for conforming objects
  - `test_lerp_conforms_to_bmethod` — Lerp passes `isinstance(lerp, BMethod)`
  - `test_steering_vector_conforms_to_bmethod` — SteeringVector passes check
  - `test_activation_patch_conforms_to_bmethod` — ActivationPatch passes check
  - `test_bmethod_rejects_non_conforming` — object without space/is_fitted/apply_trajectory fails check
  - Target: ~20 tests.
- [ ] Task 8: Tooling gate — `ruff check` + `ruff format` + `pyright` strict clean. All existing tests (~227) + new tests (~20) pass.
- [ ] Task 9: Rule of Three §4a — ghi artifact summary:
  > "B-Method #3 (ActivationPatch, model-mediated data→data) → **freeze `BMethod` Protocol**, migrate Lerp + SteeringVector. Three distinct B-Method patterns now proven: stateless latent→latent (Lerp), stateful latent→latent (SteeringVector), model-mediated data→data (ActivationPatch). The frozen `BMethod` Protocol captures the invariant surface (`space`, `is_fitted`, `apply_trajectory`) while deliberately excluding `__call__` (signatures genuinely differ). `_b_base.py` removed — superseded by frozen Protocol. `Method` Protocol unchanged (remains Layer A stateful dim-reduction). This confirms ARCHITECTURE.md's prediction that A/B/C methods have different shapes — the aspirational 'interface chung cho mọi A/B/C method' is disproven by code, replaced by separate fit-for-purpose Protocols."
- [ ] Task 10: ADR check §4c — ActivationPatch exercises the two validated ADRs:
  - `LatentSpace` geometry-keyed ADR (validated): ActivationPatch accesses `adapter.latent_space` and uses `.dim` for validation. Exercised through adapter coupling.
  - Geometry-dispatch ADR (validated): If adapter's latent_space has `unit_norm` geometry, patch delta could be normalized via `space.normalize()`. Exercised indirectly through adapter's `LatentSpace`.
  - `ModelAdapter` 3-mode ADR: `pending` → **partially exercised**. ActivationPatch is the first B-Method to consume a `ModelAdapter` directly. It works with VAE (mode i, explicit learned latent) and RandomProjection (mode i-like, stateless projection). This proves that `ModelAdapter` is consumable from Layer B, but modes (ii) and (iii) remain untested. ADR stays `pending` but now has consumer-side evidence.
  - Append routine entry to `decisions.md`.
- [ ] Task 11: Update `CHANGELOG.md` `[Unreleased]` — add ActivationPatch B-Method, `BMethod` Protocol freeze, Lerp/SteeringVector migration, trajectory-to-data-space demo, and Protocol architecture note under `Added`. Breaking change note under `Changed`: `BMethod` Protocol frozen, `_b_base.py` removed.
- [ ] Task 12: Update `docs/PLAN.md` — Sprint 12 → Completed, Sprint 13 → Active, Milestone 2 nearing completion (one sprint left).

## Rule-of-Three checkpoint (to verify at end)
| Check | Status |
|---|---|
| B-Method instances | Lerp (#1, stateless latent→latent), SteeringVector (#2, stateful latent→latent), ActivationPatch (#3, model-mediated data→data) |
| Philosophies differ? | **Yes** — three genuinely different patterns: stateless pure function, stateful fit→apply, model-mediated encode→patch→decode |
| Rule branch | **Instance #3, khác triết lý** → **Freeze `BMethod` Protocol, migrate** |
| `Method` Protocol? | **Unchanged** — remains Layer A stateful dim-reduction only. A/B/C unification disproven. |
| `_b_base.py`? | **Removed** — UNSTABLE sketch superseded by frozen `BMethod` Protocol in `b_protocols.py`. |
| Lerp migrated? | Added `is_fitted` (always True) + generic `apply_trajectory(**kwargs)`. Docstring notes conformance. |
| SteeringVector migrated? | Already conforms — docstring updated to note conformance. |
| Public surface? | `BMethod` promoted to public via `methods/__init__.py` __all__. |

## ActivationPatch Design Notes
```python
class ActivationPatch:
    """Model-mediated activation patching via encode→patch→decode.

    Unlike Lerp and SteeringVector which operate directly on latent
    points, ActivationPatch works through a ModelAdapter: it encodes
    input data, patches the latent representation, and decodes back
    to data space. The output is in data space (e.g., images), not
    latent space.

    This is B-Method #3 — the third distinct B-Method pattern,
    triggering the Rule of Three freeze for the BMethod Protocol.

    Parameters
    ----------
    adapter
        A ModelAdapter-like object with ``encode``, ``decode``,
        and ``latent_space``. Duck-typed — no inheritance required.
    """

    def __init__(self, adapter) -> None:
        self._adapter = adapter
        self._delta: np.ndarray | None = None

    @property
    def space(self) -> LatentSpace:
        """Return the adapter's LatentSpace."""
        return self._adapter.latent_space

    @property
    def is_fitted(self) -> bool:
        """True if fit() has been called successfully."""
        return self._delta is not None

    @property
    def delta(self) -> np.ndarray:
        """The learned patch direction in latent space."""
        if self._delta is None:
            raise RuntimeError("Not fitted. Call fit() first.")
        return self._delta.copy()

    def fit(self, source_data: np.ndarray, target_data: np.ndarray) -> None:
        """Learn patch delta: mean(target_latent) - mean(source_latent)."""
        if source_data.ndim != 2 or target_data.ndim != 2:
            raise ValueError("Expected 2D arrays (n_samples, n_features)")
        source_latent = self._adapter.encode(source_data)
        target_latent = self._adapter.encode(target_data)
        if source_latent.shape != target_latent.shape:
            raise ValueError(
                f"Latent shape mismatch: source {source_latent.shape}, "
                f"target {target_latent.shape}"
            )
        self._delta = target_latent.mean(axis=0) - source_latent.mean(axis=0)

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        """Encode → patch latent → decode → return data-space output."""
        if self._delta is None:
            raise RuntimeError("Not fitted. Call fit() first.")
        latent = self._adapter.encode(input_data)
        patched = latent + self._delta  # broadcast
        return self._adapter.decode(patched)

    def apply_trajectory(self, trajectory: Trajectory, **kwargs: float) -> np.ndarray:
        """Apply patching to each trajectory point, decode, return stacked data.

        Returns data-space output (np.ndarray), NOT a Trajectory —
        because ActivationPatch outputs are in data space, not latent space.
        """
        _ = kwargs  # reserved for future use (e.g., blend factor)
        data = trajectory.to_numpy()
        decoded = np.stack([self(self._adapter.decode(pt)) for pt in data])
        # Simpler: encode trajectory points as batch, patch, decode
        # But trajectory points are latent already, so we just decode + patch? 
        # Actually the trajectory contains latent points. We decode them,
        # but patching is latent→latent. Let me redesign...
        ...
```

**⚠️ Design challenge — `apply_trajectory` for model-mediated methods:**

Lerp/SteeringVector `apply_trajectory` takes a `Trajectory` (latent points) and returns a `Trajectory` (patched latent points). But ActivationPatch's `__call__` takes **data** (images, etc.), not latent points. A trajectory of latent points cannot be directly fed to `__call__` — it would go through `encode` (double-encoding).

Two options:
1. `apply_trajectory` patches each latent point directly (add delta) → decodes each → returns decoded data. This is the natural trajectory-level operation for ActivationPatch.
2. Skip `apply_trajectory` for ActivationPatch — it doesn't make sense for model-mediated methods.

**Decision: Option 1.** `apply_trajectory(trajectory)` adds delta to each latent point in the trajectory, decodes each, returns `np.ndarray` of decoded outputs. This is a valid trajectory-level operation — it says "given this trajectory of latent states, what does each state decode to after patching?"

Updated implementation:
```python
def apply_trajectory(self, trajectory: Trajectory, **kwargs: float) -> np.ndarray:
    """Patch each latent point in trajectory, decode, return data-space outputs."""
    _ = kwargs
    if self._delta is None:
        raise RuntimeError("Not fitted. Call fit() first.")
    data = trajectory.to_numpy()  # (n_points, dim)
    patched = data + self._delta   # broadcast delta
    return self._adapter.decode(patched)  # (n_points, decoder_output_dim)
```

This preserves the `BMethod` Protocol contract (`apply_trajectory` exists) while correctly handling the data-space output. The return type `Trajectory | np.ndarray` in the Protocol captures this diversity.

## Notes / Blockers
* **No new dependency.** ActivationPatch uses existing ModelAdapter instances (VAE, RandomProjection) + numpy + Trajectory.
* **`Method` Protocol is NOT modified.** Layer A and Layer B now have separate Protocols (`Method` for stateful dim-reduction, `BMethod` for manipulation). The aspirational "interface chung cho mọi A/B/C method" from ARCHITECTURE.md §2 is **disproven by code** — exactly the outcome INCREMENTAL.md §3 predicts when ADRs are treated as hypotheses to validate, not specs to build.
* **`_b_base.py` is removed.** The UNSTABLE sketch served its purpose (capturing what Lerp+SteeringVector shared). The frozen `BMethod` Protocol in `b_protocols.py` is the permanent replacement, validated by 3 instances.
* **`apply_trajectory` has different return types across B-Methods.** Lerp/SteeringVector return `Trajectory` (latent→latent). ActivationPatch returns `np.ndarray` (latent→data). The `BMethod` Protocol captures this with union return type `Trajectory | np.ndarray`. This is correct — forcing uniform return types would misrepresent the genuine diversity.
* **`__call__` is NOT in `BMethod` Protocol.** Signatures differ across the 3 instances. The Protocol captures the invariant surface, not the variant one. Callers use instance-specific `__call__` with duck-typing.
* **ActivationPatch couples to ModelAdapter.** This is the first B-Method that depends on an adapter. It proves that B-Methods can consume ModelAdapters, adding consumer-side evidence for the ModelAdapter 3-mode ADR.
* Each task one commit per Conventional Commits (`feat(methods):`, `refactor(methods):`, `test(methods):`, `chore:`).
