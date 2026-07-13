# Changelog

## [Unreleased]

### Added

- Added a revision-pinned decoder-only transformer integration (`TransformerLMIntegration`) with direct logit lens, typed input/hidden-state/lens-result values with NumPy-facing public payloads and full model/tokenizer provenance. (#sprint-39)
- Added native `output_hidden_states=True` as the canonical observation path for transformer hidden states, with verified embedding/residual/final hidden-state indexing and shapes. (#sprint-39)
- Added a direct logit lens implementation with explicit final-normalization (LayerNorm) and output-head (LM head) assumptions; learned/tuned translators deferred. (#sprint-39)
- Added validation of native hidden states and final logits against direct backend execution, including padded-token masking and final-layer parity checks. (#sprint-39)
- Added hook-based activation intervention support via `ActivationCaptureSession` for one bounded activation intervention at a specified layer, with hook cleanup verification. (#sprint-39)
- Added token rank/probability trajectory measurement across layers, with stability tracking under predeclared prompt perturbations. (#sprint-39)
- Added comprehensive test suite: 38 offline tests with fake backend + 11 marked real-checkpoint tests, plus a reproducible artifact demo script. (#sprint-39)
- Added a `transformers` optional install extra with pinned GPT-2 (`gpt2` model at revision `e7da7f2`, 124M parameters) for the transformer integration. (#sprint-39)

- Evidence-ledger validation now inventories all theory capabilities, verifies local evidence links in CI, and reports the D2/D3 stable-coverage denominator without downloading optional models. (#sprint-27)
- `LatentValue` carries immutable flat batches and structured latent states with explicit `LatentSpace` association, safe NumPy conversion, and beta `Trajectory` compatibility. (#sprint-29)
- `LatentSpace` now supports categorical `discrete_code` geometry with codebook validation, normalized Hamming distance, and an explicit no-continuous-interpolation policy. (#sprint-30)
- Registry and config construction now use canonical `analysis` and `intervention` kinds; legacy beta kinds remain aliases with migration diagnostics until `0.9.0`. (#sprint-31)
- Added internal PyTorch activation capture sessions with safe hook lifecycle, NumPy outputs, provenance metadata, and intervention callbacks. (#sprint-32)
- Added optional `transformers`, `3d`, and `lerobot` installation extras alongside the Diffusers integration. (#sprint-33)
- Added a CPU-reproducible convolutional VAE integration evaluated on sklearn digits image data. (#sprint-34)
- Added a lazy, revision-pinned Diffusers `AutoencoderKL` integration with explicit network acquisition gating and reproducible interpolation diagnostics. (#sprint-35)
- Added a control-aware VAE explanation benchmark with held-out probes, negative controls, seed intervals, and decoded steering effects. (#sprint-36)
- Added a revision-pinned conditional text-to-image diffusion integration (`DiffusersConditionalPipeline`) that records scheduler latent states via native `callback_on_step_end` and selected denoiser activations via `ActivationCaptureSession`, with typed NumPy-faced request/result objects and separate `LatentSpace` descriptors for VAE bottlenecks, scheduler states, and denoiser activations. (#sprint-37)
- Added a combined `diffusers-full` optional install extra that includes both `diffusers` and `transformers` dependencies for the conditional diffusion pipeline. (#sprint-37)
- Added `SchedulerIntervention` data type and intervention support to `DiffusersConditionalPipeline.generate()` — additive edits on scheduler latent states during denoising via `callback_on_step_end`, with `random_direction()` and `matched_norm_direction()` helpers. (#sprint-38)
- Added deterministic offline smoke tests for `SchedulerIntervention` validation, direction helpers, intervention passthrough in `generate()`, and a gated real-checkpoint benchmark that verifies the intervention changes intermediate latents. (#sprint-38)
- Added a comprehensive experiment script (`diffusers_conditional_intervention_experiment.py`) that compares no-edit/prompt-only/random-direction/matched-norm controls across multiple seeds with target change, SSIM, latent norm drift, and trajectory-cosine metrics, plus a timestep-by-strength sweep. (#sprint-38)

### Changed
- `09_gaussian_rasterization.ipynb`: improve Exp 4 and Exp 5 visualizations — denser overlap scenes, cumulative contribution breakdown, rendered image quality comparison, percent-savings heatmaps.

### Fixed

- Evidence-ledger schema-v2 validation now enforces typed `role`/`path` records and the required evidence roles for every D1, D2, and D3 claim.
- `LatentValue.metadata` now returns defensive immutable snapshots, including NumPy metadata that callers attempt to make writable again.
- `end_to_end_showcase_demo.py`: add missing `pydantic` to inline script dependencies so `uv run --script` resolves it correctly; add `sys.stdout.reconfigure(encoding='utf-8')` for Windows Unicode support.

## [0.1.0-beta.1] - 2026-07-10

### Release Summary

`0.1.0-beta.1` is the first core-framework beta for Latent Anything. It ships concrete latent-space primitives, first adapter modes, Layer A inspection methods, Layer B manipulation methods, registry/config construction, analysis/manipulation pipelines, and first Layer C runtime helpers.

This is a pre-1.0 beta, not an API-stability promise and not a claim that every theory layer is implemented.

### Demo And Artifact Links

- [Release demo index](artifacts/release_demo_index_0.1.0-beta.1.md)
- [Showcase summary](artifacts/showcase_demo_summary.txt)
- [Showcase plot](artifacts/showcase_demo_plot.png)
- [Gaussian renderer demo](artifacts/gaussian_renderer_demo.png)
- [Gaussian-set demo plot](artifacts/gaussian_set_demo_plot.png)
- [Cache demo summary](artifacts/cache_demo_summary.txt)
- [Batch executor demo summary](artifacts/batch_executor_demo_summary.txt)
- [Async runtime demo summary](artifacts/async_runtime_demo_summary.txt)

### Install And Test Notes

- Package version metadata is `0.1.0b1` for PEP 440 compatibility.
- Recommended release tag: `v0.1.0-beta.1`.
- Local install path: `uv sync --locked`.
- Release gate: `uv sync --locked`, `uv run ruff check src tests scripts`, `uv run ruff format --check src tests scripts`, `uv run pyright`, and `uv run pytest`.

### Known Limitations

- Package is not published to PyPI in this sprint.
- External plugin discovery via Python entry points is not shipped yet.
- Streaming runtime is not shipped yet.
- Interactive Plotly/notebook widget visualization is not shipped yet.
- Probing/TCAV, clustering, feature attribution, trajectory similarity, rollout, planning, and discrete latent adapters remain future work.
- Large modules such as `pipeline.py` and `latent_space.py` are beta-acceptable but recorded as post-beta refactor candidates once new concrete execution or geometry stories justify extraction.

### Theory Coverage Caveats

- The release is theory-informed, not theory-complete.
- Shipped code covers core primitives, selected representation learning adapters, selected geometry-aware operations, first manipulation methods, concrete pipelines, and runtime helpers.
- Theory layers for planning, latent prediction, discrete latents, full probing/TCAV, and large-scale world-model integrations are documented but not implemented in this beta.

### Added

- **Async runtime wrappers** — Added `AnalysisPipeline.run_async()`, `ManipulationPipeline.run_data_async()` / `run_trajectory_async()`, and `BatchExecutor.encode_async()` / `decode_async()` / `transform_async()` as thread-backed `asyncio` wrappers over the existing concrete runtime paths. Sync APIs remain available for current scripts and notebooks. (#sprint-24)
- **Runtime profiling hooks** — Added `RuntimeProfiler`, `RuntimeProfile`, and `ProfileEvent` for stage-level timing across `cache`, `encode`, `method`, and `decode` operations. `AnalysisPipeline`, `ManipulationPipeline`, and `BatchExecutor` accept optional profiling hooks without changing their return types. (#sprint-24)
- **Async runtime demo script** — `scripts/end_to_end_async_runtime_demo.py` runs one `AnalysisPipeline` job and one `ManipulationPipeline` job concurrently and writes `artifacts/async_runtime_demo_summary.txt`. Local snapshot: concurrent wall time 98.607 ms with per-stage breakdowns for both jobs. (#sprint-24)

- **`InMemoryCache` — cache backend #1** — `src/latent_anything/runtime/cache.py` with memory-only `get`, `set`, `clear`, and stats. Stores defensive copies of numpy arrays so callers cannot mutate cached values by accident. (#sprint-23)
- **Stable `CacheKey` structure** — Records namespace, operation, component name, component config hash, behavior-affecting component state hash, input data hash, and framework version when available. Data/config/state hashes use SHA-256 without introducing a pickle or disk format. (#sprint-23, #sprint-25)
- **AnalysisPipeline cache integration** — `AnalysisPipeline(adapter, method, cache=InMemoryCache())` caches adapter `encode` latents for repeated identical runs while always fitting stateful Layer A methods on the current pipeline instance. (#sprint-23, #sprint-25)
- **Cache demo script** — `scripts/end_to_end_cache_demo.py` demonstrates adapter-encode reuse through `AnalysisPipeline` and writes `artifacts/cache_demo_summary.txt`. (#sprint-23, #sprint-25)
- **New public exports** — `InMemoryCache`, `CacheKey`, and `CacheStats` exported from top-level `latent_anything` package and `latent_anything.runtime`. (#sprint-23)
- **`BatchExecutor` — Runtime #1** — `src/latent_anything/runtime/batch_executor.py` with deterministic first-axis numpy chunking, eager/synchronous execution, and output concatenation in original order. Supports generic `map_array()` plus adapter `encode()`, adapter `decode()`, and Layer A method `transform()` helpers. (#sprint-22)
- **Batch executor tests** — 23 tests covering construction, exact divisibility, remainder batches, batch size 1, batch size larger than data, invalid batch sizes, adapter `encode`/`decode`, PCA `transform`, output order/shape preservation, dtype preservation, and operation-output validation. (#sprint-22)
- **Batch executor demo script** — `scripts/end_to_end_batch_executor_demo.py` prints direct-vs-batched timing snapshots on synthetic data and writes `artifacts/batch_executor_demo_summary.txt`. Local snapshot: `RandomProjection.encode` direct 1.934 ms vs batched 3.302 ms; `PCA.transform` direct 1.507 ms vs batched 4.475 ms. (#sprint-22)
- **New public export** — `BatchExecutor` exported from top-level `latent_anything` package and `latent_anything.runtime`. (#sprint-22)
- **`ManipulationPipeline` — Pipeline #2** — `src/latent_anything/pipeline.py` with a concrete manipulation pipeline for Layer B methods. Supports two stories: (1) adapter-mediated data-space output via `run_data()` — encode → BMethod → decode → metric-ready `np.ndarray` (used with `ActivationPatch`); (2) latent-only trajectory output via `run_trajectory()` — BMethod `apply_trajectory` returning a new `Trajectory` (used with `SteeringVector`, `Lerp`). Deliberately avoids a generic `run()` because `__call__` signatures differ across B-Methods. (#sprint-21)
- **`_PipelineBase` — shared pipeline sketch** — Minimal base recording the common surface (adapter + method + optional `latent_space`) between `AnalysisPipeline` and `ManipulationPipeline`. Sketch only — freeze waits for Pipeline #3 (Rule of Three). (#sprint-21)
- **`ManipulationPipelineSpec` + `build_manipulation_pipeline_from_config`** — Config-backed construction path using Sprint 18 config machinery. Supports optional adapter spec for data-space stories. (#sprint-21)
- **28 manipulation pipeline tests** — covering construction (6), data-space story (3), trajectory story (4), fit delegation (3), convenience methods (3), spec model invariants (4), and config-backed construction errors (5). (#sprint-21)
- **Manipulation pipeline demo script** — `scripts/end_to_end_manipulation_demo.py` reproduces the Sprint 13 showcase path through Pipeline #2: ActivationPatch (data-space) and SteeringVector (trajectory) stories with matplotlib visualisation of before/after metrics and trajectory spread. (#sprint-21)
- **New public exports** — `ManipulationPipeline`, `ManipulationPipelineSpec`, `build_manipulation_pipeline_from_config` exported from top-level `latent_anything` package. (#sprint-21)

### Changed

- **`ActivationPatch` and `ManipulationPipeline` staged path** — `ActivationPatch` now exposes `apply_latent()`, allowing `ManipulationPipeline.run_data()` and `run_data_async()` to profile `encode → method → decode` as separate runtime stages instead of one opaque method call. (#sprint-24)
- **New public runtime exports** — `RuntimeProfiler`, `RuntimeProfile`, and `ProfileEvent` are exported from both `latent_anything.runtime` and the top-level `latent_anything` package. (#sprint-24)
- Test suite: 594 total tests (589 existing + 4 runtime async/profiling + 1 demo smoke). (#sprint-24)

- Test suite: 589 total tests (575 existing + 13 cache + 1 demo smoke). (#sprint-23)
- Test suite: 575 total tests (551 existing + 23 batch executor + 1 demo smoke). (#sprint-22)
- **`AnalysisPipeline`** now inherits from `_PipelineBase` — no behavioral change, purely structural. (#sprint-21)
- **`__init__.py`** — Added manipulation pipeline exports to both the import block and `__all__`. (#sprint-21)
- Test suite: 551 total tests (523 existing + 28 manipulation pipeline). (#sprint-21)

- **Separated built-in registry module** — `src/latent_anything/_plugin_builtins.py` as the single stable import location where all built-in adapters and methods are registered into `GLOBAL_REGISTRY`. This decouples `registry.py` from concrete class dependencies, making it pure infrastructure. (#sprint-19)
- **Internal plugin extraction contract** — Documented in `_plugin_builtins.py` docstring: registration-only (no re-exports), deterministic order, no circular imports, one-to-one with built-in classes, and entry-point readiness for future external plugins. (#sprint-19)
- **Parity test suite** — 22 new tests in `test_parity.py` covering registry constructor vs direct import constructor for all 10 built-in classes (4 adapters + 3 method_a + 3 method_b), plus factory identity checks proving `registry.lookup("name").factory` is the class itself. (#sprint-19)
- **Demo smoke test suite** — 15 new tests in `test_demo_smoke.py` verifying that every `scripts/end_to_end_*.py` demo's core imports and helpers still work after the registry refactoring. (#sprint-19)

- **`registry.py`** — Refactored to infrastructure-only: removed all adapter/method class imports and the registration block at module bottom. `Registry` class, kind constants, convenience helpers, and `GLOBAL_REGISTRY` singleton remain unchanged. (#sprint-19)
- **`__init__.py`** — Added `from latent_anything import _plugin_builtins` to trigger built-in registration on package import, before any registry-dependent modules (like `config.py`). (#sprint-19)
- Test suite: 502 total tests (465 existing + 22 parity + 15 demo smoke). (#sprint-19)

- **Registry-backed config instantiation** — `src/latent_anything/config.py` with pydantic v2 `ObjectSpec` model (`kind`, `name`, `params`), `build_from_config(spec)` that resolves registry entries and instantiates them, and `build_from_dict(data)` convenience wrapper. Config instantiation instance #1 — registry-local, deliberately narrow, no Pipeline/workflow language. (#sprint-18)
- **Nested spec resolution** — `build_from_config` recursively resolves nested `ObjectSpec` values inside `params`, enabling Layer B methods with adapter dependencies to be built from config (e.g. `ActivationPatch(adapter=VAE(...))`). Nested specs work as both `ObjectSpec` instances and plain dicts. (#sprint-18)
- **Clear validation errors** — `build_from_config` raises `KeyError` with sorted available names for unknown entries, `ValueError` with kind mismatch details, and `TypeError` with the failing params for instantiation failures. (#sprint-18)
- **Config-driven demo script** — `scripts/end_to_end_config_demo.py` builds the showcase object stack (VAE, PCA, Lerp, ActivationPatch with nested VAE) entirely from pydantic config specs without manual constructor calls. No Pipeline abstraction introduced. (#sprint-18)
- **New public exports** — `ObjectSpec`, `build_from_config`, `build_from_dict` exported from the top-level `latent_anything` package. (#sprint-18)
- **pydantic v2 dependency** — `pydantic>=2.0,<3.0` added to project dependencies for config model validation. (#sprint-18)
- Test suite: 36 config tests covering ObjectSpec construction (5), adapter building (4), Layer A method building (4), Layer B method building (4), error cases (7), custom registry (2), build_from_dict (3), and all six required classes (6). 465 total tests. (#sprint-18)

- **In-process registry** — `src/latent_anything/registry.py` with `Registry` class (`OrderedDict`-backed), frozen `RegistryEntry` dataclass, kind constants (`KIND_ADAPTER`, `KIND_METHOD_A`, `KIND_METHOD_B`), module-level convenience helpers (`register`, `lookup`, `list_entries`), and a `GLOBAL_REGISTRY` singleton. Designed as registry instance #1 with no Python entry points yet. (#sprint-17)
- **Built-in class registration** — All 10 built-in classes registered in `GLOBAL_REGISTRY`: VAE, RandomProjection, HiddenStateAdapter, GaussianRendererAdapter (adapters); PCA, UMAP, SAE (Layer A methods); Lerp, SteeringVector, ActivationPatch (Layer B methods). Registration uses class references as factories with metadata (protocol, description, source). (#sprint-17)
- **Deterministic ordering** — `Registry.list()` and `Registry.list(kind=...)` return entries in insertion order, guaranteed by `OrderedDict` backing. (#sprint-17)
- **Duplicate and missing-name guards** — `Registry.register()` raises `ValueError` on duplicate names (with descriptive message including registry name). `Registry.lookup()` raises `KeyError` on missing names. (#sprint-17)
- **Standalone registry demo** — `scripts/end_to_end_registry_demo.py` lists all registered entries grouped by kind, demonstrates lookup, duplicate-guard, and missing-name guard. (#sprint-17)
- Test suite: 48 registry tests covering construction (5), registration (6), lookup (5), listing (6), factory retrieval (2), RegistryEntry invariants (3), convenience helpers (6), GLOBAL_REGISTRY builtins (8), error cases (5), and no-breakage verification (2). (#sprint-17)

- **`GaussianRendererAdapter`** — ModelAdapter #4 (mode iii: deterministic renderer), closing the last evidence gap for the 3-mode ModelAdapter ADR. A concrete adapter that treats a fixed-size set of 2D Gaussian parameters as a latent representation and renders them into an RGB image via a deterministic numpy-only Gaussian splat renderer. `latent_space` returns a `gaussian_set` `LatentSpace` with position(2) + scale(2) + opacity(1) + color(3) = 8 columns. Conforms to both `ModelAdapter` and `DecodableAdapter` Protocols. (#sprint-16)
- **Deterministic 2D Gaussian splat decode** — `decode(latent) -> (H, W, 3)` renders Gaussian parameters with additive alpha composition: each Gaussian contributes `weight = opacity * exp(-((x-px)²/(2*sx²) + (y-py)²/(2*sy²)))` and colour is accumulated across all Gaussians with no normalisation (avoiding the single-Gaussian flat-colour artifact). Pure numpy, no CUDA or ``gsplat`` dependency. (#sprint-16)
- **Heuristic grid-based encode** — `encode(image) -> (n_gaussians, 8)` provides a simple approximate inverse: places Gaussians on a regular grid, samples colour from pixel centres, with small random jitter for visual variety. Documented as latent-source-first — the adapter is designed for create-then-decode workflows, not encode-then-decode roundtrips. (#sprint-16)
- End-to-end demo script `scripts/end_to_end_gaussian_renderer_demo.py` — synthetic Gaussian-set latent → decode to RGB → interpolation sequence via `LatentSpace` `gaussian_set` geometry → heuristic encode → roundtrip → 2×3 matplotlib figure with state A and B, midpoint, 3-step interpolation, and info panel. (#sprint-16)
- Test suite: 52 GaussianRendererAdapter tests covering construction (5), latent_space property (9), decode shape/range (4), determinism (2), opacity/colour constraints (3), decode input validation (9), encode output shape/constraints (9), roundtrip (3), no-mutation (2), and Protocol conformance (6). 381 total tests. (#sprint-16)

- **`gaussian_set` geometry** — Geometry case #3 for `LatentSpace`, the first structured set-like latent shape. Fixed-size Gaussian parameter sets with position, scale, opacity, and color channels. Enables `LatentSpace(dim=10, geometry="gaussian_set", n_gaussians=100)` with `shape=(100, 10)` while preserving flat-geometry ergonomics for existing callers. (#sprint-15)
- **Gaussian parameter schema** — `gaussian_set_param_layout` metadata key documenting column layout (position, scale, opacity, color start indices and lengths). (#sprint-15)
- **`LatentSpace` Gaussian-set validation** — `validate_point()` checks shape `(n_gaussians, param_dim)` plus numeric constraints: scale components > 0 (log-normal storage), opacity in [0,1], color channels in [0,1]. (#sprint-15)
- **Permutation-aware distance for Gaussian sets** — `distance()` sorts both sets by position (lexicographic) then computes Frobenius norm. Deterministic, O(n log n), avoids optimal-assignment complexity. Invariant to Gaussian reordering. (#sprint-15)
- **Interpolation for Gaussian-set states** — `interpolate()` sorts by position for correspondence, then interpolates position (lerp), scale (log-space lerp for positivity), opacity (lerp + clamp [0,1]), and color (lerp + clamp [0,1]). (#sprint-15)
- **`n_gaussians` and `param_dim` properties** — `LatentSpace.n_gaussians` returns `int | None` (set count for `gaussian_set`, `None` for flat geometries). `LatentSpace.param_dim` returns total parameter dimensionality per Gaussian. (#sprint-15)
- End-to-end demo script `scripts/end_to_end_gaussian_set_demo.py` — synthetic Gaussian-set points → validates construction/param-layout → permutation-aware distance (invariant under shuffle) → interpolation at 5 steps → 2×3 matplotlib figure showing position, scale, opacity, color, distance profile, and info panel. (#sprint-15)
- Test suite: 34 new `LatentSpace` tests covering Gaussian-set construction (8), validation (9), distance (4), interpolation (8), property-based invariants (2), normalize (1), and backward compatibility with flat geometries (2). 82 total LatentSpace tests. (#sprint-15)

- **Adapter Protocols frozen** — `adapters/protocols.py` with structural `typing.Protocol` split: base `ModelAdapter` (shape-generic: `encode` + `latent_space`), `DecodableAdapter` (shape-generic `+decode`), and `FlatBatchDecodableAdapter` (narrow flat batch-matrix `encode`/`decode` surface for methods such as `ActivationPatch`). The split reflects the core evidence from three adapters: `decode` is NOT universal, and decodable does not always mean flat-batch. Frozen at ModelAdapter #3 (Sprint 14), then refined after Sprint 16's structured deterministic renderer. (#sprint-14, #sprint-16)
- **`HiddenStateAdapter`** — ModelAdapter #3 (mode ii: no-explicit-latent), demonstrating the pattern where hidden-state activations *are* the latent representation. Fixed random 2-layer ReLU MLP with He initialisation. `encode(data)` returns `(n_samples, hidden_dim)` hidden activations. No `decode` method — there is no decoder. `latent_space` returns Euclidean `LatentSpace` with `exposure_mode="hidden_state"` metadata. Pure numpy — no torch or heavy transformer dependency. (#sprint-14)
- **`ActivationPatch` runtime guard** — now requires `FlatBatchDecodableAdapter` with `isinstance` check at construction. `HiddenStateAdapter` (no `decode`) and `GaussianRendererAdapter` (structured decodable, not flat-batch) are cleanly rejected with `TypeError`. (#sprint-14, #sprint-16)
- End-to-end demo script `scripts/end_to_end_hidden_state_demo.py` — synthetic 8D clusters → HiddenStateAdapter encode → PCA/UMAP 2D visualization → no decode story. (#sprint-14)
- Test suite: 30 HiddenStateAdapter tests covering construction, latent_space property, encode shape/determinism/nonlinearity/validation, ModelAdapter conformance, DecodableAdapter and FlatBatchDecodableAdapter non-conformance, ActivationPatch rejection, and reproducibility. (#sprint-14)

- **First composition showcase** — Sprint 13 end-to-end story: generate synthetic 8D cluster data → train VAE → encode to 3D latent → PCA Layer A introspection (source/target/failure regions visible) → baseline metrics → ActivationPatch Layer B edit (encode → patch → decode) → post-edit metrics (68.2% distance improvement toward target) → Lerp trajectory panel → composite 2×2 figure. Proves existing primitives compose without new abstractions. (#sprint-13)
- Showcase config `scripts/showcase_config.py` — lightweight local dict (not framework-wide config system) with seed 42, data generation params, VAE params, split config, and output paths. Guarantees reproducibility for the Sprint 13 narrative. (#sprint-13)
- Showcase script `scripts/end_to_end_showcase_demo.py` — orchestration entry point implementing the full compose narrative. Reuses existing patterns from `end_to_end_vae_demo.py`, `end_to_end_activation_patch_demo.py`, and `end_to_end_lerp_demo.py` rather than duplicating. (#sprint-13)
- Test suite: 18 showcase tests covering config parsing, data generation shapes/range/labels, split correctness, baseline metric semantics (finite, positive), post-edit improvement verification, input non-mutation, PCA projection shapes, and trajectory panel lengths. (#sprint-13)

- `ActivationPatch` — B-Method #3 (Layer B / Manipulation), model-mediated activation patching that works through a ModelAdapter (encode → patch → decode). The first B-Method whose output is in **data space** (not latent space), with a required adapter (VAE/RandomProjection). Supports `fit(source_data, target_data)` to learn a latent delta, `__call__(input_data)` for data-space patching, and `apply_trajectory(trajectory)` returning `np.ndarray` (not `Trajectory`). Pure numpy public surface — torch used internally via adapter. (#sprint-12)
- `BMethod` Protocol — frozen structural `typing.Protocol` defining `space` / `is_fitted` / `apply_trajectory`, promoted to public surface. Deliberately excludes `__call__` (signatures differ across instances). Frozen at B-Method #3 (ActivationPatch, Sprint 12), validated by 3 distinct philosophies: stateless latent→latent (Lerp), stateful latent→latent (SteeringVector), model-mediated data→data (ActivationPatch). (#sprint-12)
- End-to-end demo script `scripts/end_to_end_activation_patch_demo.py` — two scenarios: (A) VAE latent arithmetic with blob cluster data → patch reconstruction morphs from cluster A toward cluster B, (B) trajectory patching with Lerp trajectory → decode each point → visualize data-space morphing sequence. 2×2 matplotlib grid with PCA latent space arrow and trajectory comparison. (#sprint-12)
- Test suite: 29 ActivationPatch + BMethod Protocol tests covering construction with VAE/RandomProjection, structured decoder rejection, space delegation, is_fitted lifecycle, fit delta computation, `__call__` semantic correctness (moves toward target), input non-mutation, error cases (before-fit, empty, mismatched dim), apply_trajectory returning ndarray with correct shape, and Protocol conformance checks (Lerp/SteeringVector/ActivationPatch all pass; non-conforming objects fail). (#sprint-12)
- `SteeringVector` — B-Method #2 (Layer B / Manipulation), stateful steering method that learns a unit direction from contrast pairs (`fit(positives, negatives)`) and applies it to latent vectors (`__call__(latent, strength)`). Supports `direction` property, `apply_trajectory(trajectory, strength)` for trajectory-level steering, and optional `LatentSpace` for geometry-aware post-steer normalization (e.g. project back to sphere). Pure numpy — no torch leakage. (#sprint-11)
- End-to-end demo script `scripts/end_to_end_steering_demo.py` — two scenarios: (A) Euclidean steering with synthetic 8D contrast clusters → PCA to 2D → steering path at multiple strengths, (B) spherical steering with unit-norm 3D contrast data → geometry-aware normalization → points stay on sphere. 1×2 matplotlib visualization with arrows and strength annotations. (#sprint-11)
- Test suite: 32 SteeringVector tests covering construction with/without LatentSpace, fit direction learning, direction property (before/after fit), `__call__` edge cases (zero/negative/wrong-dim/strength scaling), input non-mutation, apply_trajectory shape/semantics, spherical normalization at multiple strengths, and no-torch-leakage verification. (#sprint-11)

- **`ModelAdapter` 3-mode ADR gained mode (ii) evidence** — modes (i) and (ii) confirmed by running code (VAE, RandomProjection, HiddenStateAdapter). The Protocol split was frozen at instance #3, while full ADR validation remained pending until Sprint 16's deterministic renderer mode (iii). (#sprint-14)
- **`_ModelAdapterBase` removed** — superseded by frozen `ModelAdapter`/`DecodableAdapter` Protocols. The UNSTABLE internal sketch (`_base.py`) is replaced by the public Protocol surface. (#sprint-14)
- **`ActivationPatch`** — adapter parameter now has a runtime `isinstance` guard requiring `FlatBatchDecodableAdapter`. `HiddenStateAdapter` (no `decode`) and structured decoders like `GaussianRendererAdapter` are cleanly rejected at construction with `TypeError`. (#sprint-14, #sprint-16)
- **`adapters/__init__.py`** — exports `ModelAdapter`, `DecodableAdapter`, `FlatBatchDecodableAdapter`, and `HiddenStateAdapter`. Docstring updated to reflect frozen Protocol design. (#sprint-14, #sprint-16)
- **`VAE` and `RandomProjection` docstrings** — updated to reference `ModelAdapter` and `DecodableAdapter` Protocols instead of the removed `_ModelAdapterBase`. (#sprint-14)

- **`BMethod` Protocol frozen** — `_b_base.py` removed, superseded by frozen `BMethod` Protocol in `methods/b_protocols.py`. Lerp and SteeringVector migrated to note Protocol conformance. This is the third Rule of Three freeze in the project (after `Method` at Sprint 6 and geometry dispatch patterns at Sprint 9). The separation of A (`Method`) and B (`BMethod`) Protocols confirms that A/B/C layers have fundamentally different method shapes — the aspirational unified interface is disproven by code. (#sprint-12)
- `Lerp` — added `is_fitted` property (always `True` — stateless methods are always ready) and generic `apply_trajectory(**kwargs)` method supporting `other`+`t` (delegates to `between`) or `n_steps` (delegates to `blend_sequence`). Docstring updated to note `BMethod` Protocol conformance. (#sprint-12)
- `SteeringVector` — docstring updated to note `BMethod` Protocol conformance. No code changes (already conforms structurally). (#sprint-12)

- `Lerp` — B-Method #1 (Layer B / Manipulation), stateless interpolation method wrapping `LatentSpace.interpolate()` for geometry-aware dispatch. Supports `__call__(a, b, t)` for single-point interpolation, `between(traj_a, traj_b, t)` for pointwise trajectory interpolation, and `blend_sequence(trajectory, n_steps)` for trajectory densification. Pure numpy — no torch leakage. (#sprint-10)
- End-to-end demo script `scripts/end_to_end_lerp_demo.py` — two scenarios: (A) Euclidean lerp with two random 8D vectors → PCA to 2D, (B) spherical slerp on unit-norm vectors → PCA projection → lerp-vs-slerp path comparison → trajectory blending with `blend_sequence`. 1×2 matplotlib visualization with t-value annotations and sphere reference outline. (#sprint-10)
- Test suite: 28 Lerp tests covering construction with/without LatentSpace, correct interpolation (t=0→a, t=1→b, t=0.5→midpoint), geometry dispatch (slerp stays on sphere), trajectory between (shape, endpoints, error cases), and blend_sequence (densification, endpoint preservation, edge cases). (#sprint-10)
- **Giai đoạn 2 begins** — first Layer B method. Both validated ADRs (geometry-keyed `LatentSpace`, geometry-dispatch) exercised from the consumer side. `ModelAdapter` 3-mode ADR remains pending. (#sprint-10)

- Spherical geometry (`unit_norm`) as geometry case #2 for `LatentSpace` — validates the geometry-keyed and geometry-dispatch ADRs with real code. `geometry` moved from class-level constant to instance-level parameter, validated at construction. (#sprint-9)
- `LatentSpace.distance(a, b) -> float` — dispatches on `self.geometry`: Euclidean (`||a-b||`) or angular (`arccos`). (#sprint-9)
- `LatentSpace.interpolate(a, b, t) -> np.ndarray` — dispatches on `self.geometry`: lerp for Euclidean, proper slerp for spherical (geodesic on unit sphere) with edge-case handling for `sin(ω) ≈ 0`. (#sprint-9)
- `LatentSpace.normalize(point) -> np.ndarray` — euclidean returns copy; unit_norm projects to unit sphere. Zero vector raises for spherical. (#sprint-9)
- End-to-end demo script `scripts/end_to_end_spherical_demo.py` — synthetic unit-norm data → LatentSpace with unit_norm geometry → demonstrates validate_point, angular distance, slerp-vs-lerp, and normalization. 1×3 matplotlib visualization (3D scatter + lerp path + slerp path). (#sprint-9)
- Test suite: 35 new LatentSpace tests covering geometry construction, validate_point for unit_norm, distance (6 cases including known angles), interpolate/slerp (midpoint, endpoints, unit-norm invariance, edge cases), and normalize. (#sprint-9)
- **ADR milestone**: `LatentSpace` geometry-keyed ADR and geometry-dispatch ADR both moved from `pending` → `validated`. This is the last sprint of Giai đoạn 1 — core primitives confirmed with two geometry cases. (#sprint-9)

- RandomProjection adapter — ModelAdapter #2 (fixed-weight/stateless, pretrained pattern), pure numpy with Johnson-Lindenstrauss-style random Gaussian projection matrix, `encode`/`decode`/`latent_space`, no `fit` method. (#sprint-8)
- Internal `_ModelAdapterBase` ABC sketching the shared ModelAdapter shape (`encode`/`decode`/`latent_space`), marked UNSTABLE and not part of the public surface. `fit` deliberately excluded (VAE-specific). (#sprint-8)
- End-to-end demo script `scripts/end_to_end_random_projection_demo.py` — synthetic cluster data → RandomProjection encode → LatentSpace → Trajectory → PCA/UMAP → 1×3 matplotlib visualization (PCA original, PCA latent, UMAP latent) demonstrating JL distance preservation. (#sprint-8)
- Test suite: 24 RandomProjection tests covering construction, projection matrix normalisation, encode/decode shapes and error cases, roundtrip, reproducibility (same/different seeds), and approximate Johnson-Lindenstrauss distance preservation. (#sprint-8)
- VAE (Variational Autoencoder) adapter — ModelAdapter #1 (mode i: explicit learned latent), torch-based with `encode`/`decode`/`latent_space` and all-numpy public surface. Trains from scratch on [0,1]-scaled data with MSE + KL divergence loss. (#sprint-7)
- `adapters` sub-package (`src/latent_anything/adapters/`) — namespace for model adapter implementations, sibling to `methods/`. Not exported from top-level `__init__.py`. (#sprint-7)
- End-to-end demo script `scripts/end_to_end_vae_demo.py` — synthetic cluster data → VAE training → encode → LatentSpace → Trajectory → PCA + UMAP → 2×2 matplotlib visualization (original, reconstruction, PCA latent, UMAP latent). First pipeline exercising ALL Layer A primitives together. (#sprint-7)
- Test suite: 23 VAE tests covering construction, `hidden_dim` heuristic, `latent_space` property, fit/loss, encode/decode shape invariants, error cases, roundtrip reconstruction, and `random_state` reproducibility. (#sprint-7)
- SAE (Sparse Autoencoder) dimensionality reduction method — Method #3 with fundamentally different philosophy (gradient-descent training, L1 sparsity, encoder/decoder architecture), torch-based with all-numpy public surface. (#sprint-6)
- `Method` Protocol — frozen structural `typing.Protocol` defining `fit(data: np.ndarray) -> None` / `transform(data: np.ndarray) -> np.ndarray`, promoted to public surface as third core primitive. (#sprint-6)
- End-to-end demo script `scripts/end_to_end_sae_demo.py` — synthetic latent → Trajectory → SAE training → sparse feature projection → side-by-side PCA vs UMAP vs SAE 2D visualization with matplotlib. (#sprint-6)
- Test suite: 16 SAE tests covering construction, fit, transform, sparsity verification, `random_state` reproducibility, error cases, and Protocol conformance smoke tests. (#sprint-6)
- UMAP dimensionality reduction method (`fit`/`transform`/`fit_transform`) wrapping `umap-learn` with numpy public surface, stochastic and stateful. (#sprint-5)
- Internal `_MethodBase` base class sketching the shared shape for stateful methods (`fit`/`transform`/`fit_transform`), marked UNSTABLE and not part of the public API. (#sprint-5)
- PCA migrated to `_MethodBase` inheritance, removing the now-redundant `fit_transform` override. (#sprint-5)
- End-to-end demo script `scripts/end_to_end_umap_demo.py` — synthetic latent → Trajectory → UMAP fit → 2D projection → side-by-side matplotlib visualization with PCA comparison. (#sprint-5)
- Test suite: 12 UMAP tests covering construction, fit/transform shape invariants, fit_transform roundtrip, `random_state` reproducibility, and input validation errors. (#sprint-5)

- `_MethodBase` docstring updated from "UNSTABLE — will be replaced" to "Internal convenience base backed by the frozen `Method` Protocol" following Rule of Three (Method #3 freeze trigger). (#sprint-6)
- PCA and UMAP docstrings updated to note conformance to the frozen `Method` Protocol. (#sprint-6)
- `LatentSpace` concrete class for Euclidean flat vector spaces with `dim`, `geometry`, `source_model`, metadata, and `validate_point()`. (#sprint-4)
- `Trajectory` immutable sequence class holding 2D numpy latent arrays with `len`, indexing/slice (returns new `Trajectory`), and `to_numpy()`. (#sprint-4)
- PCA dimensionality reduction method (`fit`/`transform`/`fit_transform`) wrapping scikit-learn with numpy public surface. (#sprint-4)
- End-to-end demo script `scripts/end_to_end_pca_demo.py` — synthetic latent → Trajectory → PCA → 2D matplotlib visualization. (#sprint-4)
- Test suite: 41 tests covering LatentSpace (11), Trajectory with hypothesis property-based tests (16), PCA fit/transform/roundtrip (12), and package smoke (2). (#sprint-4)
- Package scaffold: `src/latent_anything/` (src-layout) with root-level `pyproject.toml` and PEP 621 metadata.
- Development tooling: `ruff` (lint + format), `pyright` (strict type-checking), `pytest`, and `hypothesis` configured in `pyproject.toml`.
- Smoke test suite: `tests/test_latent_anything/test_package.py` verifying import and version.
- CI workflow (`.github/workflows/ci.yml`) running `ruff check`, `ruff format --check`, `pyright`, and `pytest` on Python 3.12, 3.13, and 3.14.
- README updated with package installation via `uv`, Quick Start example, and current project structure.
- Python version range adjusted: `requires-python` set to `>=3.12,<3.15`, local development targets Python 3.13.
- Deploy workflow restricted: `deploy-latent-anything-theory.yml` now triggers only on `theory-v*` tag pushes or manual `workflow_dispatch`.

### Fixed

- Made release-note extraction fail when a matching changelog section contains only a version heading and no release body content.
- Prevented `AnalysisPipeline` cache entries from crossing between adapters that share hyperparameters but have different learned or randomly initialized state.
- Ensured cached analysis runs always fit the current stateful Layer A method instead of returning a transformed array while leaving the method unfitted.
- Narrowed the frozen `BMethod.apply_trajectory` invariant to the shared trajectory argument while preserving method-specific optional arguments on concrete implementations.
- Restored strict Pyright coverage for every Python file changed in Sprints 17–25.
- Expanded the full test suite to 596 passing tests with cache identity and state-consistency regressions.
