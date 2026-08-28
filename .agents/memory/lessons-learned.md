# Lessons Learned

A log of past bugs, edge cases, and environment-specific quirks discovered during development.

**Purpose:** Prevents the agent from repeating the same mistakes. Before implementing anything non-trivial, scan this file for related gotchas. When a new bug or unexpected behavior is resolved, record it here immediately.

**Format for each entry:**

```
## [YYYY-MM-DD] <Short title of the issue>

**Symptom:** What went wrong or behaved unexpectedly.
**Root cause:** Why it happened.
**Fix / workaround:** What resolved it.
**Watch out for:** Conditions that would trigger this again.
```

---

## [2026-06-20] Untracked files left behind after commit

**Symptom:** After committing Sprint 14 changes, `git status` showed an untracked `artifacts/hidden_state_demo_plot.png` that was never staged or committed.

**Root cause:** The demo plot file was generated during the sprint but never added to the staging area. The `git commit` only included files explicitly staged via `git add <file>...`, and the PNG was inadvertently skipped because it was treated as a "build artifact" rather than a deliverable that should be versioned alongside the demo script.

**Fix / workaround:** Stage all files — including generated outputs like demo plots — with `git add <path>` and then `git commit --amend --no-edit` to incorporate them into the existing commit. No file should remain untracked after a commit; every untracked file is either (a) committed, (b) added to `.gitignore`, or (c) deleted.

**Watch out for:** After running any script that generates an output file (plot, report, artifact), check `git status` before committing. Demo plots in `artifacts/` are tracked alongside their scripts, not ignored.

**Symptom:** What went wrong or behaved unexpectedly.
**Root cause:** Why it happened.
**Fix / workaround:** What resolved it.
**Watch out for:** Conditions that trigger this again.
```

---

<!-- Add new entries below, newest at the bottom -->

## [2026-08-03] Element-wise pose averaging leaves SO(3)

**Symptom:** Averaging corresponding homogeneous-matrix elements produced a rotation orthogonality error of `0.7071` in the controlled pose benchmark.
**Root cause:** SO(3) is a nonlinear group; its matrix entries cannot be interpolated independently.
**Fix / workaround:** Interpolate rotations with `R_a exp(t log(R_a^-1 R_b))` and translations in their declared metre coordinates.
**Watch out for:** Any future robot pose interpolation or smoothing code that treats 3x3 rotation matrices as ordinary Euclidean vectors.

## [2026-06-10] PowerShell here-string `@'...'@` leaves literal `@` lines when run via the Bash tool

**Symptom:** A `git commit` made through the **Bash** tool produced a malformed message — a lone `@` as the subject line and a trailing `@`, e.g. `git log -1 --format=%B` showed `@\ndocs(theory): ...\n...\n@`. The intended subject got pushed to line 2.
**Root cause:** The command used PowerShell here-string syntax `git commit -m @'...'@`. Bash does not understand `@'...'@`; it parses `@` as a literal char, `'...'` as an ordinary single-quoted string, and the closing `@` as another literal. So the `@` delimiters end up *inside* the message. The two tools have different multiline-string syntax and the wrong one was used for the wrong shell.
**Fix / workaround:** Re-ran with `git commit --amend -F - <<'EOF' ... EOF` (a real bash here-doc piped to `-F -`). For the Bash tool, pass multiline commit messages via a `<<'EOF'` here-doc to `-F -`, never `@'...'@`. Reserve `@'...'@` for the **PowerShell** tool only.
**Watch out for:** Mixing shells. This environment exposes both a Bash tool and a PowerShell tool; the project's CLAUDE.md shows PowerShell examples, so `@'...'@` is easy to reach for by reflex. Before sending a multiline string, match the here-string/here-doc syntax to the tool actually being invoked — `@'...'@` (PowerShell) vs `<<'EOF'` (bash). Always verify the result with `git log -1 --format=%B` after committing.

## [2026-06-17] `MPLBACKEND=Agg` drops inline notebook figures during nbconvert execution

**Symptom:** MkDocs pages rendered from executed notebooks showed code cells but not matplotlib images produced by `plt.show()`.
**Root cause:** The GitHub Pages deploy workflow set `MPLBACKEND=Agg` while executing notebooks with `jupyter nbconvert --execute`. That made matplotlib use a non-inline backend, so cells completed and produced outputs, but no `image/png` display data was written into the notebook.
**Fix / workaround:** Do not set `MPLBACKEND=Agg` for notebook execution. Let ipykernel use the inline matplotlib backend, then verify executed notebooks with `plt.show()` contain at least one `image/png` output before running `mkdocs build`.
**Watch out for:** Any headless CI workflow that executes notebooks before publishing docs. Avoid global matplotlib backend overrides unless a test confirms the executed `.ipynb` still contains inline image outputs.

## [2026-06-17] Sprint plan file left untracked after sprint completion

**Symptom:** The user asked "why don't you update sprint and commit?" — the sprint plan file `docs/sprint-plans/sprint-9.md` existed on disk but was never staged or committed, despite all 15 tasks being completed and the code already committed.

**Root cause:** The sprint plan file was treated as planning scaffolding — written upfront, read during implementation, but mentally excluded from the "source code" that needs committing. The user expected the tracking document (tasks marked `[x]`) to be committed as part of the sprint artifact, alongside the code, changelog, and decisions.md updates.

**Fix / workaround:** After amending, the sprint plan is now committed. Procedure for future sprint completions:
1. Update all `[ ]` → `[x]` in the sprint plan file.
2. Stage and commit it together with the code, changelog, ADR updates, and artifact summary in the same commit (or amend if already committed).
3. Verify with `git status` that the sprint plan file is no longer listed as untracked.

**Watch out for:** Any sprint where the plan file lives in `docs/sprint-plans/` as a `.md` file. It's easy to mentally categorize it as "documentation notes" that don't need committing, but the sprint plan is a deliverable tracking artifact that belongs in version control alongside the changes it describes.

## [2026-06-17] Slerp division by sin(ω) fails at ω ≈ π as well as ω ≈ 0

**Symptom:** `test_unit_norm_slerp_opposite_vectors` failed — `interpolate(a, b, 0.5)` for antipodal unit vectors returned `[0, 0, 0]` but the test expected `[1, 0, 0]`. The actual failure was a division by near-zero `sin(ω)` producing NaN, then silently falling through.

**Root cause:** The initial slerp implementation guarded against `ω ≈ 0` (`if abs(omega) < 1e-10: return a.copy()`) but not against `ω ≈ π`. For antipodal vectors, `a·b ≈ -1`, so `ω ≈ π` and `sin(ω) ≈ 0`, causing division by near-zero in the slerp formula. The guard checked the wrong condition.

**Fix / workaround:** Changed the edge-case guard from `abs(omega) < 1e-10` to `sin_omega < 1e-10`, which catches both `ω ≈ 0` and `ω ≈ π` in one check. When degenerate, falls back to lerp `(1-t)*a + t*b` rather than returning one endpoint.

**Watch out for:** Any geometric formula dividing by `sin(ω)` where ω comes from `arccos`. The zero-crossing happens at both ends of the cosine range (ω=0 and ω=π) because both make sin(ω)=0. Always guard on `sin_omega`, not on `omega` directly.

## [2026-06-20] Showcase scripts can fail review even when pytest is green

**Symptom:** Sprint 13's showcase work looked healthy because `pytest` passed, but the latent-anything review still failed on hundreds of strict `pyright` errors and the trajectory demo was not actually exercising the public `ActivationPatch.apply_trajectory()` API it claimed to use.
**Root cause:** The showcase script and tests were written as lightweight local artifacts with raw `dict`/`tuple` annotations and dynamic imports, so strict typing degraded into `Unknown` across the whole file. Separately, the trajectory panel bypassed the public patch API by reading the private `_delta` field directly, which let tests stay green without validating the intended contract.
**Fix / workaround:** Give even local `scripts/` artifacts explicit `TypedDict` payloads when they are reviewed with `pyright`, and add typed fixtures/protocol casts in tests so helper imports do not collapse to `Unknown`. For trajectory-level patching, call `ActivationPatch.apply_trajectory()` directly and add an assertion that the showcase panel output matches that public API.
**Watch out for:** Any future showcase/demo/research-style script that gets pulled into the repo's formal review gate. If it is linted/type-checked like product code, treat its config/result payloads as first-class typed structures and avoid reaching into private fields just because the code is "only a demo".

## [2026-06-21] `__len__` makes empty `Registry` falsy — `or` fallthrough to global singleton

**Symptom:** The convenience `register(kind, name, factory, registry=empty_registry)` function silently added entries to `GLOBAL_REGISTRY` instead of the explicitly-passed `empty_registry`, causing tests to fail and the global registry to accumulate test entries.

**Root cause:** `Registry` implements `__len__`, which Python uses as the fallback truthiness check. An empty registry (len=0) is falsy, so `registry or GLOBAL_REGISTRY` evaluated to `GLOBAL_REGISTRY` when the target registry was empty, even though `registry` was not `None`.

**Fix / workaround:** Two changes: (1) Added `__bool__` method to `Registry` that always returns `True`, overriding Python's default `__len__`-based truthiness. (2) Changed the `register` helper from `registry or GLOBAL_REGISTRY` to `registry if registry is not None else GLOBAL_REGISTRY` for defense-in-depth.

**Watch out for:** Any class that implements `__len__` and is used with `or` for default-value logic. `__len__` makes `bool(obj)` return `False` when `len(obj) == 0`. Always use explicit `is not None` checks with such classes.

## [2026-07-08] Runtime counters accidentally invalidate cache keys

**Symptom:** Sprint 23 cache tests showed repeated identical `AnalysisPipeline.run()` calls missing the cache and recomputing adapter encode/method output.
**Root cause:** The first `hash_component_config()` implementation hashed every public attribute that did not start with `_` or end with `_`. Test doubles used public `encode_calls`, `fit_calls`, and `transform_calls` counters, so the component config hash changed after each call even though construction config was unchanged.
**Fix / workaround:** Exclude obvious runtime counter fields ending in `_calls` from the config hash, while still excluding private state and fitted artifacts.
**Watch out for:** Any future cache-key logic that derives config from `vars(component)`. Separate construction/config fields from mutable runtime bookkeeping; otherwise cache hits turn into misses after the first call.

## [2026-07-09] Equal component configs can hide different model behavior

**Symptom:** Two `AnalysisPipeline` instances with identical adapter hyperparameters but different random weights shared an encode cache entry, so the second pipeline returned latents produced by the first adapter.
**Root cause:** `CacheKey` hashed only public construction fields and deliberately excluded fitted/random state such as projection matrices and learned weights.
**Fix / workaround:** Add a stable component-state hash to encode cache keys, including numpy arrays, nested object state, and tensor-backed `state_dict` values while excluding runtime call counters.
**Watch out for:** Any cache operation whose output depends on learned, initialized, loaded, or mutated component state. Configuration equality does not imply behavioral identity.

## [2026-07-09] Cached fit-transform output can leave a fresh method unfitted

**Symptom:** A fresh `AnalysisPipeline` sharing a populated cache returned a transformed result, but its own PCA remained unfitted and raised on the next `transform()` call.
**Root cause:** The fit-transform cache stored only the output array, not the state learned during `fit()`, so a cache hit skipped the state transition required by the method contract.
**Fix / workaround:** Cache adapter encode outputs only and always execute Layer A fit-transform on the current method instance. Do not cache a state-producing operation unless its state can also be restored coherently.
**Watch out for:** Caching any operation that both returns a value and mutates reusable component state; output parity alone is not enough.

## [2026-07-10] Dataclass script imports fail when importlib modules are not registered

**Symptom:** A pytest module that loaded `scripts/extract_release_notes.py` with `importlib.util.module_from_spec()` failed during collection at the `@dataclass` decorator with `AttributeError: 'NoneType' object has no attribute '__dict__'`.
**Root cause:** The manually created module was executed without first registering it in `sys.modules`. During dataclass processing, Python looks up `sys.modules[cls.__module__]`; because the module name was missing, the lookup returned `None`.
**Fix / workaround:** Insert the module before execution: `sys.modules[spec.name] = module`, then call `spec.loader.exec_module(module)`.
**Watch out for:** Tests that import standalone scripts via `importlib.util.spec_from_file_location()` and those scripts define dataclasses or other decorators that inspect the module namespace during import.

## [2026-07-16] pyright strict mode catches type annotation gaps in new feature code

**Symptom:** CI failed with 70+ pyright strict-mode errors after feature commits (K-means clustering, TCAV, MLP probe, LinearProbe). Errors spanned source, tests, and standalone scripts — not just the new code.

**Root cause:** Feature commits focused on functionality and tests but did not update type annotations for pyright strict mode (`reportUnknownVariableType`, `reportUnknownArgumentType`, `reportUnknownMemberType`, `reportPrivateUsage`). Common patterns: missing annotations on `list` variables (pyright infers `list[Unknown]`), `field(default_factory=dict)` inferred as `dict[Unknown, Unknown]`, redundant `None` checks on already-narrowed types, `dict[str, X]` passed to `dict[str | int, X]` parameters (dict key invariance), private function access in tests, and matplotlib/numpy type stub gaps in standalone scripts.

**Fix / workaround:** Targeted changes across 10 files:
- Add `# type: ignore[error-code]` for known pyright limitations (numpy/matplotlib stubs in scripts, private function testing, `field(default_factory=...)` partial unknown types)
- Remove redundant `intervention is not None` after `need_intervention` type guard narrows the variable
- Change `dict[str | int, np.ndarray]` to `Mapping[str | int, np.ndarray]` to accept both `dict[str, ...]` and `dict[int, ...]` (Mapping key is invariant but the broader Union works in practice)
- Replace `type("FakeConfig", (), {...})()` with a real `FakeConfig` class with typed attributes
- Annotate list variables (`list[np.ndarray]`, `list[list[int]]`) to prevent `list[Unknown]`
- Fix `integration.make_request()` → direct `TransformerGenerationRequest()` calls

**Watch out for:** Every feature commit that adds new files or modifies existing ones must run `pyright` before CI. Pay special attention to: untyped `list`/`dict` variables in tests, `field(default_factory=...)` on typed dataclass fields, function signatures with missing parameter annotations, and any test that accesses private module members. Scripts under `scripts/` are included in pyright's strict check — they need the same treatment as source code (either annotate or pragma-ignore).

## [2026-08-01] Network-marked tests were not actually opt-in

**Symptom:** The default full pytest run attempted Diffusers and Transformers model integrations even though those tests were documented as network/offline-optional, causing failures when optional extras were absent.
**Root cause:** Registering the `network` marker in pytest configuration only labels tests; it does not skip them.
**Fix / workaround:** Added collection-time skipping for `network` items unless `LATENT_ANYTHING_RUN_NETWORK=1` is set. The default CI suite now remains offline while the explicit integration lane still runs.
**Watch out for:** Any future optional integration test that relies on a marker being opt-in must be gated in `tests/conftest.py`, not merely decorated.

## [2026-08-02] Unnormalized SAE L1 penalty collapses every feature to dead

**Symptom:** The linear SAE produced `mean_l0 = 0` and all features dead for any `l1_coef >= 1e-4`, and with `l1_coef = 0` it failed to recover known sparse dictionary structure (L0 ≈ 3–4 instead of the true 1–2).
**Root cause:** The loss used `l1_coef * sum(|latent|)` over the whole batch, so the L1 gradient per element was `l1_coef` across all samples — orders of magnitude larger than the reconstruction gradient. Even a tiny coefficient starved every feature.
**Fix / workaround:** Normalize the penalty per element: `l1_coef * mean(|latent|)`. On standardized activations, effective sparsity then needs `l1_coef` around 0.1–0.5 (retune the default; 1e-3 is now effectively pure reconstruction). Standardizing activations before fitting also dramatically improves dictionary recovery (decoder alignment 0.49 → 0.98).
**Watch out for:** Any SAE/dictionary-learning loss that sums raw activations without normalizing by batch/element count. Also: when asserting "recovered features", compare decoder columns to source dictionary columns *after* undoing standardization (`decoder * std`), not in the raw fit space.

## [2026-08-02] Re-assigning an already-registered `nn.Module` submodule breaks `named_modules()`

**Symptom:** A tiny test transformer registered `self.transformer.h = self.blocks` where `self.blocks` was already a submodule of `self`. The forward worked, but `model.named_modules()` listed `blocks.0`, not `transformer.h.0`, so the hook seam ("transformer.h.{layer}") raised "Layer not found".
**Root cause:** A PyTorch module can have only one parent; once `self.blocks` was registered under `self`, re-assigning the same object under `self.transformer.h` did not re-parent it, and `named_modules()` only surfaced the first registration.
**Fix / workaround:** Register the `ModuleList` exactly once, under the seam name: build a typed container module (`transformer` with `h: nn.ModuleList`) and hold a typed attribute reference to it, so `named_modules()` yields `transformer.h.0` and pyright resolves `.h[0]`.
**Watch out for:** Any test fixture that mirrors a HuggingFace `transformer.h.{layer}` hook path. Do not reuse one ModuleList under two parent names.

## [2026-08-03] An unstable docstring does not make a premature Protocol concrete-first

**Symptom:** The M10 review still failed the Rule-of-Three gate after the renderer camera and backend Protocols were documented as unstable internal sketches.
**Root cause:** `INCREMENTAL.md` section 4a permits a tentative internal shared shape at two instances, but explicitly does not permit freezing that shape as a `Protocol`; the camera had one concrete implementation and the rasterizer backend had only two.
**Fix / workaround:** Remove the Protocols, annotate renderer methods with the existing concrete `GaussianCamera` through a `TYPE_CHECKING` import, and type the adapter against the concrete `GsplatBackend | ReferenceGaussianBackend` union.
**Watch out for:** Any first or second implementation where adding an "unstable" marker is mistaken for satisfying the Rule of Three. Keep concrete types until a third genuinely differing implementation exists.

## [2026-08-04] LeRobot falls back from TorchCodec to PyAV on this Windows environment

**Symptom:** A real `LeRobotDataset` episode smoke emitted a long TorchCodec DLL load traceback before successfully reading the sample.
**Root cause:** The installed CPU PyTorch/TorchCodec combination does not have loadable FFmpeg-backed TorchCodec DLLs on Windows.
**Fix / workaround:** No bridge change was needed; LeRobot selected its supported PyAV fallback automatically. Keep video-disabled metadata/state/action smoke tests independent of TorchCodec, and treat a successful PyAV fallback as an environment note rather than a bridge failure.
**Watch out for:** Windows runs that instantiate video-backed LeRobot datasets with TorchCodec installed but without matching FFmpeg shared libraries; avoid interpreting the warning as evidence that the canonical dataset API is unavailable.

## [2026-08-04] Optional LeRobot smoke can fail in a base environment with a newer NumPy

**Symptom:** The existing LeRobot factory smoke failed during the default full pytest run because the workspace had LeRobot 0.6.1 installed alongside NumPy 2.4.6, outside the bridge's declared `<2.3` compatibility window.
**Root cause:** Optional packages can remain installed in the base environment, while `uv run` without `--extra lerobot` resolves the base NumPy profile. The test checked only that LeRobot was importable, not that the complete optional runtime was compatible.
**Fix / workaround:** The smoke now skips with the compatibility diagnostic when the installed optional runtime is outside the supported window; the strict bridge loader still raises, and the optional CI lane runs the test in `uv sync --locked --extra lerobot`, which resolves NumPy 2.2.6.
**Watch out for:** Any optional integration test guarded only by `importorskip` can still fail when a stale/incompatible extra is installed locally. Check the bridge compatibility report before invoking upstream factories.


## [2026-08-06] SmolVLA fixed seed noise must match the policy device and dtype, and the real postprocessor returns a tensor

**Symptom:** The GPU intervention lane failed twice on the CUDA server: first `RuntimeError: Expected all tensors to be on the same device, but got mat1 is on cpu, different from other tensors on cuda:0`, then `IndexError: too many indices for tensor of dimension 2` in a parity assertion.
**Root cause:** (1) The adapter created the fixed action-chunk noise as a plain CPU float64 tensor, but the real policy lives on CUDA and its float32 action expert requires the noise on the model device. (2) The test assumed the post-processor returns a mapping with an `action` key like the fixture; LeRobot's real `PolicyProcessorPipeline` returns the action tensor itself, and `np.asarray` on the raw result failed. (3) A direct-path parity comparison without resetting the action queue compared the first and second pops of the same chunk instead of two identical queries.
**Fix / workaround:** Place seed noise on the policy device (`_policy_device(policy)`) and cast it to float32; treat post-processor results as tensor-or-mapping through a tolerant helper; reset the policy queue before the direct parity query so both sides compute the same fresh chunk.
**Watch out for:** Any policy adapter that forwards a caller-provided noise tensor into `select_action`; CPU-only fixtures cannot expose device-mismatch bugs, so GPU lanes are the first real check. Also keep the direct-parity pattern (reset, then compare) identical to the offline fixture tests.

## [2026-08-06] `uv run` without `--extra` silently resyncs to the base profile

**Symptom:** After `uv sync --extra lerobot-smolvla` installed NumPy 2.2.6, a follow-up `uv run python -c ...` reported NumPy 2.4.6 and the LeRobot compatibility check failed.
**Root cause:** `uv run` auto-syncs the default (extra-less) profile when no `--extra` flag is passed, replacing the lerobot-resolved NumPy with the base profile's 2.4.6.
**Fix / workaround:** Always pass the same `--extra` flags to `uv run` as to `uv sync` when the command needs the optional runtime (e.g. `uv run --extra lerobot-smolvla python ...`).
**Watch out for:** Any command mixing `uv sync --extra <profile>` and plain `uv run`; the environment silently changes between the two invocations.

## [2026-08-07] LIBERO advances its initial-state index per reset — conditions must create a fresh environment per cell

**Symptom:** The causal benchmark's baseline was not bit-exact with the no-hook control on the real LIBERO environment (mean action deviation 0.35, success mismatches) even though both used the same seed and fixed noise.
**Root cause:** `LiberoEnv` selects its initial state from `init_state_id`, an env-instance counter incremented on every reset; the seed only drives the sim RNG. Reusing one vector environment across conditions silently moved each condition to a different initial state, so trajectories were not comparable.
**Fix / workaround:** The benchmark environment bundle now owns an `env_factory`; `run_episode` creates and closes a fresh vector environment per (seed, condition) cell so every cell starts from initial state 0 with the same seed.
**Watch out for:** Any rollout harness that resets one gym env multiple times expecting identical starts. Check whether the env consumes a per-instance state counter (LIBERO init states, some wrappers) before comparing trajectories across resets.

## [2026-08-07] `hf-libero` prompts on stdin at import when `~/.libero/config.yaml` is missing or stale

**Symptom:** Two remote failures: first a `FileNotFoundError` for a `.pruned_init` path under a deleted temporary clone, then `OSError: pytest: reading from stdin while output is captured` inside the statistical lane.
**Root cause:** `libero.libero.__init__` runs `input()` when `~/.libero/config.yaml` is absent, and the config records absolute asset paths. The remote-cuda-test workflow deletes its temporary clone, so the config written by a previous run pointed at a vanished venv; under pytest's captured stdin the same `input()` raises instead of accepting the default.
**Fix / workaround:** `build_libero_benchmark_environment` bootstraps `~/.libero/config.yaml` itself before any upstream import: it resolves the installed package paths via `importlib.util.find_spec` (without executing the module) and rewrites the config whenever the recorded `init_states` path no longer exists.
**Watch out for:** Any upstream package that writes absolute install paths into a user config at first run. Under ephemeral environments (temp clones, container rebuilds) such configs go stale; bootstrap them from the current install instead of relying on upstream's interactive init.

## [2026-08-07] LeRobot env factories key LIBERO suites by the suite name, not the registered env type

**Symptom:** `make_env(libero_config, n_envs=1)` returned `{"libero_spatial": {0: <vec env>}}`, and the harness looking for the `libero` key raised "environment factory returned no 'libero' suite".
**Root cause:** `create_libero_envs` keys its return mapping by the LIBERO suite (task) name (`libero_spatial`, `libero_10`, ...), while the registered env-config type is `libero`; `env_cfg.type` returns the choice name only for the config itself.
**Fix / workaround:** Resolve the suite key as `config.task` first, falling back to `config.env_type`.
**Watch out for:** Upstream factories that return suite/task-keyed mappings whose keys are not the registered type name; assert against the real factory output, not the config class name.

## [2026-08-09] SmolVLA query cadence follows the action queue, not chunk size

**Symptom:** The simulation benchmark reported queries at steps `(0, 4)` for `chunk_size=4`, while the policy executed fresh model passes at `(0, 2, 4)` with `n_action_steps=2`.
**Root cause:** The rollout loop inferred query boundaries from `chunk_size`, even though LeRobot's action queue can be configured with a smaller `n_action_steps` value.
**Fix / workaround:** Expose an explicit `model_query_executed` signal from the hooked adapter and use a separate action-expert execution probe on the no-hook path; drive sample capture, latency, query steps, and counts exclusively from that signal.
**Watch out for:** Any policy where the returned action queue length differs from the model's nominal chunk size; never infer execution cadence from rollout configuration when the upstream policy owns queue state.

## [2026-08-09] Windows schema-v1 artifact references use a legacy separator

**Symptom:** Existing schema-v1 run records written on Windows failed to load because their artifact references contained `artifacts\\<digest>` instead of the canonical `artifacts/<digest>`.
**Root cause:** The original serializer used `str(Path("artifacts") / digest)`, which is platform-specific; strict canonical validation correctly rejected the historical Windows spelling but did not migrate it.
**Fix / workaround:** During schema-v1 migration, normalize only the exact legacy path whose suffix matches the record's lowercase SHA-256 digest, then validate the canonical path as usual.
**Watch out for:** Do not replace separators generically during migration; malformed, traversal, mixed, or otherwise non-canonical paths must remain rejected.

## [2026-08-11] A recurrent transition can fit one-step error but still drift in imagination

**Symptom:** The RSSM-style transition improved teacher-forced one-step MSE on the partially observed benchmark, but its open-loop final error and interval coverage were worse than the deterministic and memoryless Gaussian baselines.
**Root cause:** Teacher forcing supplies the observed latent at every fit step, while rollout feeds the model's own sampled predictions back into the recurrent state; the compact model has no learned posterior encoder, free-bits objective, or long-horizon training loss to control that distribution shift.
**Fix / workaround:** Keep one-step, KL-proxy, calibration, and horizon-drift metrics separate; retain the negative result in the comparison artifact and do not promote recurrent state to real-model evidence.
**Watch out for:** Any future world-model benchmark that reports only teacher-forced reconstruction or one-step prediction; require masked open-loop rollout metrics and a failure analysis before claiming useful imagination.

## [2026-08-11] Trajectory metadata cannot be deep-copied through InMemoryCache

**Symptom:** Caching a rollout by passing its `Trajectory` directly to `InMemoryCache.set_object()` raised `TypeError: cannot pickle 'mappingproxy' object`.
**Root cause:** `Trajectory.metadata` is intentionally immutable via `MappingProxyType`, while the generic object cache uses `deepcopy`, which cannot pickle that proxy.
**Fix / workaround:** Rollout caching stores a plain dictionary containing the trajectory array and copied metadata, then reconstructs a fresh `Trajectory` on a cache hit.
**Watch out for:** Any future object cached through `InMemoryCache` that contains immutable proxy-backed metadata; cache a plain serialization payload or extend the cache with an explicit copy protocol.

## [2026-08-12] Token rollout boundaries must distinguish single frames from batches

**Symptom:** The first tokenized-world-model smoke test rejected a valid `(1, tokens_per_frame)` batch as an invalid single-frame input, and raw temporal images were initially checked against a six-dimensional shape.
**Root cause:** The public prediction path accepts both a one-dimensional transition state and a batched two-dimensional context, but the validator treated the `allow_single` flag as exactly one dimensional. The temporal image fixture is `(episodes, time, channels, height, width)`, not an extra-dimensional batch.
**Fix / workaround:** Let `allow_single` accept either one- or two-dimensional token contexts and only add a batch axis for a one-dimensional state; validate raw sequences as five-dimensional arrays and reshape only the episode/time prefix for tokenizer calls.
**Watch out for:** Any future token sampler or rollout adapter that receives both direct `step()` states and batched `predict_next()` contexts; keep the shape normalization at the public boundary and retain integer-ID validation after normalization.

## [2026-08-12] Pipeline metadata must participate in rollout cache identity

**Symptom:** Repeating identical initial states and actions with different provenance metadata returned the first run's metadata on a cache hit.
**Root cause:** The rollout cache key covered only numeric inputs and transition state even though metadata is part of the returned trajectory contract.
**Fix / workaround:** Include a stable hash of caller metadata in the cache key and add a regression test that requires separate entries for distinct episode metadata.
**Watch out for:** Any cached operation whose result includes provenance, sampling, or configuration metadata; every result-affecting input must be represented in the key.

## [2026-08-12] Keyword Protocol conformance requires matching parameter names

**Symptom:** Runtime-checkable adapter Protocol checks passed while generic `encode(data=...)` calls failed for structured adapters.
**Root cause:** Runtime Protocol checks inspect attribute presence, not parameter names or keyword compatibility.
**Fix / workaround:** Align public adapter parameter names with the frozen Protocol and test keyword calls for JEPA, VQ-VAE, and tokenized adapters.
**Watch out for:** New structural implementations of a public Protocol; verify both `isinstance` conformance and representative keyword-based calls.

## [2026-08-25] Random VQ codebooks can lock compact full-batch training into one code

**Symptom:** The pinned digits VQ-VAE benchmark mapped every encoded position to one code, producing perplexity `1.0`; unused codebook entries stayed dead across training and made downstream token dynamics look perfectly predictable for the wrong reason.
**Root cause:** Before the decoder learned a useful reconstruction signal, random codebook initialization placed nearly all initial encoder outputs in one Voronoi region. The straight-through reconstruction path updated the encoder and selected embedding, while never-selected embeddings received no reconstruction gradient, so the initial assignment became self-reinforcing on the compact full-batch lane.
**Fix / workaround:** Initialize the codebook deterministically from evenly spaced initial encoder outputs before the first optimizer step, then require measured perplexity greater than `1.0` and dead-code rate below `1.0` in both regression tests and evidence generation.
**Watch out for:** Any compact VQ/VQ-VAE benchmark that treats finite reconstruction loss or constant-token accuracy as sufficient evidence. Always gate code usage explicitly and evaluate downstream token models only after the fitted tokenizer passes the non-degenerate usage gate.

## [2026-08-25] Splitting one delegated implementation-validation loop weakens traceability

**Symptom:** Routine owner intervention during an explicitly delegated subagent workflow split coherent implementation, validation, and audit context across turns.
**Root cause:** The primary owner did not keep the delegated scope, approval boundary, and synthesis role distinct from the subagent's technical loop.
**Fix / workaround:** Keep one explicitly assigned subagent on the coherent implementation-validation-audit loop; limit owner intervention to required authorization, concrete findings, and final synthesis.
**Watch out for:** When `subagent-workflow` explicitly assigns one subagent, do not split routine technical work across agents or context windows unless a concrete blocker requires it.

## [2026-08-25] SQLite context managers must close connections on Windows

**Symptom:** A cross-process portable-artifact reproduction passed its child checks but temporary-directory cleanup failed with `WinError 32` because the SQLite database remained open.
**Root cause:** `sqlite3.Connection` context management commits or rolls back transactions but does not close the connection.
**Fix / workaround:** Wrap every cache operation in an explicit session context that commits/rolls back and always calls `connection.close()`; retain the cross-process cleanup test.
**Watch out for:** Any SQLite WAL/cache backend on Windows, especially subprocess tests that remove temporary databases immediately after a child exits.

## [2026-08-25] Arrow read-all can bypass downstream allocation limits

**Symptom:** Portable decoding enforced array limits only after Arrow had already materialized the complete table and manifest.
**Root cause:** A convenient `read_all()` call precedes schema, row, manifest, and payload validation, so limits applied in the domain decoder cannot protect the Arrow reader itself.
**Fix / workaround:** Bound input and manifest bytes before opening, inspect schema and record-batch counts first, read bounded batches, cap rows/rank, and reject malformed/object dtypes before NumPy restoration.
**Watch out for:** Any decoder that accepts untrusted binary containers; validate the container allocation boundary before converting rows into domain objects.

## [2026-08-25] Typed envelope trees must preserve immutable sequence contracts

**Symptom:** CEM/MPPI/profile tuples silently restored as mutable lists, and nested metadata remained mutable behind a top-level mapping proxy.
**Root cause:** Generic recursive serialization erased tuple markers and shallow wrappers were mistaken for deep immutability.
**Fix / workaround:** Encode explicit tuple markers and recursively freeze decoded mappings and sequences; add tests that assert declared sequence types and reject nested mutation.
**Watch out for:** Frozen dataclasses and `Mapping` annotations do not make nested containers immutable; test the exact behavior-affecting field types after restoration.

## [2026-08-25] Validate cache row size before loading SQLite BLOBs

**Symptom:** A tampered SQLite row could claim an oversized payload and be loaded before cache limits were checked.
**Root cause:** The cache selected the BLOB and digest in one query, applying bounds only on writes.
**Fix / workaround:** Read and validate the stored size/digest in a transaction before selecting the BLOB; delete malformed rows as misses and validate portable envelopes at the coherent seam.
**Watch out for:** SQLite corruption or local tampering is still untrusted input; bounded writes do not imply bounded reads.

## [2026-08-25] Settle blocking worker calls before async-stream cancellation

**Symptom:** Cancelling an async generator while a synchronous producer or
transition runs in `asyncio.to_thread` can leave the worker active while source
cleanup races with `next()`.
**Root cause:** Cancelling the awaiting task does not stop the underlying
executor thread, and closing a generator concurrently with its active `next`
call is unsafe.
**Fix / workaround:** Shield each bounded worker call, await it to settle when
cancellation arrives, discard its result, then close the source in the async
generator's `finally` block. Keep one in-flight chunk and document that
cancellation is observed at an await boundary.
**Watch out for:** Do not claim immediate thread interruption for arbitrary
CPU/blocking Python code; bound each operation and test source finalization and
worker completion explicitly.

## [2026-08-25] Async iterator setup and array conversion can bypass streaming bounds

**Symptom:** A streaming audit found that asynchronous iterator construction and
cleanup still ran on the event-loop thread, while action validation converted
arbitrary chunks before enforcing the row limit.
**Root cause:** The implementation dispatched `next()` but not `iter()` or
`close()`, and relied on `np.asarray()` to discover shape and row count. Custom
iterables or array-protocol objects can execute blocking code or allocate before
those checks run.
**Fix / workaround:** Settle iterator construction, `next()`, and cleanup in
worker calls; require exact NumPy chunks and inspect rank/width/rows before any
dtype conversion; require an explicit/reset transition-state contract.
**Watch out for:** Any async wrapper around synchronous user objects must move
setup and finalization—not only the main operation—off the event loop, and any
size bound must be checked before a conversion that can materialize input.

## [2026-08-25] W&B offline init reuses the active run unless reinit is explicit

**Symptom:** Starting a W&B offline child while its parent was active returned the parent provider run; finishing the child then made later parent artifact logging fail, and temporary cleanup could hit locked log files on Windows.
**Root cause:** W&B's default `init()` behavior reuses the active run, unlike the fake provider used by the original parity tests.
**Fix / workaround:** Request `reinit="create_new"` for new adapter runs, use the adapter-owned offline artifact mirror, and call `wandb.teardown()` in the real offline lane before cleanup assertions.
**Watch out for:** Any future W&B parent/child or offline integration test that relies only on a fake SDK; verify provider run IDs, active-run lifecycle, local artifact bytes, and Windows cleanup.

## [2026-08-25] `Path` objects bypass URI lexical validation on Windows

**Symptom:** An MLflow tracking root supplied as `Path(r"\\server\share")` was normalized into a non-empty-authority file URI, even though equivalent string and URI inputs were rejected.
**Root cause:** The `Path` branch skipped the raw lexical checks used by the string/URI branch, so UNC, encoded, device, and ADS syntax reached resolution.
**Fix / workaround:** Apply platform-independent lexical checks to the raw `Path` spelling before resolution, while allowing ordinary native drive-rooted Windows paths.
**Watch out for:** Any security-sensitive local-root validator accepting both `str` and `Path` must test both representations; `Path` normalization can hide the original separator and URI syntax.

## [2026-08-25] W&B resume can lose adapter provenance

**Symptom:** A provider run whose adapter identity field had been removed could be resumed with changed configuration.
**Root cause:** Resume validation treated absent identity as an old-provider compatibility case and checked mismatches only when a value was present.
**Fix / workaround:** Require a canonical 64-hex adapter identity in provider config and fail closed on missing or malformed provenance before constructing the resumed adapter state.
**Watch out for:** External resume must reject missing provenance; a provider-generated run ID alone is not proof that the run belongs to this recorder contract.

## [2026-08-25] W&B offline mode still needs loopback service sockets

**Symptom:** Denying every socket connection made the real offline W&B test hang during local service teardown.
**Root cause:** W&B offline uses loopback or IPC sockets for its local service even though it must not contact a remote provider.
**Fix / workaround:** Deny URL/HTTP and non-local socket destinations while allowing loopback/IPC, then assert no newly created threads remain after `teardown()`.
**Watch out for:** Offline network tests must distinguish provider-network destinations from the SDK’s local coordination sockets; a blanket socket denial can deadlock cleanup and produce a false failure.

## [2026-08-25] Provider provenance does not prove resume continuity

**Symptom:** A fake provider that ignored the requested resume ID but echoed the current canonical provenance was accepted as a resumed run with a new provider ID.
**Root cause:** Resume validation checked provenance but assumed the provider honored the requested ID.
**Fix / workaround:** Require exact returned-ID equality for MLflow and W&B; best-effort finish unexpected runs as failed and raise a contract error, including cleanup failures.
**Watch out for:** Every external resume test must simulate a provider that creates a fresh run while preserving provenance, then verify fail-closed cleanup and retry behavior.

## [2026-08-26] Real backend fidelity needs eval mode and independent RNG

**Symptom:** A revision-pinned Diffusers VAE comparison showed a mean-output mismatch and non-repeatable posterior samples even though direct and adapter code used the same checkpoint.
**Root cause:** The adapter backend remained in training mode, and the direct and adapter sample calls consumed one shared global Torch RNG stream in sequence.
**Fix / workaround:** Set the loaded backend to evaluation mode and compare seeded posterior samples using a fresh local generator per backend and repeat; keep mean and sample semantics explicit in the evidence harness.
**Watch out for:** Exact direct-vs-adapter fidelity requires matching model mode and RNG ownership. Do not relax tolerances or infer parity from sequential global-RNG calls.

## [2026-08-26] Repeated Torch evidence runs must reuse interop configuration

**Symptom:** A deterministic interpolation evidence test passed once but failed on its second in-process run when setting Torch inter-op threads.
**Root cause:** Torch rejects `set_num_interop_threads()` after parallel work has already started, even when the requested value is unchanged.
**Fix / workaround:** Set the bounded thread configuration once and tolerate only Torch's specific already-started error on repeat runs; propagate unrelated configuration failures.
**Watch out for:** Reproducibility tests that run multiple model lanes in one process must not assume global Torch thread setters are idempotent.

## [2026-08-26] GPT-2 immutable pin can be a near-match typo

**Symptom:** The remote CUDA smoke could reach the GPU preflight, but Hugging Face returned HTTP 404 `RevisionNotFoundError` for `gpt2@e7da7f221d5bf496a4811970ad59b19a5b3ff2a4`.
**Root cause:** The recorded 40-character revision was not an existing commit in the canonical `openai-community/gpt2` repository; it differed from the official commit by a short middle segment.
**Fix / workaround:** Use the explicit canonical model ID `openai-community/gpt2` and the immutable official commit `e7da7f221d5bf496a48136c0cd264e630fe9fcc8`, verified from the repository's commit page and MIT model card.
**Watch out for:** Never infer that a model pin is valid from a familiar short prefix. Resolve the exact model ID and full commit through the authoritative Hugging Face repository/API before updating source, tests, or evidence contracts; keep the failed attempt as historical evidence.

## [2026-08-27] Direct script invocation breaks sibling imports

**Symptom:** Running `uv run python scripts/m14_l02_geometry.py` failed before execution with `ModuleNotFoundError: No module named 'scripts'`.
**Root cause:** Python puts the script directory on `sys.path` for direct-file execution, not the repository root package context required by absolute `scripts.*` sibling imports.
**Fix / workaround:** Use the package-aware canonical command `uv run python -m scripts.m14_l02_geometry`; add a side-effect-free `--check` mode to validate imports and the exact plan before a real run.
**Watch out for:** Any repository script using absolute sibling-package imports will fail when launched by file path; document and test module execution instead of adding `sys.path` hacks.

## [2026-08-27] Remote capture wrappers must persist report bytes before cleanup

**Symptom:** Three remote L03 capture attempts reached the real runner but lost the report during wrapper normalization, transfer, or local marker parsing; the cleanup trap then removed the only remote copy.
**Root cause:** The wrapper emitted or deleted report data in the wrong order, parsed base64 marker lines as ordinary key/value metadata, and used system `python` even though only the uv-managed interpreter existed on the remote image.
**Fix / workaround:** Emit report bytes inside the remote command before the cleanup trap, persist the raw local transcript before parsing, parse base64 begin/end markers before key/value metadata, use `uv run python` rather than system `python`, and rehearse the exact transcript parser before the canonical run. Preserve each failed attempt as an unchanged capture-only record; sanitize and retain audit metadata after verified output capture.
**Watch out for:** Any remote evidence lane whose output is captured over SSH must make report persistence precede cleanup, keep raw bytes until digest verification, and distinguish wrapper/capture failures from runner metrics or focused network-test failures.

## [2026-08-27] Structured module outputs can hide the hooked activation

**Symptom:** A real GPT-2 intervention hook at `transformer.h.6` received a tuple and the Tensor-only capture seam raised a `TypeError`, causing the focused network lane to report 6 passed and 2 failed.
**Root cause:** PyTorch forward hooks observe the complete module return value; decoder transformer blocks place the primary hidden-state Tensor first and retain auxiliary cache or attention values that downstream code still needs.
**Fix / workaround:** Centralize extraction and reconstruction in a private helper supporting exact Tensor, plain tuple, and plain list outputs. Replace only position 0, preserve tuple auxiliary identities and list type without mutating the original, and reject mappings, empty/non-Tensor primaries, and custom containers where reconstruction is not provably exact. Migrate capture, Integrated Gradients, TCAV, and SAE intervention/observation hooks.
**Watch out for:** Any new hook consumer that returns a replacement must preserve the original structured container and its auxiliary fields; observe-only hooks should extract the primary Tensor. Keep cleanup, gradient flow, shape validation, and zero-strength identity covered for both Tensor and structured outputs.

## [2026-08-27] GPT-2 block hooks are offset from native hidden-state indices

**Symptom:** A real intervention at `transformer.h.6` changed no value at native hidden-state index 6, causing the network oracle to fail after structured hook-output handling was fixed.
**Root cause:** Hugging Face GPT-2 appends the input to each block before invoking it, so block `h.L` produces native `output_hidden_states[L + 1]`; native index 0 is the embedding output and the final index is post-`ln_f`.
**Fix / workaround:** Keep intervention layers as direct zero-based transformer block indices, document the distinction from native capture indices, and assert block 6's effect at native index 7.
**Watch out for:** Network tests that compare a block intervention against the same-numbered native tuple index are off by one; use a nonconstant direction for final-logit effect checks because an all-ones shift can be removed by LayerNorm.

## [2026-08-28] Shape-only logit-lens parity masked double normalization

**Symptom:** The final-layer logit-lens parity test passed while the runtime applied GPT-2's final `ln_f` twice to native hidden-state index 12.
**Root cause:** Hugging Face GPT-2 stores the terminal `output_hidden_states` entry after `ln_f`, but the test asserted only layer identity and logits shape; the fake backend also did not model the post-`ln_f` terminal state accurately.
**Fix / workaround:** Pass a private final-normalization control to the lens, apply `ln_f` only to pre-final native states, and compare the terminal lens logits numerically with the model's forward logits using an affine post-normalization fake.
**Watch out for:** Any native hidden-state parity test that checks only shapes or ranks can remain green under duplicate normalization; model-specific output-state ordering and exact value parity must be tested, including capture subsets containing the terminal state.

## [2026-08-28] Windows text writes changed the JSONL line-ending contract

**Symptom:** Temporary JSONL fixtures written with `Path.write_text()` were rejected by the offline checker as CRLF, even though the test string explicitly used `\n`.
**Root cause:** Windows newline translation converts text-mode `\n` writes to CRLF; the L04 fixture contract requires exact UTF-8 bytes with LF-only row terminators.
**Fix / workaround:** Write mutated JSONL fixtures with `Path.write_bytes(...encode("utf-8"))` and keep the checker validating raw bytes before parsing.
**Watch out for:** Any digest-sensitive JSONL or evidence fixture test running on Windows; text-mode writes alter bytes and invalidate content digests.

## [2026-08-28] PowerShell here-string CRLF corrupts remote Bash exit status

**Symptom:** A direct authenticated PowerShell `ssh.exe` invocation reported `exit: 1: numeric argument required` and SSH exit `2`, even though the embedded remote status was `1` for the genuine semantic Integrated Gradients failure.
**Root cause:** The PowerShell here-string transported CRLF line endings into the remote Bash script's status encoding, so Bash parsed the trailing carriage return as part of the numeric `exit` argument.
**Fix / workaround:** For future runs, normalize the remote script to LF bytes before piping it directly to authenticated PowerShell `ssh.exe`; keep Bash as the remote shell. The frozen L04 evidence was not altered and was not rerun.
**Watch out for:** Always distinguish the embedded remote status (`1`) from the transport-level SSH exit (`2`) when a Windows PowerShell here-string feeds Bash; normalize bytes before transport rather than changing frozen evidence or thresholds.

## [2026-08-28] Generic remote-test transport can conflict with the project contract

**Symptom:** The frozen L04 remote lane required a direct authenticated PowerShell `ssh.exe` transport, while the generic remote-test workflow suggested a different shell/connection path; early payloads also failed at remote setup or post-run bundling.
**Root cause:** Repository-specific transport requirements override stale generic guidance, and shell payloads were not syntax-checked in the exact byte form that would cross SSH. Bare system `python` and shell-specific bundling assumptions were unavailable on the remote image.
**Fix / workaround:** Use direct PowerShell `ssh.exe` with Bash as the remote shell, normalize scripts to LF bytes, run `bash -n` on the exact payload before SSH, and use the uv-bound interpreter or POSIX-native bundling tools rather than assuming system Python.
**Watch out for:** Any remote evidence lane with a repository transport override; validate the exact LF payload and interpreter/tool availability before consuming a model run.

## [2026-08-28] Transport, semantic, and envelope ordinals are different identities

**Symptom:** Recovered TCAV evidence required describing a third transport attempt, second semantic execution, and first semantic envelope without treating those numbers as interchangeable.
**Root cause:** Transport retries can fail before execution, while one semantic execution can emit one envelope; collapsing the ordinals obscures whether a model was actually rerun and which files came from that execution.
**Fix / workaround:** Record transport attempt, semantic execution ordinal, and semantic envelope attempt as separate fields and reconcile each with command, result, bundle, and exit/status singleton markers.
**Watch out for:** Any retry/recovery workflow where setup, model execution, and artifact transfer have independent failure boundaries.

## [2026-08-28] Remote raw capture must survive verified extraction

**Symptom:** TCAV attempt 1 deleted the raw capture before the bundle was verified, leaving only an audit and no recoverable envelopes; attempt 2 initially overstated cleanup despite having no cleanup marker.
**Root cause:** Cleanup was treated as a completion fact before raw bytes, size/hash, markers, and sanitized records had been verified.
**Fix / workaround:** Persist and hash raw bytes first, extract and validate envelopes, complete the sanitized audit, then delete only an exact size/hash-matched quarantine file and verify absence. If a cleanup marker is missing, record cleanup as unverified rather than pass.
**Watch out for:** Every capture wrapper and quarantine cleanup path; never delete the only raw evidence before verified extraction, and do not infer cleanup from the absence of an error.

## [2026-08-28] Marker multiplicity must match the emitter schema

**Symptom:** TCAV attempt 3 emitted duplicate preflight and ordinal markers; acceptance required a manual byte-identical-value review instead of treating the output as a clean singleton capture.
**Root cause:** The marker schema's expected multiplicity was not aligned with the approved emitter's output path.
**Fix / workaround:** Define singleton versus repeatable marker fields explicitly, validate counts and values against that schema, and retain a duplicate-marker defect audit when identical duplicates are accepted by owner review.
**Watch out for:** Any wrapper that combines preflight, execution, and bundle metadata; schema counts must be tested against the exact emitted transcript before a canonical run.

## [2026-08-29] Raw file hashes differ from canonical evidence digests

**Symptom:** The first DirectLogitLens transport stopped before preflight because a raw plan-file SHA-256 was compared with the plan's canonical object digest, making setup failure look like a plan mismatch.
**Root cause:** Raw file bytes (including formatting/line endings) and canonical JSON object bytes are different hash domains; transport and semantic/envelope ordinals also identify different retry boundaries.
**Fix / workaround:** Lint and simulate the exact guard before remote execution, compare each digest only within its declared domain, use collision-proof attempt-specific raw capture names, and transport the normalized LF payload through direct authenticated PowerShell `ssh.exe` with Bash.
**Watch out for:** Any evidence wrapper that hashes a plan or fixture before execution; preserve raw-file and canonical-object digests as separate fields and never infer semantic execution from a transport attempt.

## [2026-08-29] Native terminal diagnostics must not enter fitted-layer aggregation

**Symptom:** The first real TunedLogitLens execution failed with `tuned-lens macro metric requires exactly fitted native layers 0..11` even though evaluation returned the expected 13 native-layer mappings.
**Root cause:** The production aggregation boundary passed direct, tuned, and shuffled mappings containing diagnostic-only native layer 12 into a macro helper that intentionally accepts only fitted layers 0..11.
**Fix / workaround:** Filter every evaluator mapping to `FITTED_LAYERS` immediately before macro aggregation; retain native layer 12 only for terminal post-`ln_f` parity. Add an integration-level regression test using realistic 13-key evaluator outputs.
**Watch out for:** Any metric API that deliberately rejects extra keys; diagnostic/native terminal states must be separated from fitted translator acceptance inputs at the production boundary.

## [2026-08-29] Concrete CUDA device names are not backend markers

**Symptom:** A failed real CUDA attempt with device `NVIDIA GeForce RTX 4060 Ti` was classified as dispatcher-only because provenance logic compared the concrete device string to the literal `cuda`.
**Root cause:** Device identity and execution backend/attempt state are different provenance dimensions; an early failure can have a concrete device name without a successful execution result.
**Fix / workaround:** Carry explicit `execution_attempted` and `execution_backend` markers from the real dispatcher, classify real CUDA attempts from those markers, and retain D0 plus `resource_peak: "not measured"` on early failure. Validate marker coherence fail-closed.
**Watch out for:** Real, injected, and dispatcher-only paths sharing envelope builders; never infer execution origin from a human-readable GPU name or from an absent result payload.
