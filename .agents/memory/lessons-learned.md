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
