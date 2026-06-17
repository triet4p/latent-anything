# Review Checklist — what to look for

This is the condensed, review-specific checklist. It turns the rules and ADRs into things you can
actually look for in a diff. The authoritative sources are the live files (linked from SKILL.md);
this list exists so a review is fast and consistent, and so the judgment calls (especially Rule of
Three) have a concrete procedure.

Each item is tagged **[B]** Blocking or **[A]** Advisory. Blocking ⇒ FAIL.

## Table of contents
1. Tooling gate (objective)
2. Public surface purity
3. ADR conformance
4. Rule of Three (the judgment call)
5. Test integrity / anti-cheat
6. ADR reconciliation
7. Python rules hygiene
8. Git / changelog / files
9. Markdown (only if docs changed)

---

## 1. Tooling gate (objective — run, don't guess)

- **[B]** `uv run ruff check` clean on scope.
- **[B]** `uv run ruff format --check` shows no diff (formatting already applied).
- **[B]** `uv run pyright` (strict) zero errors — especially on public APIs.
- **[B]** `uv run pytest` all green.
- **[A]** New behavior added without a test. Core primitives (`LatentSpace`, `Trajectory`) should
  also get `hypothesis` property-based tests, not just example-based ones.

If the package/tooling doesn't exist yet, report "gate not runnable: <reason>" — that is not a pass.
A green `pytest` run only proves the suite is green, not that the tests are meaningful; changed
tests still need the integrity checks in section 5.

---

## 2. Public surface purity

The public API is the contract with plugin authors and is the hardest thing to change later, so it
gets the strictest eye.

- **[B]** Any public function/method parameter or return type is `torch.Tensor` (or another internal
  type). Public tensor/array types must be `numpy.ndarray`. `torch` is internal-only; it must not
  appear in a public signature. → Convert at the boundary; keep torch inside.
- **[B]** A non-approved package introduced for a concern already covered by the approved list
  (e.g. Hydra for config instead of `pydantic`; pandas where numpy suffices). → Use the approved one
  or log a decision first.
- **[A]** Bare `Any` in a public signature without a comment explaining why it's unavoidable.

---

## 3. ADR conformance

From [decisions.md](../../memory/decisions.md). These are deliberate and load-bearing — violating
one silently is the worst failure mode this skill guards against.

- **[B]** `LatentSpace` keyed on container **shape** instead of **geometry/manifold**, or its
  geometry hint carries only dimensionality and not the **metric**. The same handle must be able to
  represent a flat vector, a sequence/grid, an unordered Gaussian set, and hidden-state activations.
- **[B]** `ModelAdapter` assuming a single VAE-style world: `decode` treated as always a learned,
  invertible map. It must allow three modes — explicit learned latent, no-explicit-latent
  (activations *are* the latent), and explicit non-latent structured (decode = deterministic
  renderer). → Don't bake "decode is a learned decoder" into the interface.
- **[B]** Metric-dependent `Trajectory` ops (interpolate, compare, distance, average) hardcoding
  Euclidean lerp instead of dispatching on `LatentSpace.geometry`. Euclidean is one case, not the
  default; spherical needs slerp, anisotropic needs Mahalanobis, manifold needs log/exp-map, pose
  needs SO(3)/SE(3). → Dispatch on geometry.
- **[B]** A `Trajectory` operation that mutates in place instead of returning a new `Trajectory`.
  Immutability is a chosen invariant (easy to reason about, cache, parallelize).

Note: the geometry/adapter ADRs are dated as **hypotheses to validate in Giai đoạn 1**. Early code is
*allowed to be narrower* (e.g. only `euclidean`). Narrow ≠ violation. A violation is code that makes
the eventual general shape *impossible* (e.g. hardcoding Euclidean into the type, or shape-keying the
space) — not code that simply hasn't reached the general case yet.

---

## 4. Rule of Three (the judgment call)

From [INCREMENTAL.md §4](../../../docs/INCREMENTAL.md). This is the highest-value check and the one a
generic reviewer never makes. The project builds **concrete-first** and extracts interfaces only when
forced. Over-abstraction is as much a defect here as under-testing.

**How to count instances.** For any abstraction the diff introduces or touches (a `Method`, a
`ModelAdapter`, a geometry case, a `Trajectory` op), count the concrete implementations that exist
*after* this change — in the diff plus the existing tree. "Differing philosophy" means genuinely
different mechanics (e.g. stateless pure transform vs. stateful fit-from-data vs.
train-inside-forward-pass), not three near-clones.

- **[B]** A `Protocol` / ABC / abstract base / generic interface introduced with **< 3** concrete
  differing-philosophy implementations. → Keep it concrete. At 1 instance, hardcode (duplication is
  fine). At 2, a *tentative* shared shape is OK but it must be marked unstable in the docstring and
  stay off the public surface — not a frozen `Protocol`.
- **[A]** Exactly 2 instances and the author froze a public interface anyway. → Downgrade to an
  internal, docstring-"unstable" shape until the 3rd differing instance.
- **[B]** The 3rd differing-philosophy instance **did** land, an interface was extracted/changed, but
  prior call-sites were **not** all migrated in the same change. Half-migrated interfaces are the
  thing INCREMENTAL.md §4b explicitly forbids ("incremental is not careless"). → Migrate all in this
  commit or don't freeze yet.
- **[A]** The 3rd differing instance landed but no interface was extracted at all (under-abstraction
  / missed freeze point). → Consider extracting now.
- **[B]** Something promoted to the public plugin surface (`ModelAdapter` / `Method` / `Pipeline`)
  with fewer than **2 real use cases** needing it. The public surface stays minimal by design.

When flagging premature abstraction, always name the abstraction, list the concrete impls you counted,
and cite §4a — so the author sees the count, not just a verdict.

---

## 5. Test integrity / anti-cheat

Tests are part of the evidence for whether code is safe. A test that cannot fail for the intended
bug, or that exists mainly to game the gate, is worse than no test because it creates false
confidence.

- **[B]** A new/changed test is effectively guaranteed to pass regardless of whether the behavior is
  correct. Common smells: `assert True`, comparing a value to itself, asserting only that a mock
  returned the value the test itself configured, swallowing the expected exception and then passing,
  or verifying only truthiness / non-`None` when the real contract is more specific. → Replace it
  with assertions over observable behavior and a failure mode that would break if the code regressed.
- **[B]** A test was added mainly to satisfy the review/tooling gate while dodging a project rule or
  invariant. Examples: weakening assertions to avoid exposing a public-API contract leak, using
  over-broad mocks so the forbidden path is never exercised, or encoding today's buggy behavior as
  "the spec" without an explicit decision. → Fix the code/rule mismatch, or log the decision
  honestly; do not use tests to launder it.
- **[A]** A test mirrors the implementation step-for-step instead of asserting the external contract
  or invariant. It may still fail, but it is low-signal and brittle. → Prefer behavior/invariant
  assertions that survive refactors.

When flagging a meaningless test, explain *why it would still pass if the real behavior were broken*.

---

## 6. ADR reconciliation

From [INCREMENTAL.md §4c](../../../docs/INCREMENTAL.md). Code is how the pending ADRs get validated or
overturned; the log must stay coupled to what the code proves.

- **[A]** The change exercises a `pending` ADR area (geometry keying, adapter modes, geometry
  dispatch) and **confirms** it, but decisions.md wasn't updated to note it `validated`. → Suggest
  the one-line update.
- **[B]** The change **contradicts** a pending or accepted ADR and no new reversing/amending ADR was
  appended. Silent divergence is not allowed — the contradiction must become a deliberate, logged
  decision (append-only; don't edit the old entry). → Use the `log-decision` skill.

---

## 7. Python rules hygiene

From [python.md](../../rules/python.md). Mostly **[A]** unless it breaks a contract.

- **[A]** Function signatures not fully annotated (params + return).
- **[A]** `typing.Sequence/Mapping/Callable` used instead of the `collections.abc` equivalents.
- **[A]** ABC used where a `Protocol` would fit (Protocols are lighter, structural, cross-language).
- **[A]** Naming off-convention: modules/funcs/vars `snake_case`, classes `PascalCase`, constants
  `UPPER_SNAKE_CASE`, private leading `_`.
- **[A]** An I/O-bound public op exposed only as blocking, with no `async def` + `run_sync` wrapper
  (the API is async-primary). Blocking calls inside `async def` without `asyncio.to_thread`.
- **[A]** A standalone utility script not using PEP 723 inline metadata (`# /// script`).
- **[A]** Test files not mirroring `src/` layout or not named `test_<module>.py`; test names that
  don't read as a sentence.

---

## 8. Git / changelog / files

- **[B]** A user-visible change (new feature, changed behavior, bug fix) with no `CHANGELOG.md`
  `[Unreleased]` entry in the same change. Internal refactors / test-only / docs-only are exempt.
- **[A]** Commit message (if in scope) not Conventional Commits shape `<type>(<scope>): <desc>`,
  lowercase imperative, no trailing period.
- **[B]** Forbidden artifacts staged: `.venv/`, `__pycache__/`, `*.pyc`, `.env`/secrets, large
  binaries, build outputs. → Unstage and add to `.gitignore`.
- **[A]** Evidence of `git add .` / `git add -A` (unrelated files swept in). Stage by name.
- **[A]** One commit mixing unrelated logical changes (e.g. a fix + a refactor).

---

## 9. Markdown (only if docs / research notes / notebooks changed)

From [markdown.md](../../rules/markdown.md). The site is Python-Markdown (stricter than GitHub).

- **[A]** A list, table, or `$$…$$` block with no blank line before it (Python-Markdown glues it into
  the preceding paragraph).
- **[A]** `$$…$$` inside a list item not indented 4 spaces / not blank-line-surrounded.
- **[A]** Notebook markdown-cell LaTeX using double backslash `$\\Sigma$` (MathJax reads it as a line
  break) — should be single backslash.
- **[A]** New research note missing the `TL;DR → body → Liên quan → Tham khảo` structure, or a new
  page/notebook not added to `mkdocs.yml` nav.
