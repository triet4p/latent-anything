---
name: latent-anything-review
description: >
  Review Python source in THIS latent-anything repo to verify it is "good" and "compliant" before
  it lands — runs the real tooling gate (ruff, pyright strict, pytest), then judges the code against
  the project's coding rules (.agents/rules), the architecture ADRs (.agents/memory/decisions.md),
  and the incremental Rule-of-Three process (docs/INCREMENTAL.md). Use it whenever the user asks to
  review/check/verify new or changed `src` code, asks "is this good / compliant / safe to
  commit/merge", finishes a task or sprint increment, or is about to commit Python here — even
  without "review". Prefer it over a generic review for any latent-anything source: it
  catches torch leaking into the public API, premature abstraction (a Protocol/ABC before the 3rd
  impl), a mutable Trajectory, Euclidean-hardcoded interpolation, or fake-green tests that always
  pass / only exist to dodge the rules — things a generic reviewer often misses.
  Do NOT use it for reviewing docs/research notes/notebooks, code in another repo, or non-review
  tasks like writing research, planning sprints, or running a formatter.
---

# Latent-Anything Review

## Purpose

Give a single, trustworthy verdict on whether new source code is safe to keep: it both **proves**
the mechanical guarantees by *running* the tooling, and **judges** the project-specific guarantees
(architecture ADRs, the incremental Rule-of-Three, the language/git rules) that no linter can see.

The reason this skill exists separately from a generic code review: latent-anything has deliberate,
load-bearing decisions — `Trajectory` is immutable, the public API never leaks `torch`, interfaces
are *discovered* not designed, and three ADRs are *hypotheses to validate by code*. A generic
reviewer will happily approve code that quietly violates these. This reviewer's main value is
catching exactly those.

## Source of truth (read these live — they evolve)

The skill does **not** restate the rules; it points at the authoritative files so a review never
drifts from the current state of the project. At the start of a review, read:

- [.agents/rules/python.md](../../rules/python.md) — coding conventions, approved packages, typing, async.
- [.agents/rules/git.md](../../rules/git.md) and [.agents/rules/changelog.md](../../rules/changelog.md).
- [.agents/memory/decisions.md](../../memory/decisions.md) — the ADRs, including which are still `pending`.
- [docs/INCREMENTAL.md](../../../docs/INCREMENTAL.md) — the Rule of Three and the per-increment checklist.
- [references/checklist.md](references/checklist.md) — the condensed, review-specific checklist that
  turns all of the above into things you can actually look for, including the judgment calls.

If a markdown file changed, also skim [.agents/rules/markdown.md](../../rules/markdown.md).

## Workflow

### 1. Determine scope

Default to the uncommitted/branch changes: `git status` + `git diff main...HEAD` and `git diff`
(staged + unstaged). If the user named specific files or a directory, review exactly those instead.
List the files in scope back to the user in one line so the boundary is explicit.

### 2. Run the tooling gate (objective — actually run it)

Run from the package root (the directory with `pyproject.toml`). Use `uv`, never bare `pip`/`python`.
On any tool, scope to the changed files when possible, but run the full test suite.

- `uv run ruff check <scope>` — lint must be clean.
- `uv run ruff format --check <scope>` — formatting must already be applied.
- `uv run pyright <scope>` — strict mode, zero errors on public APIs.
- `uv run pytest` — all tests pass; note if the increment added no test for new behavior.

Capture the real output. A gate failure is **blocking** — do not soften it. If the tooling can't run
(no package yet, missing config), say so plainly rather than guessing a pass. A green `pytest` run is
necessary but not sufficient: changed tests still need an integrity check so the suite is not
"passing" on tautologies or rule-circumvention.

### 3. Static conformance review (judgment — what tools can't see)

Read the in-scope code and walk [references/checklist.md](references/checklist.md). The high-value,
project-specific checks, in priority order:

1. **Public surface purity** — no `torch.Tensor` (or other internal types) in any public signature;
   arrays are `numpy.ndarray`. Leaking torch is blocking (ADR / python rule).
2. **ADR conformance** — `LatentSpace` keyed on geometry (carrying a *metric*, not just shape);
   `ModelAdapter` not assuming a single VAE-style learned/invertible `decode`; metric-dependent
   `Trajectory` ops dispatch on geometry instead of hardcoding Euclidean lerp; `Trajectory`
   immutable (every op returns a new instance). Violations are blocking.
3. **Rule of Three** — the subtle one. Did this change introduce a `Protocol`/ABC/abstract base with
   fewer than three concrete *differing-philosophy* implementations? That is premature abstraction —
   the project builds concrete-first and extracts interfaces only at the third differing instance.
   Conversely, did a third differing instance land without the interface being extracted and all
   prior call-sites migrated? Both are findings. See the checklist for how to count and the
   exceptions.
4. **Test integrity / anti-cheat** — if tests changed, make sure they can actually fail when the
   behavior is wrong. Treat as **Blocking** any test that is effectively guaranteed to pass
   (`assert True`, a value compared to itself, over-broad mocks/stubs that make the assertion true by
   construction, swallowing the expected failure and then passing), or any test whose main purpose
   is to satisfy the gate while dodging a project rule/invariant. A fake-green test is worse than no
   test because it creates false confidence.
5. **ADR reconciliation** — if the code touches a `pending` ADR area, decisions.md must be updated
   (mark `validated`) or a new reversing/amending ADR appended. Silent divergence from an ADR is a
   finding.
6. **Rules hygiene** — approved packages only; full type annotations; `collections.abc` over `typing`
   aliases; `Protocol` over ABC when an interface *is* warranted; naming conventions; async-primary
   with a `run_sync` wrapper; changelog updated in the same change for user-visible behavior; no
   forbidden files staged (`.venv/`, `__pycache__/`, secrets); conventional-commit shape if a commit
   message is in scope.

Classify every finding as **Blocking** (breaks a guarantee) or **Advisory** (style/quality nit).
When unsure whether something is intentional, surface it as a finding with a question rather than
assuming — but cite the exact rule/ADR and `file:line`.

### 4. Produce the report

Use the template below verbatim. Lead with the verdict so it's the first thing seen.

**Verdict rules:**

- **FAIL** — any tooling-gate failure OR any Blocking finding.
- **PASS-WITH-WARNINGS** — gate clean and no Blocking findings, but ≥1 Advisory finding.
- **PASS** — gate clean and zero findings.

Every finding must be **actionable**: name the file:line, the rule/ADR it violates, and the concrete
fix. A finding the author can't act on is not worth printing.

## Report template

```
# Latent-Anything Review — <PASS | PASS-WITH-WARNINGS | FAIL>

**Scope:** <files/diff reviewed>

## Tooling gate
- ruff check:        <pass/fail — detail>
- ruff format:       <pass/fail>
- pyright (strict):  <pass/fail — detail>
- pytest:            <pass/fail — N passed, M failed>

## Blocking findings
1. <file:line> — <rule/ADR> — <what's wrong> → <fix>
   (omit this whole section if none)

## Advisory findings
1. <file:line> — <what could be better> → <suggestion>
   (omit if none)

## ADR / Rule-of-Three notes
- <e.g. "Introduced Method Protocol with 2 impls (PCA, UMAP) — premature; extract at the 3rd
  differing instance per INCREMENTAL.md §4a." or "Touches pending LatentSpace ADR; decisions.md not
  updated.">
- <e.g. "Confirms geometry-dispatch ADR — suggest marking it validated in decisions.md.">
```

## Notes

- This skill is **advisory**: it reports a verdict, it does not block commits or edit code. If the
  user wants fixes applied, that's a follow-up they ask for explicitly.
- Be specific and grounded; if you didn't run a tool, don't report it as passing. Honest "couldn't
  run X" beats a fabricated green check — the whole point of this skill is to be the trustworthy
  gate the project leans on.
