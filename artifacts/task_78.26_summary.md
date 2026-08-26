# Sprint 78 Atomic Task 78.26 — Documentation and Release-Readiness Conflict Inventory

## Scope and source-of-truth order

This is a read-only audit. No source, tests, `pyproject.toml`, `CHANGELOG.md`, or
documentation content was changed. The only task-scoped writes are this artifact
and the audit-only checkbox in `docs/sprint-plans/sprint-78.md`.

The audit read `docs/INDEX.md` and `docs/LANGUAGE.md` first, then every document
listed by the index in order (with no nested `docs/INDEX.md`), followed by the
root README/changelog/build configuration, all root-level release/API/evidence
documents, Sprints 78–80 and relevant prior sprint records, memory decisions and
lessons, examples/plugin links, and the M14 ledger. Graphify query/explain were
run before the inventory; graphify is updated after this artifact.

For current claims the precedence is: machine-readable
`docs/evidence-ledger.json` and its validator, then
`docs/M14_REAL_SYSTEM_VALIDATION.md`, then `docs/EVIDENCE_LEDGER.md`/`docs/PLAN.md`,
then current README/integration guides. Sprint plans and dated beta artifacts are
historical/planning records, not evidence promotion by themselves.

## Deterministic results

| Check | Result |
|---|---|
| Top-level exports | **PASS** — `src/latent_anything/__init__.py` and the API snapshot each contain 202 unique names. |
| Built-in registry | **PASS** — 32 `GLOBAL_REGISTRY.register` calls in `src/latent_anything/_plugin_builtins.py`; M14 lists the same 32 names. |
| Optional profiles | **PASS** — 12 `[project.optional-dependencies]` keys in `pyproject.toml`, matching M14. |
| Plugin groups | **PASS** — five groups in `docs/PLUGIN_AUTHOR_GUIDE.md` and M14; the separately installed fixture uses the canonical adapter group. |
| Evidence paths | **PASS** — 323 typed `path` fields, zero missing paths. |
| Evidence validator | **BLOCKED FOR RELEASE** — `uv run python scripts/validate_evidence_ledger.py` reports `core: 25/63 (39.7%)`, `overall: 25/65 (38.5%)`; M14 requires 95%/90%. |
| Canonical Markdown links | **FAIL** — two broken links at `docs/GEODESIC_INTERPOLATION.md:6-7` (`../../../latent-anything-theory/...` resolves outside this checkout). Root README/changelog links resolve after treating root files as repository-relative. |
| MkDocs nav paths | **PASS** — 202 `.md`/`.ipynb` nav targets, zero missing targets under `latent-anything-theory/`. |
| Conflict markers | **PASS** — none in README, changelog, `docs/`, sprint plans, artifacts, or `mkdocs.yml`. |
| Strict MkDocs build | **BLOCKED BY ENVIRONMENT** — `uv run mkdocs build --strict` aborts because `mkdocs-jupyter` is not installed; the locked docs extra in `pyproject.toml:26` was not installed and no network/install was used. |
| Static quality gates | **PASS** — Ruff check; Ruff format (`315 files already formatted`); strict Pyright (`0 errors, 0 warnings, 0 informations`); `git diff --check` (only existing LF→CRLF warnings). |
| Full pytest evidence | **REUSED** — unchanged source/test tree since the clean 78.24/78.25 run: `1,545 passed, 36 skipped, 39 warnings` (recorded in `artifacts/task_78.24_summary.md:55-59`). No test was rerun for this read-only audit. |

## Finding ledger

| ID / severity | Exact claim and location | Authoritative state | Proposed resolution and dependency/order | Release impact |
|---|---|---|---|---|
| B-01 Blocking | Evidence validator output; `docs/EVIDENCE_LEDGER.md:39-48`; `docs/M14_REAL_SYSTEM_VALIDATION.md:23-25`; Sprint 78 line 43 remains unchecked. | Current ledger is honest but below the stable gate: 39.7% core and 38.5% overall qualifying coverage. | Keep D0/D1 claims unpromoted; create explicit issues/artifacts for every remaining D0/D1 and rerun the validator in Sprint 79. This precedes Sprint 80 release. | Stable release blocked; intentional M14 stop condition, not a validator bug. |
| B-02 Blocking | `docs/LEROBOT_INTEGRATION.md:287-289` says the committed SmolVLA artifact “promoted” `THY-T05...` to D3. `docs/EVIDENCE_LEDGER.md:138-147`, `docs/evidence-ledger.json:435-446`, `docs/sprint-plans/sprint-61.md:18,36`, and M14 `:111-113` say the corrected CUDA rerun is pending and authoritative status is D2. | Ledger/M14/Sprint 61 are the current override; the integration guide is stale. | Change the guide to state D2 pending rerun, or add a signed, corrected D3 artifact and update all authorities together. Do this before the docs-conflict checkbox or API freeze is closed. | True current-claim contradiction; blocks release-readiness cleanup. |
| B-03 Blocking | Sprint 78 goal `docs/sprint-plans/sprint-78.md:5` says “Cut `0.9.0`”, while package metadata is `pyproject.toml:3` `0.1.0b1`, README `:9,32,38` is still pre-1.0 beta, and PLAN `:7,42` says the path remains `0.1.0-beta.1` → `1.0.0` with M14 pending. | Different lifecycle layers are present, but no document declares whether `0.9.0` is an internal milestone or public release target. | Owner must choose and record one release/version source of truth; synchronize metadata, README, changelog, migration guide, API reference, tag workflow, and Sprint 78/80 before freeze. | Blocks a non-ambiguous release contract; do not infer a version bump. |
| B-04 Blocking | Sprint 78 `:40-45` leaves compatibility snapshots, exception/docstring/typing review, docs reconciliation, theory gate, migration/API guide, API-freeze ADR, changelog, and count gate unchecked. Sprint 78 notes `:49` explicitly reject freeze while these or Actions access remain unresolved. | These are real unfinished release tasks, not stale historical checkboxes. | Complete each atomic gate and only then mark its checkbox; preserve parity snapshots and a dated migration/API decision. Depends on B-01/B-02 and M14 L23/L24. | API freeze and 0.9/1.0 publication blocked. |
| B-05 Blocking (environment) | `mkdocs.yml:14-16` enables `mkdocs-jupyter`; `pyproject.toml:26` declares it in the docs extra, but the strict build aborts with “plugin is not installed.” | Nav target scan passes; the build gate was not executable in this environment. | Run `uv sync --extra docs --locked` in the clean release environment and rerun `mkdocs build --strict`; do not weaken the plugin/nav config. | Docs release gate unverified, not evidence of a content failure. |
| B-06 Blocking (planned) | M14 `:22-25`, `:47`, `:53-54`, `:108-113`, `:124-126` retain explicit external Actions-account, named-3DGS-checkpoint, missing license/access, real-lane, and threshold blockers. | M14 is expressly planned and incomplete; lanes remain `planned`, L17 is `blocked`. | Provision/record required clean/remote lanes, licenses/access, checkpoint and signed waiver only where allowed; stop before release if any remains. No network/model/CUDA was run here. | Blocks Sprint 79/80 release candidate; intentionally not solved by wording. |
| A-01 Advisory → release cleanup | `docs/GEODESIC_INTERPOLATION.md:6-7` links to `../../../latent-anything-theory/...`, resolving to `F:\latent-anything-theory` rather than the sibling project at `F:\ai-ml\latent-anything\latent-anything-theory`. | Broken relative links are an actual deterministic link failure. | Correct to a repository-relative path (likely `../latent-anything-theory/...`) or explicitly exclude external-project links from the link gate; rerun the scan. | Blocks a clean documentation link gate until fixed/waived. |
| A-02 Advisory | `docs/INDEX.md:3-20` says the index covers all high-level documents but omits root `PLAN.md`, `EVIDENCE_LEDGER.md`, `DIFFUSERS_INTEGRATION.md`, `LEROBOT_INTEGRATION.md`, `GEODESIC_INTERPOLATION.md`, `RSSM_TRANSITION.md`, `STOCHASTIC_TRANSITION.md`, `VAE_EXPLANATION_BENCHMARK.md`, `VQ_VAE_INTEGRATION.md`, and `visual-qa-checklist.md`. | The indexed set was read exactly as required; omitted documents are present and referenced by current guides/artifacts. | Add each intended high-level document to the index or state that these are secondary/current integration records. Rerun the read-docs index audit. | Incomplete navigation/source discovery; not a code blocker. |
| A-03 Advisory | `README.md:14` names only four “Built-in adapters,” while the current registry has 32 entries and M14 includes ConvVAE, VQVAE, JEPA, and 3D renderer entries. | “Includes” makes this an incomplete beta summary rather than an exact count, but readers can interpret it as the complete adapter surface. | Clarify that the list is representative or link the authoritative 202/32 inventory; do not change public exports in this audit. | Stale/incomplete release messaging. |
| A-04 Advisory → release cleanup | Dated beta artifacts retain claims now false for current code: `artifacts/release_readiness_0.1.0-beta.1.md:47-72` says no shipped probes/planning/rollout/discrete lanes, and `release_theory_coverage_matrix_0.1.0-beta.1.md:16-24` repeats those claims; `release_readiness...:5` says “Ready to tag.” | Intentionally historical by filename, but no prominent “historical snapshot” banner; M14 `:109-110` specifically requires reconciliation. | Add a dated historical/superseded banner and link current scope/ledger, or archive/replace the old claim. Preserve historical metrics. Depends on B-02/B-03. | Current readers could mistake beta readiness for current readiness. |
| A-05 Advisory | `CHANGELOG.md:218` concatenates two logical bullets (`...#sprint-41)- Evidence-ledger...`) without a newline; current `[Unreleased]` has no Sprint 78 refactor/release-readiness entry. | Changelog history is otherwise chronological; this is a formatting/completeness defect. | Split the bullets and add a user-facing Sprint 78 entry only when release messaging is decided; do not rewrite historical sections. | Changelog extraction/release readability risk. |
| A-06 Advisory → release blocker if freeze attempted | No `docs/MIGRATION.md` or API reference exists; README `:40-43` only points to the config migration diagnostic, while Sprint 78 `:44` requires a `0.9.0` migration guide/API reference. | Absence confirmed by repository inventory and unchecked plan item. | Publish migration/API material after B-03 and alias policy; include import/signature/config/schema/plugin/serialization compatibility snapshots. | Required for no-unplanned-break freeze; not needed for current beta README. |
| A-07 Advisory | `docs/INDEX.md:9` still labels ARCHITECTURE “Initial architecture (not verified)” although later ADRs, PLAN, M14, and implementation artifacts treat architecture decisions as validated or explicitly pending by scope. | Historical wording, not an API/evidence contradiction. | Reword as “living architecture hypothesis/contract” with ADR-status links, or retain only if intentionally historical. | Reader confusion; no direct release gate failure. |

## Historical and intentionally dated claims (not new conflicts)

The following were verified as intentionally scoped and should not be “corrected”
by erasing history: README beta scope and `0.1.0b1` metadata; the
`CHANGELOG.md` `[0.1.0-beta.1]` section; Sprint 26 release artifacts; the Sprint
61 historical CUDA result; synthetic CPU D2 world-model/tokenized/streaming
claims; Diffusers fake-backend D1 versus cached-checkpoint D2; and M14's
explicitly `planned` lanes. The required action is labeling/current-linking,
not rewriting historical metrics or promoting D0/D1 evidence.

## Closure order and verdict

1. Resolve B-03 version authority and B-02 SmolVLA status contradiction.
2. Publish compatibility snapshots, migration/API reference, and the API-freeze
   ADR; repair the two broken links and changelog formatting.
3. Run the M14 clean-environment/real-system matrix, resolve or owner-waive the
   named blockers, and rerun the evidence validator to 95% core / 90% overall.
4. Install the locked docs extra and pass strict MkDocs, then perform Sprint 80
   release-candidate/package gates.

**Verdict: FAIL / NOT RELEASE-READY.** Source/static gates are green and the
202/32/5/12 inventory is consistent, but the D2/D3 threshold is 39.7%/38.5%,
one current SmolVLA D3-vs-D2 contradiction is open, the docs build is
environment-blocked, two links are broken, and release/API/matrix gates are
explicitly unfinished. No new implementation/refactor task is recommended by
this audit; owner should create only the ordered documentation/evidence release
remediation tasks above.
