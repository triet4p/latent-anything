# Sprint 81 Task 9 — Final Release Evidence and Post-1.0 Backlog

**Evidence date:** 2026-09-27  
**Status:** Task 9 remains `[~]`. This artifact is an evidence handoff, not a review verdict. Main's evidence review and the final deep-review gate are pending; Sprint 81 and Milestone 14 are not marked complete.

## Executive disposition

Latent Anything `1.0.0` was published from the exact reviewed release commit after the bounded Sprint 80 depth gate and audited release workflow passed. The post-publication PyPI install, release-link/digest check, separately installed plugin scenario, and pinned CPU encoder-v3 example also passed. Sprints 78–80 and Sprint 81 Tasks 1–8 are represented as complete in the global plan; Task 9 remains `[~]` until its review and the final deep-review gate. Milestone 14 remains open until Main records the final status transition.

The stable claim remains deliberately bounded to the accepted ordinary-DL encoder and transformer evidence. SmolVLA is still **BLOCKED** and non-gating. The live PyPI `1.0.0` long description and immutable `docs-v1.0.0-r1` copies of five API/evidence/plugin documents retain pre-release wording. The corresponding five mutable current-source documents are corrected at the cited post-release commits; that does not rewrite the published PyPI metadata or immutable docs snapshot. The backlog below schedules corrections only at a separately authorized future 1.x package-metadata publication and a separately authorized immutable documentation snapshot; it promises neither.


## Release, CI, provenance, and documentation archive

| Evidence | Observed result and public citation |
|---|---|
| Release source and tag | Protected annotated tag [`v1.0.0`](https://github.com/triet4p/latent-anything/tree/v1.0.0) resolves to tag object `3bdb7ff2c5e09def32e39b9b4628750729104b22` and release commit [`a449ca33b4b83ce109c29db7cf471919820b1d56`](https://github.com/triet4p/latent-anything/commit/a449ca33b4b83ce109c29db7cf471919820b1d56). This closeout is later source; the release tag and package assets were not changed. |
| Exact-SHA CI | [Run 36250365395](https://github.com/triet4p/latent-anything/actions/runs/36250365395) passed on the release commit for Python 3.12, 3.13, and 3.14. |
| Audited release workflow | [Run 36251907256](https://github.com/triet4p/latent-anything/actions/runs/36251907256) completed its five jobs, including artifact attestation, protected-tag creation, GitHub Release publication, and the PyPI OIDC upload. The [GitHub Release](https://github.com/triet4p/latent-anything/releases/tag/v1.0.0) contains five assets. |
| Provenance and checksums | [`PROVENANCE.json`](https://github.com/triet4p/latent-anything/releases/download/v1.0.0/PROVENANCE.json) binds repository `triet4p/latent-anything`, tag `v1.0.0`, commit `a449ca33b4b83ce109c29db7cf471919820b1d56`, and workflow run `36251907256`. Its asset SHA-256 is `8d9e2a2d16da372bf55eb7afebb472c6670570835b2bd6503edc83f053d5accf`. [`SHA256SUMS`](https://github.com/triet4p/latent-anything/releases/download/v1.0.0/SHA256SUMS) has SHA-256 `4fbc2e8690aefa0a36d94776a89704683fb113f4a149e9e475ca5531437d5e05`. |
| Published PyPI wheel | [`latent_anything-1.0.0-py3-none-any.whl`](https://files.pythonhosted.org/packages/2f/4c/3f8dbf03ddf95c92537115366fa23b5ef674e117c6b1ff88fb2d46d8702d/latent_anything-1.0.0-py3-none-any.whl), 635,199 bytes; SHA-256 `3f7081d4cb8994c6a719d85d51a3c0ccff76171673c5a1dc33737d7770b408ea`. The [GitHub release asset](https://github.com/triet4p/latent-anything/releases/download/v1.0.0/latent_anything-1.0.0-py3-none-any.whl) has the same size and digest. |
| Published PyPI source distribution | [`latent_anything-1.0.0.tar.gz`](https://files.pythonhosted.org/packages/02/3f/047ad92d1adf257238e3cc43d3afa9fd3c5bb6a5c0e821a9f4e5edffc883/latent_anything-1.0.0.tar.gz), 886,293 bytes; SHA-256 `36b6b3950fd791cf9ffa5eeac79050c5052d5b9902d1c98a9a2b7a6d38e8e3a9`. The [GitHub release asset](https://github.com/triet4p/latent-anything/releases/download/v1.0.0/latent_anything-1.0.0.tar.gz) has the same size and digest. See the live [PyPI version page](https://pypi.org/project/latent-anything/1.0.0/) and [version JSON](https://pypi.org/pypi/latent-anything/1.0.0/json). |
| Authoritative documentation archive | The corrected annotated [`docs-v1.0.0-r1` tag](https://github.com/triet4p/latent-anything/tree/docs-v1.0.0-r1) points to tag object `653b34f7608913a49531026a266e93deb6818f8a` and commit `3cbc925de464a090bc1d65a2fc8858a11a0dcbf3`. The commit-pinned [versioned documentation record](https://raw.githubusercontent.com/triet4p/latent-anything/3cbc925de464a090bc1d65a2fc8858a11a0dcbf3/docs/VERSIONED_DOCS_1.0.0.md) is 6,611 bytes / SHA-256 `f92766f5fda3998559d58247bde9a75bd86b07479648f1c8bea9c809d0df8e28`; the metadata-only [revision archive](https://raw.githubusercontent.com/triet4p/latent-anything/3cbc925de464a090bc1d65a2fc8858a11a0dcbf3/artifacts/benchmark_revision_archive_1.0.0.json) is 13,507 bytes / SHA-256 `269b9148cedb0d459c901944ac0c7d7bbd188f90724162a1319108b855b100ff`. |

The earlier immutable [`docs-v1.0.0` tag](https://github.com/triet4p/latent-anything/tree/docs-v1.0.0) is retained but superseded by r1 because one manifest checksum used Windows checkout CRLF bytes instead of the canonical release-commit Git-blob bytes. The corrected SmolVLA v1 manifest is 5,068 bytes / SHA-256 `285f540624b389ca274425ef264ca4bbc91be7bad38521879cfc9c27dd2d5ddf`; the original tag and its contents were not rewritten. These docs are versioned GitHub source files, not a claim that root package documentation is deployed to GitHub Pages; Pages serves the separate theory site. The archive records identifiers, revisions, hashes, licenses, and provenance only, and redistributes no model weights or dataset bytes. See the [release notes](../../docs/RELEASE_NOTES_1.0.0.md), [versioned record](../../docs/VERSIONED_DOCS_1.0.0.md), and [Sprint 80 depth-evidence report](../../docs/SPRINT_80_DEPTH_EVIDENCE.md).

## Evidence-review gates and bounded claims

The eight task handoffs are linked below. Each listed reviewer returned PASS; these are prior review results, not a Task 9 or Sprint 81 closure verdict.

| Task | Evidence review | Result |
|---|---|---|
| [1 — Sprint 80 depth-gate signoff](task-1.md) | `Review81Task1Corrected` | PASS for the bounded ordinary-DL core; SmolVLA remains blocked. |
| [2 — Stop-before-release workflow](task-2.md) | `Review81Task2` | PASS; actual publication later passed through the guarded audited workflow. |
| [3 — 1.0.0 metadata and release docs](task-3.md) | `Review81Task3`; post-release correction: `Review81Task3Postrelease` | PASS; current API/evidence docs corrected in commit [`379aaec`](https://github.com/triet4p/latent-anything/commit/379aaec2a7936edb2b5ba08a125c4002d1f58a39), with release/tag/source boundaries preserved. |
| [4 — Clean candidate build/install matrix](task-4.md) | `Review81Task4` | PASS for the recorded Windows x64 / CPython 3.13.3 build and scoped clean-install profiles; not a cross-platform build claim. |
| [5 — GitHub/PyPI publication](task-5.md) | `Review81Task5Final`; deep correction: `Review81Task5DeepCorrection` | PASS; the scoped Task 5 handoff correction is commit `2975da8`; published artifact evidence remains unchanged. |
| [6 — Versioned docs and revision archive](task-6.md) | `Review81Task6Tags`; deep correction: `Review81Task6DeepCorrection` | PASS; current companion/index/release-note corrections are commits `5dea107` and `3d86b22`; immutable `docs-v1.0.0-r1` remains unchanged. |
| [7 — Post-publication consumer smoke](task-7.md) | `Review81Task7Initial` | PASS for the public install, hashes, plugin behavior, and bounded CPU example; immutable wording caveat remains open. |
| [8 — Stable support and release policies](task-8.md) | `Review81Task8Final` | PASS for the published 1.x policy. The private-reporting setting is enabled, but non-maintainer route usability remains unverified. |

The signed-off evidence set is limited to:

- One prospective controlled model-weight lesion on a locally trained four-dimensional linear autoencoder, the pinned `scikit-learn==1.9.0` digits fixture, and one frozen brightness-bin split (selected-data digest `b3ed6f2d420b7dc8de35bb8adc0088b49f76a3b05d31057a0a48e37430966148`). It is not a natural production defect, general encoder result, digit-recognition claim, or GPU result.
- A `transformer.h.0` section-header attribute separability/localization case on `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8` and `Salesforce/wikitext@f776294184f13b8ff2337b3841cf9269a6216d1e` (`wikitext-2-raw-v1`). This is protocol-bounded evidence, not causal feature-use proof or a general model/dataset claim.
- A separate grouped-split stability supplement on the same pinned GPT-2/WikiText inputs. It does not retroactively change the immutable target-evidence-v2 artifact's `threshold: null`.

These cases do not establish arbitrary-model or arbitrary-dataset support, production quality, broad VLA support, deployment readiness, or GPU/CUDA readiness. The former theory-row percentages are portfolio-health facts, not 1.0 release gates.

## Post-publication consumer smoke

The detailed public evidence handoff is [Task 7](task-7.md); its tested code and example were fetched from the immutable release commit, not from this later working tree.

- A fresh CPython 3.13.3 environment, with uv cache disabled and no checkout import, installed `latent-anything==1.0.0` from [public PyPI](https://pypi.org/project/latent-anything/1.0.0/). The import resolved under that environment's `site-packages` and reported version `1.0.0`. Independently downloaded PyPI files matched the GitHub assets and release digests above.
- The separately installed hello-world plugin was discoverable without importing its module; explicit loading, `ObjectSpec` construction, invocation, and provenance checks passed. The observed call result was `hi:world`.
- The release-tagged encoder-v3 CPU example completed capture, detection, localization, explanation, intervention, comparison, and reporting. Run `a6db6481e3483c1a` passed its acceptance gate and independent report validator. The controlled lesion localized to `brightness-axis-dim0`; healthy-counterexample, benign-low-variance, and null-shuffle controls passed. Run artifact SHA-256 `a4ed225a29c9e346a005759d57c7a40bb3c22e9904223b015a947abacda8f7d6`; report SHA-256 `8fb1528f39d355a311d2a8db39e9cf7db6455e1c7392915e21f377a05d13f194`; deterministic rendered-report SHA-256 `9aa1c46f3be3dc77ee5300a057f8eb0dca207f6453d44c96a8ef46d9c639dbf9`.

The consumer smoke used no GPU or pretrained-model download. Its generated run/report files were in external temporary scratch and are not represented as repository-hosted artifacts; the public Task 7 handoff records the exact run and output digests.

## Open findings and evidence-led post-1.0 backlog

| Priority | Follow-up grounded in observed evidence | Completion evidence / boundary |
|---|---|---|
| **P1 — Correct release-facing copy at permitted publication boundaries** | Published PyPI `1.0.0` metadata and five immutable r1 documents retain pre-release wording, while the corresponding five current-source documents are corrected. See the scoped copy-status evidence below. | At the next separately authorized 1.x package-metadata publication, correct its metadata source and verify public version JSON/description and distribution hashes. At the next separately authorized immutable docs snapshot, include all five corrected files and verify each public copy. Keep the published `1.0.0` metadata/assets, `v1.0.0`, and `docs-v1.0.0-r1` untouched; do not rewrite or move tags. This backlog authorizes and promises no release, target version/date, or docs tag. |
| **P1 — Verify security-reporting access for a non-maintainer** | GitHub private vulnerability reporting is enabled, but end-to-end use of **Security → Report a vulnerability** by a non-maintainer remains unverified. The project makes no claim of usable confidential intake, substitute email/form, or response SLA. | Have an owner verify the route from a non-maintainer account; record the observed result and update [`SECURITY.md`](../../SECURITY.md) and the [support policy](../../docs/SUPPORT_POLICY.md) only to match it. If access fails, keep the route explicitly unverified and do not invite sensitive public reports. |
| **P2 — Re-evaluate the secondary SmolVLA lane** | The v1 capture observed `(64, 768)` while asserting `(64, 1024)`; the frozen estimator also rejects 64 samples for 768 dimensions. Execution stopped before diagnosis, no validator-clean report exists, and the prospective 768-to-32 v2 protocol remains unexecuted. | Only proceed under a separately reviewed, predeclared protocol that resolves capture shape and sample/feature constraints, preserves immutable v1 history, and produces independently validated evidence. Keep this lane **BLOCKED**, secondary, and non-gating unless a future owner-approved scope explicitly changes that boundary. |
| **P3 — Consider additional model/dataset claims only when evidence-backed** | Current evidence is narrow and does not establish arbitrary-model or arbitrary-dataset behavior; integration count and theory-row percentages are not diagnostic evidence. | Start only from a concrete consumer use case. Pin model/data revisions, predeclare positive and negative controls and acceptance thresholds, record hardware/runtime scope, and require independent artifact validation before extending a claim. Do not promote general VLA, production, or GPU/CUDA claims from adapter presence or a smoke import. |

### P1 copy-status evidence

The live [PyPI 1.0.0 metadata](https://pypi.org/pypi/latent-anything/1.0.0/json) reports version `1.0.0`, but its description still calls the release an unpublished candidate, says the first upload is pending, and directs installation to `0.9.0`. The immutable [`docs-v1.0.0-r1` tag](https://github.com/triet4p/latent-anything/tree/docs-v1.0.0-r1), pinned at commit [`3cbc925`](https://github.com/triet4p/latent-anything/commit/3cbc925de464a090bc1d65a2fc8858a11a0dcbf3), retains pre-release status prose in exactly these five files; each mutable current-source counterpart is separately linked:

- [`docs/API_REFERENCE.md` — immutable r1 copy](https://raw.githubusercontent.com/triet4p/latent-anything/3cbc925de464a090bc1d65a2fc8858a11a0dcbf3/docs/API_REFERENCE.md); [corrected current source](https://raw.githubusercontent.com/triet4p/latent-anything/379aaec2a7936edb2b5ba08a125c4002d1f58a39/docs/API_REFERENCE.md).
- [`docs/API_COMPATIBILITY.md` — immutable r1 copy](https://raw.githubusercontent.com/triet4p/latent-anything/3cbc925de464a090bc1d65a2fc8858a11a0dcbf3/docs/API_COMPATIBILITY.md); [corrected current source](https://raw.githubusercontent.com/triet4p/latent-anything/379aaec2a7936edb2b5ba08a125c4002d1f58a39/docs/API_COMPATIBILITY.md).
- [`docs/EVIDENCE_GAP_PLAN.md` — immutable r1 copy](https://raw.githubusercontent.com/triet4p/latent-anything/3cbc925de464a090bc1d65a2fc8858a11a0dcbf3/docs/EVIDENCE_GAP_PLAN.md); [corrected current source](https://raw.githubusercontent.com/triet4p/latent-anything/379aaec2a7936edb2b5ba08a125c4002d1f58a39/docs/EVIDENCE_GAP_PLAN.md).
- [`docs/PLUGIN_AUTHOR_GUIDE.md` — immutable r1 copy](https://raw.githubusercontent.com/triet4p/latent-anything/3cbc925de464a090bc1d65a2fc8858a11a0dcbf3/docs/PLUGIN_AUTHOR_GUIDE.md); [corrected current source](https://raw.githubusercontent.com/triet4p/latent-anything/8d288c83501823ce5674c0111e4b37dabd0a935b/docs/PLUGIN_AUTHOR_GUIDE.md).
- [`docs/PLUGIN_TEMPLATE.md` — immutable r1 copy](https://raw.githubusercontent.com/triet4p/latent-anything/3cbc925de464a090bc1d65a2fc8858a11a0dcbf3/docs/PLUGIN_TEMPLATE.md); [corrected current source](https://raw.githubusercontent.com/triet4p/latent-anything/8d288c83501823ce5674c0111e4b37dabd0a935b/docs/PLUGIN_TEMPLATE.md).

Task 3 corrected the three API/evidence current-source files at commit `379aaec` and passed `Review81Task3Postrelease`. The mutable plugin guide/template corrections are at `8d288c8`; their source links do not imply the immutable r1 copies or PyPI metadata changed. Task 5 and Task 6 correction reviews and their evidence remain recorded in the review table.

## Closure status and exact next gate

- Sprints 78, 79, and 80 are complete. Sprint 81 Tasks 1–8 passed review. The global plan reflects these facts and leaves Sprint 81 in progress.
- Task 9 remains `[~]`; Main must review this artifact, public links, source corrections, and plan language. This file does not mark its own checkbox `[x]`.
- After Task 9 evidence review passes, a separate final deep review must pass with no actionable findings. Only then may Main mark Task 9 and Sprint 81 complete and mark Milestone 14 complete. If either review finds an issue, keep the corresponding status open and record the correction/findings here.
- Main owns the project-wide validation pass after all sibling work has landed. No project-wide build, test suite, formatter, or linter was run for this documentation closeout.

## Correction lineage and changed files

The earlier public closeout commit [`8d288c8`](https://github.com/triet4p/latent-anything/commit/8d288c83501823ce5674c0111e4b37dabd0a935b) added the initial final handoff, Task 7's post-publication evidence, current-source plugin-guide corrections, `docs/PLAN.md` release status, and the then-current Task 9 narrative. Subsequent reviewed corrections are separately scoped:

- [`docs/API_REFERENCE.md`](https://raw.githubusercontent.com/triet4p/latent-anything/379aaec2a7936edb2b5ba08a125c4002d1f58a39/docs/API_REFERENCE.md), [`docs/API_COMPATIBILITY.md`](https://raw.githubusercontent.com/triet4p/latent-anything/379aaec2a7936edb2b5ba08a125c4002d1f58a39/docs/API_COMPATIBILITY.md), and [`docs/EVIDENCE_GAP_PLAN.md`](https://raw.githubusercontent.com/triet4p/latent-anything/379aaec2a7936edb2b5ba08a125c4002d1f58a39/docs/EVIDENCE_GAP_PLAN.md) — current-source publication-status correction at Task 3 commit `379aaec`; PASS at `Review81Task3Postrelease`.
- [`docs/PLUGIN_AUTHOR_GUIDE.md`](https://raw.githubusercontent.com/triet4p/latent-anything/8d288c83501823ce5674c0111e4b37dabd0a935b/docs/PLUGIN_AUTHOR_GUIDE.md) and [`docs/PLUGIN_TEMPLATE.md`](https://raw.githubusercontent.com/triet4p/latent-anything/8d288c83501823ce5674c0111e4b37dabd0a935b/docs/PLUGIN_TEMPLATE.md) — mutable-source 1.x wording was corrected; the copies in immutable r1 remain historical.
- Task 5's closeout-scope correction (`2975da8`, `Review81Task5DeepCorrection`) and Task 6's companion/index/release-note corrections (`5dea107`, `3d86b22`, `Review81Task6DeepCorrection`) are recorded above; neither changes release assets or immutable r1 content.
- [`docs/sprint-plans/sprint-81.md`](../../docs/sprint-plans/sprint-81.md) — Main-authored narrative links the deep-review finding and correction reviews; Task 9 remains `[~]`, and Sprint 81/Milestone 14 remain open.
- [`artifacts/sprint-81/task-9.md`](task-9.md) — this follow-up reconciles corrected mutable sources with the still-stale PyPI/r1 copies and records the next permitted boundaries.

`docs/PLAN.md` already links this artifact and keeps Task 9, Sprint 81, and Milestone 14 open, so it is intentionally unchanged. The sprint-plan narrative edit is included without changing its task-status checkbox.

## Focused verification and graph update

The live PyPI 1.0.0 JSON and all ten commit-pinned raw documentation URLs linked in the P1 status map were fetched directly. The JSON reports version `1.0.0` while retaining the stale candidate/unpublished and 0.9.0-install text; the current-source files describe the published 1.x contract, while the immutable r1 copies retain the earlier wording. No PyPI upload, release asset, package tag, or immutable documentation tag was changed.

The scoped local Markdown/link check covered this artifact and `docs/sprint-plans/sprint-81.md`: **17 relative links, 0 missing**; Python-Markdown rendered the backlog as a five-row table and the P1 status list as `<ul>`.

Graphify 0.9.68 was updated incrementally for only these two documents; the graph was not reclustered and its report/HTML were not regenerated. The initial scoped merge extracted **16 AST nodes / 29 edges**, preserved **7 prior semantic source nodes / 8 edges**, and added **3 explicit concepts / 4 reference edges**. It grew the graph from **16,799 nodes / 39,411 edges / 39 hyperedges** to **16,803 / 39,415 / 39**, preserving all **16,777 unrelated nodes**, **39,377 unrelated edges**, and all hyperedges; losses were zero. After the verification text was finalized, Task 9 alone was re-extracted (**11 AST nodes / 24 edges**); the persisted Task 9 semantic slice remained **9 concept/rationale nodes / 10 edges**, and the graph totals were unchanged. The manifest has **1,821** rows, with only the two target rows refreshed and **0** other rows changed. Final diagnostics matched the baseline's **1 unverified node** and reported zero malformed edges, missing/dangling endpoints, external references, self-loops, duplicates, or same-endpoint collapses. A focused graph query returned the new node **“Immutable docs-v1.0.0-r1 copies retain pre-release wording in five files.”**

