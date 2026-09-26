# Task Summary: Audited 1.0.0 GitHub and PyPI Publication Path

**Sprint:** Sprint 81  
**Task:** Task 5 — Publish signed/checksummed package artifacts and the stable GitHub/PyPI release through the audited workflow  
**Plan status:** Remains `[~]`; Main owns the evidence review and status change.

## Outcome

Audited release run [36251907256](https://github.com/triet4p/latent-anything/actions/runs/36251907256) published protected tag `v1.0.0`, the five-asset GitHub Release, and the first PyPI wheel/sdist through OIDC. PyPI project JSON is HTTP 200 for version `1.0.0`; the wheel/sdist digests match the GitHub Release assets, and provenance plus GitHub attestations verify the tag, release commit, and run. Exact-release-commit CI run 36250365395 passed. The post-publication PyPI readiness metadata and present-tense documentation are reconciled in this follow-up. Task 5 remains `[~]` pending Main's evidence review.

Historical record: the pre-publication sections below preserve the 2026-09-26 pre-upload/release-preflight snapshot, including the failed ruleset probe and expected PyPI HTTP 404. Those statements describe the state before release run 36251907256; the current outcome above and reconciliation section below supersede them.

## Work and Changed Files

- [`.github/workflows/release.yml`](../../.github/workflows/release.yml) — the existing task work adds artifact attestation and OIDC-only PyPI publishing; the PyPI job uses environment `pypi` and job-scoped `id-token: write`.
- [`tests/test_release_workflow_contract.py`](../../tests/test_release_workflow_contract.py) — structural workflow contract coverage for gate ordering, artifact integrity, OIDC scope, environment, and absence of PyPI credentials.
- [`scripts/check_release_readiness.py`](../../scripts/check_release_readiness.py) — now matches required tasks by title prefix instead of plan position (owner-approved Task 8 reorder fix); pending PyPI is accepted only when the screenshot-verified publisher configuration exactly matches `latent-anything` / `triet4p/latent-anything` / `release.yml` / `pypi`, configuration evidence is present, and project JSON status is HTTP 404. Active status instead requires the same verified configuration, HTTP 200, and activation evidence. Missing, malformed, mismatched, or unexpected HTTP states remain blocking.
- [`tests/test_release_readiness.py`](../../tests/test_release_readiness.py) — covers the verified pending bootstrap, exact identity match, active-after-upload evidence, fail-closed cases, plus the reordered-plan title-matching regression test.
- [`docs/release-gates.json`](../../docs/release-gates.json) — records the PyPI publisher as `pending` with verified configuration and expected HTTP 404; records the stable-tag ruleset as `pending` with the refreshed 2026-09-26 authenticated API evidence (empty rulesets, HTTP 422 Actions-actor probe, no update/delete-only shortcut).
- [`docs/M14_REAL_SYSTEM_VALIDATION.md`](../../docs/M14_REAL_SYSTEM_VALIDATION.md) and [`docs/PLAN.md`](../../docs/PLAN.md) — document the pending-publisher bootstrap semantics and corrected blocker state: Task 8 complete (no longer blocking), stable-tag ruleset the only external blocker, Task 5 still `[~]`. `SECURITY.md` and `docs/SUPPORT_POLICY.md` record the verified enabled private-reporting setting while stating non-maintainer usability remains unverified; Task 8's historical artifact predates that enablement check, as clarified below.
- [`pyproject.toml`](../../pyproject.toml), [`uv.lock`](../../uv.lock) — PyYAML dependency added/locked for the existing workflow contract tests.
- `graphify-out/graph.json`, `manifest.json`, `GRAPH_REPORT.md`, `graph.html`, `.graphify_analysis.json`, `.graphify_labels.json`, and `.graphify_labels.json.sig` — targeted graph refresh; `graphify-out/.graphify_python` pins the Graphify interpreter. Graphify preserved a dated backup under `graphify-out/2026-09-26/`.

`docs/sprint-plans/sprint-81.md` and Task 8 were not changed; no task checkbox was bypassed. `CHANGELOG.md` remains unchanged because no 1.0.0 release date exists.

## PyPI Bootstrap Semantics

The user-provided PyPI Publishing screenshot matches project `latent-anything`, repository `triet4p/latent-anything`, workflow `release.yml`, and environment `pypi`. The observed `https://pypi.org/pypi/latent-anything/json` response was HTTP 404. PyPI's [first-project Trusted Publishing guidance](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) documents that a matching pending publisher can create the project on its first successful OIDC upload; the project is not created or reserved beforehand. Thus HTTP 404 is expected before that upload, not proof of a publisher mismatch. The manifest remains `pending`: this does not establish an active PyPI project or authorize a release. After a successful first upload, the manifest must record verified active publisher evidence and project JSON HTTP 200.

## External Controls and Blockers

- Authenticated GitHub access as `triet4p` (`permissions.admin: true`); `GET /repos/triet4p/latent-anything/rulesets?includes_parents=true` returned `[]` on 2026-09-26.
- A valid-JSON creation+update+deletion tag ruleset for `refs/tags/v*` with the GitHub Actions integration (app id 15368) as `Integration` bypass actor was re-attempted on 2026-09-26 and returned HTTP 422: `Actor GitHub Actions integration must be part of the ruleset source or owner organization`. No ruleset was created; `release-tag-ruleset` remains `pending`. An update/delete-only ruleset was not accepted as sufficient: the required rule must block creation by ordinary direct writers while permitting only the audited owner-installed GitHub App/workflow credential.
- Owner action (smallest external prerequisite, owner-only): create/install an owner-managed GitHub App with minimum repository permission to create the stable version tag (for example repository `contents: write`), use its token only in the audited tag-creation job, and configure/verify a stable-tag ruleset (`target: tag`, `refs/tags/v*`, `creation`+`update`+`deletion`, `Integration` bypass actor = that app's numeric app ID, `bypass_mode: always`, `enforcement: active`) through the authenticated GitHub API. The available OAuth/PAT credential cannot supply an installable Integration actor ID.
- `uv run --locked --no-sync python scripts/check_release_readiness.py` now reports only `External release prerequisite is not ready: release-tag-ruleset (status: pending)` (exit 1). Sprint 81 task 8 is complete (owner-approved reorder, Review81Task8 PASS) and the title-matched preflight no longer false-blocks on its reordered position. The pending PyPI publisher is not itself a blocker when the exact verified bootstrap configuration and HTTP 404 are recorded.
- Private vulnerability reporting: `PUT /repos/triet4p/latent-anything/private-vulnerability-reporting` returned HTTP 204 and `GET` now returns `{"enabled": true}` (verified 2026-09-26). `SECURITY.md` and `docs/SUPPORT_POLICY.md` state the enabled setting while noting non-maintainer end-to-end usability remains unverified; no confidential-intake promise, contact, or SLA was added. Task 8's historical artifact predates this enablement check; the current policy documents are authoritative for the present setting and do not claim tested reporter usability.
- The latest observed GitHub release remains `v0.9.0`; no `v1*` tag exists locally (`git tag --list "v1*"` empty); PyPI project JSON remains HTTP 404. No 1.0.0 GitHub Release or PyPI upload is claimed. End-to-end GitHub Actions and PyPI OIDC publication remain unverified because the ruleset prerequisite is blocked.

## Testing and Verification

- **Passed:** `uv run --locked --no-sync pytest -q tests/test_release_workflow_contract.py tests/test_release_readiness.py` — **18 passed**, including the missing-App-credential and App-bypass contract scenarios.
- **Expected fail-closed result:** `uv run --locked --no-sync python scripts/check_release_readiness.py` — exit 1 with exactly one blocker: `External release prerequisite is not ready: release-tag-ruleset (status: pending)`.
- **Live API evidence (2026-09-26):** rulesets `[]`; ruleset creation probe HTTP 422 with no ruleset created; private-vulnerability-reporting `GET {"enabled": true}`; PyPI JSON HTTP 404; `gh release list` latest `v0.9.0`; `git tag --list "v1*"` empty.
- No project-wide suite, formatter, linter, build, GitHub Actions run, tag, push, workflow dispatch, or upload was performed.

## Graph Update

The Graphify refresh was scoped to the eight Task 5 paths below. The initial full scan found 301 changed files; only these eight were processed and stamped, with no other repository changes stamped:

- `SECURITY.md`, `docs/SUPPORT_POLICY.md`, `docs/M14_REAL_SYSTEM_VALIDATION.md`, `docs/PLAN.md`,
  `docs/release-gates.json`, `scripts/check_release_readiness.py`,
  `tests/test_release_readiness.py`, and this artifact.
- Final clustered graph: **16,630 nodes, 37,996 edges, 1,110 communities**. It preserved
  **16,509 unrelated nodes and 37,744 unrelated edges**, lost **0** unrelated nodes or edges,
  and had **0** unselected manifest-hash mismatches. Manifest rows for all eight selected paths are current.
- Graph integrity: **0** missing/dangling endpoints, self-loops, or directed/undirected same-endpoint
  collapsed edges. The single `verification=unverified` node (`π0 (Pi0)` at
  `latent-anything-theory/10-world-models-vla/research/07-pi0.md`) was already present in the
  pre-refresh graph; it is an unchanged, unrelated diagnostic.
- Vocabulary-expanded query `[stable, ruleset]` returned the pending owner-managed stable-tag ruleset
  node, the Task 5 correction, and the pending first-upload PyPI publisher. Query
  `[private, reporting, enabled, usability, unverified]` returned current Task 5, `SUPPORT_POLICY.md`,
  and `SECURITY.md` nodes that distinguish the enabled setting from unverified reporter usability,
  plus the Task 8 historical-snapshot note. No graph result claims usable confidential intake.
  The query answer was saved to
  `graphify-out/memory/query_20260926_055955_5436220e_what_does_the_current_sprint_81_task_5_graph_say_a.md`.
- `GRAPHIFY_NO_BACKUP=1 <pinned graphify.exe> cluster-only .` completed and regenerated the
  aggregated `graph.html` with 1,110 communities. The dated
  `graphify-out/2026-09-26/graph.json` backup was left untouched.

No tag, push, upload, release, or publication occurred. Task 5 remains `[~]`. The owner-managed GitHub App/ruleset is still the release blocker; the exact configured PyPI first-upload bootstrap and expected HTTP 404 remain pending setup, not a second blocker. Private reporting is enabled, but non-maintainer end-to-end usability remains unverified. Task 8's historical artifact predates enablement; the current policy documents state the setting and its verification limits.

## Open Findings

1. Stable release-tag ruleset remains pending; only the repository owner can supply the installed-App bypass actor. No tag/upload may occur until the effective ruleset verifies.
2. Title-matching preflight fix is in place with regression coverage; the old positional false-block on the reordered plan is resolved.
3. Private vulnerability reporting is setting-enabled (`GET {"enabled": true}`) but end-to-end non-maintainer usability is unverified; usable confidential intake still cannot be promised.
4. The first successful PyPI OIDC upload has not occurred. Pending configuration and HTTP 404 are valid bootstrap evidence only; active status and HTTP 200 remain unverified.
5. No 1.0.0 publication occurred, so workflow execution and release-date changelog evidence remain outstanding. `docs/sprint-plans/sprint-81.md` task 5 stays `[~]`; Main owns the review and status change.

## App-token correction (2026-09-26)

### Implementation

The `create-release-tag` job now checks both required repository secrets before token creation or checkout, then uses `actions/create-github-app-token@v3` with the numeric `RELEASE_APP_ID`, `RELEASE_APP_PRIVATE_KEY`, this repository only, and `contents: write`. `actions/checkout` persists that token for the annotated tag push. The job's `GITHUB_TOKEN` is read-only (`contents: read`); attestation and PyPI permissions, including the PyPI `pypi` environment and `id-token: write`, remain unchanged.

The focused workflow contract test executes the credential guard with each secret missing and with both present, checks guard/token/checkout/push ordering, the repository and permission scope, checkout's token, and the Integration bypass contract. The owner setup contract is recorded in `docs/release-gates.json`; `docs/M14_REAL_SYSTEM_VALIDATION.md` gives the owner steps and API verification; `docs/PLAN.md` now distinguishes the default Actions actor failure from the new custom-App workflow wiring.

Changed for this correction: `.github/workflows/release.yml`, `tests/test_release_workflow_contract.py`, `docs/release-gates.json`, `docs/M14_REAL_SYSTEM_VALIDATION.md`, and `docs/PLAN.md`. No private-key value or App credential was added to tracked files.

### External state and owner action

The current read-only `GET /repos/triet4p/latent-anything/rulesets?includes_parents=true` returned `[]`. The available authenticated evidence records repository-admin access as `triet4p`, the same empty ruleset list, and HTTP 422 when attempting to add GitHub Actions integration app 15368 as the bypass actor. The App-settings browser attempt failed before navigation because the browser relay broker was unavailable, so no owner App registration, installation, repository secret, or ruleset change was made. `release-tag-ruleset` remains `pending`.

The owner must register an App under the repository-owner account with webhooks disabled and only repository `contents: write`, install it only on `triet4p/latent-anything`, and save its numeric App ID and private key as `RELEASE_APP_ID` and `RELEASE_APP_PRIVATE_KEY` repository secrets. Create an active tag ruleset for `refs/tags/v*` with `creation`, `update`, and `deletion` rules and no bypass actor other than that App as `Integration`, using `actor_id` equal to the App ID and `bypass_mode: always`. Verify the response from `GET /repos/triet4p/latent-anything/rulesets?includes_parents=true`; update the prerequisite to ready only after the exact active target, rules, and sole App actor are confirmed.

### Verification

- **Passed:** `uv run --locked --no-sync pytest -q tests/test_release_workflow_contract.py tests/test_release_readiness.py` — **18 passed**.
- **Expected fail-closed result:** `uv run --locked --no-sync python scripts/check_release_readiness.py` — exit 1 with exactly one blocker: `External release prerequisite is not ready: release-tag-ruleset (status: pending)`.
- **Scoped graph refresh:** the initial 298-file detection selected only `.github/workflows/release.yml`, `tests/test_release_workflow_contract.py`, `docs/release-gates.json`, `docs/M14_REAL_SYSTEM_VALIDATION.md`, and `docs/PLAN.md`; the other 293 pending changes were left untouched. The AST pass produced 12 nodes / 35 edges; five manually grounded semantic nodes and nine edges recorded the new App-token contract because the pinned Graphify environment lacked the Gemini/OpenAI client packages. No packages were installed.
- **Graph result:** the refresh grew the graph from 16,637 nodes / 38,003 edges to 16,648 / 38,025; it lost zero unrelated nodes or edges, changed zero unselected manifest rows, and left none of the five selected paths pending. Diagnostics reported zero missing/dangling endpoints, self-loops, exact duplicates, or collapsed edges. `cluster-only` completed with 1,088 communities and 1,564 cross-community edges; Graphify automatically renamed communities by their hubs, and no LLM relabel was run.
- **Artifact-only graph refresh:** after this summary was added, Graphify selected only `artifacts/sprint-81/task-5.md` (293 other pending paths untouched); it replaced 6 prior semantic nodes with 8, preserved all unselected nodes/edges and all 37 hyperedges, and recorded 16,650 nodes / 38,036 edges with zero endpoint, duplicate, self-loop, or edge-collapse diagnostics. Only this artifact's manifest row changed and is current, with no target path pending. The subsequent `cluster-only` report has 1,076 communities and 1,623 cross-community edges; hub-based renaming occurred without LLM relabeling. One unverified node remains.
- No GitHub Actions run, tag, push, dispatch, upload, release, or publication was attempted. Task 5 remains `[~]`; Main owns the evidence review and status change.

## Review81Task5 workflow-runtime correction (2026-09-26)

### Changes

- `.github/workflows/release.yml` now keeps the original `dist` files as the checksum, provenance, and GitHub-attestation subjects. After those original files pass checksum/provenance checks and GitHub attestation verification, the PyPI job stages only the wheel and sdist in `dist-pypi`, checks each source and staged SHA-256 against the verified provenance digest, and rejects any other staging contents. The GitHub asset path still contains `SHA256SUMS`, `PROVENANCE.json`, and `release-body.md`; `pypa/gh-action-pypi-publish@v1.14.2` receives only `dist-pypi`.
- The tag job configures the deterministic, nonsecret `github-actions[bot]` Git identity after checkout and before annotated tag creation. The App installation token remains the only push credential; no user-personal identity or credential was added.
- `tests/test_release_workflow_contract.py` now executes the staging script against checksum-recorded inputs and checks the exact wheel/sdist set and byte equality. Its tag scenario disables system/global Git configuration, observes tag creation fail before identity setup, runs the workflow identity commands, and verifies the annotated tag succeeds with the expected tagger.

### Focused failure-before/passing-after evidence

- **Git tag:** in a temporary repository with clean `HOME`, absent global config, and system Git config disabled, annotated tag creation failed before identity setup with exit 128 (`unable to auto-detect email address`). Running the workflow's identity setup then allowed creation; the tag object records `github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>`.
- **Twine selection:** using a temporary `uv build` of the release candidate plus the workflow's evidence files, Twine rejected the shared `dist` file selection (exit 1, `Unknown distribution format: '.gitignore'`). Running the workflow staging script created exactly `latent_anything-1.0.0-py3-none-any.whl` and `latent_anything-1.0.0.tar.gz`, each matching its verified provenance SHA-256; `twine check` on those staged packages passed (exit 0). No upload was attempted.
- **Passed:** `uv run --locked --no-sync pytest -q tests/test_release_workflow_contract.py` — **8 passed**, including the clean-HOME tag scenario, staged distribution behavior, and existing OIDC/publisher and fail-closed contracts.
- No GitHub Actions run, workflow dispatch, remote release-tag push, GitHub Release, PyPI upload, or publication occurred. The tag created in verification exists only in a temporary local repository. The owner-managed App/ruleset prerequisite remains pending, so Task 5 remains `[~]`.

### Graph update

The scoped Graphify refresh detected **296** pending files and selected only the three files changed for this correction; **293 unrelated pending files** were left untouched and unstamped.

- The graph grew from **16,650 nodes / 38,036 edges** to **16,659 nodes / 38,055 edges**. All **16,622 unrelated nodes**, **37,964 unrelated edges**, and **37 hyperedges** were preserved. Three source-less AST import nodes from the changed contract test were added; no unrelated manifest row changed.
- Graph integrity reported **0** missing/dangling endpoints, self-loops, exact duplicates, directed/undirected endpoint collapses, or external-reference edges. The three assigned manifest rows are current and none of those files remains pending.
- `graphify cluster-only .` completed with **1,103 communities** and regenerated `graph.html` with **1,103 community nodes / 1,741 cross-community edges**.
- The final artifact-only Graphify refresh after this status clarification selected only `artifacts/sprint-81/task-5.md`; the graph remained at **16,659 nodes / 38,055 edges**, preserving **16,650 unrelated nodes**, **38,030 unrelated edges**, and all **37 hyperedges**. The other 293 pending files and all other manifest rows were unchanged; this artifact's final manifest hash is current.

## Resume continuation (2026-09-26)

### External re-verification

- Active stable-tag ruleset `24036634` (`latent-anything-release-rules`,
  `target: tag`, `enforcement: active`,
  `conditions.ref_name.include: [refs/tags/v*]`,
  `rules: [deletion, non_fast_forward, creation, update]`,
  `bypass_actors: [{actor_id: 5083983, actor_type: Integration,
  bypass_mode: always}]`, `current_user_can_bypass: never`).
- `pypi` environment id `22803163260`: main-only custom branch policy
  (`61095674`), required reviewer `triet4p` (id `119602471`,
  `prevent_self_review: false`), environment secrets empty, and
  `can_admins_bypass` changed `true` -> `false` via a preserving `PUT`
  (`reviewers`, `prevent_self_review`, `deployment_branch_policy` unchanged;
  verified `updated_at: 2026-09-26T09:46:39Z`).
- Repository secrets present by metadata only: `RELEASE_APP_ID`
  (`updated 2026-09-26T08:51:40Z`) and `RELEASE_APP_PRIVATE_KEY`
  (`updated 2026-09-26T09:11:30Z`); values remain unreadable by API design.
- PyPI project JSON still HTTP 404 (expected pre-first-upload bootstrap);
  latest GitHub release still `v0.9.0`; `git tag --list "v1*"` and remote
  `refs/tags/v1*` both empty; no dispatch, tag, upload, or publication.
- Controlled non-publishing App-token proof was attempted safely and stopped
  before risk: an isolated `task5-app-token-check.yml` was pushed only to a
  temporary branch, but GitHub resolves dispatchable workflows from the
  default branch, so dispatch returned HTTP 404 and no run was created. The
  temporary branch and worktree were fully deleted (remote 404 confirmed; no
  workflow runs created). No secret value was printed or stored.

### Readiness and remaining gates

- `docs/release-gates.json` now records `release-tag-ruleset: ready` with the
  exact API evidence above; PyPI publisher remains valid `pending` bootstrap.
- `uv run --locked --no-sync python scripts/check_release_readiness.py`
  passes (exit 0).
- Focused tests pass: `20 passed` for
  `tests/test_release_workflow_contract.py` +
  `tests/test_release_readiness.py`.
- `docs/PLAN.md` and `docs/M14_REAL_SYSTEM_VALIDATION.md` no longer claim a
  pending/empty ruleset; they record the verified active ruleset and the
  dispatch-time PEM proof limit.
- Not done: audited `release.yml` dispatch from `main`, App-scoped tag push,
  GitHub Release `v1.0.0`, first PyPI OIDC upload / HTTP 200, changelog
  release date, and any full CI/build/quality proof on the release candidate.
  `origin/main` CI is separately red (ruff `E501`/`SIM101` etc. on pushed
  Sprint 80 code), and local `ruff check` also reports unrelated pre-existing
  errors plus task-5 long lines; per task constraints no formatter/linter,
  build, or project-wide suite was run mid-flight beyond the focused checks.
- Task stays `[~]`; no release workflow dispatch, tag, or PyPI upload was
  performed. Next exact action: Main decides whether to authorize the
  irreversible audited dispatch (only after the release candidate's full
  quality/build gates are proven on the exact dispatch commit).

### Graph update

- Ran `graphify update .` (code re-extract, no LLM): rebuilt to **16,675
  nodes / 39,253 edges / 1,085 communities** (`graph.html` regenerated as
  aggregated community view; dated backup preserved under
  `graphify-out/2026-09-26/`). 329 source files produced zero AST nodes
  (unchanged behavior, retried next run).
- `graphify-out/` is git-ignored (`git check-ignore` confirms), so the
  refresh produced no tracked diff (`git status --short -- graphify-out/`
  empty). The four changed paths this round are docs-only; `graphify update`
  covers code files (doc/paper semantic extraction needs the LLM pipeline),
  so no doc-semantic nodes were added — recorded as a limit, not a claim.
### Correction after FAIL review and local verification (2026-09-26)

This continuation supersedes the earlier preflight-exit-0 and “no full release
quality run” statements above. No release action was taken.

#### CI revision diagnosis and local quality

- GitHub run `35755750287` failed on `origin/main` SHA
  `24850717ca3f55afbd8d19dc28aeca587744f996` with 203 Ruff findings across
  18 `src/` files. The local branch is at `9d9c613393976e5cd96da1ba79e38f97a4729f8d`,
  24 commits ahead and zero behind; the committed range changes 45 `src/`
  paths. The failed remote run is not a CI result for this candidate.
- Before correction, the local worktree Ruff run had 14 findings: 13 `E501`
  lines in release-readiness code/tests and one `F821` (`Path`) in
  `tests/test_api_compatibility.py`. They were corrected, including
  formatting only the four release-related files that Ruff identified.
- `uv run --locked --no-sync ruff check src tests scripts`: **PASS**.
- `uv run --locked --no-sync ruff format --check src tests scripts`: **PASS**
  (`511 files already formatted`).
- `uv run --locked --no-sync pyright`: **PASS**, 0 errors, 0 warnings,
  0 informations.
- `uv run --locked --no-sync mkdocs build --strict`: **PASS**; documentation
  built in 36.24 seconds. MkDocs emitted its nonfatal Material/MkDocs 2.0
  warning.

#### Release-gate correction and focused proof

- `release-app-token-proof` is now a required external prerequisite. Its
  `pending` status and evidence are recorded in `docs/release-gates.json`;
  `docs/PLAN.md` and `docs/M14_REAL_SYSTEM_VALIDATION.md` say that secret
  metadata is visible but token minting, actor identity, repository scope, and
  `contents: write` have not been verified. This does not contradict the
  user's reported PEM re-paste; no token-mint run was observed.
- The release-gates ruleset record and plan now include the API-verified
  `non_fast_forward` rule alongside creation/update/deletion. The
  `test_release_tag_app_matches_integration_bypass_contract` expectation was
  aligned to that evidence.
- A full-suite run initially found the test fixture missing the required
  release-prerequisite schema. The fixture now includes
  `latent-anything-release-prerequisites-v1`; the contract test also records
  `non_fast_forward`. After these fixes:
  `uv run --locked --no-sync pytest -q tests/test_release_readiness.py
  tests/test_release_workflow_contract.py tests/test_api_compatibility.py`
  passed: **29 passed**.
- `uv run --locked --no-sync python scripts/check_release_readiness.py`
  exits **1** as intended, with the sole blocker
  `release-app-token-proof (status: pending)`.

#### Full-suite findings still blocking a green candidate

- The one full-suite run collected 2,578 items and ended
  **23 failed, 2,511 passed, 46 skipped, 39 warnings**. The 15 release
  readiness/workflow assertions from that run were corrected and the focused
  post-fix tests pass; the full suite was not rerun after those corrections.
- Two other failures are the Sprint 80.25 pinned-input checks:
  `pyproject.toml` currently hashes to
  `2fd9fac2516956bc09873fa88a069afa846fe2c9225fc23365b8ced42721c2a9`
  instead of its pinned `106da07c210ec587de33e93564777d080c0eaddba2bdcd851a0fd6986b2b05c6`;
  the package-source-tree hash is `8df33742de6c3072969aff823b514500c4b067592fc4d0ae303638f994fed3b6`
  instead of `3226dc26ea19e15889e29f70c33c83ac29a0cf66115a81bd229a3b8eace75be4`.
  I did not change these pins: the clean-replay evidence must be rerun and
  reviewed for the changed inputs rather than re-pinned without proof.
- Six `tests/test_transformer_end_to_end_proof.py` cases failed to import
  `datasets`. The proof driver documents a `transformers` plus
  `datasets>=2.19,<4`/`huggingface-hub` environment, while `.github/workflows/ci.yml`
  syncs only `--extra viz`. This test-profile gap remains unresolved; no tests
  were skipped or stubbed to conceal it.

#### Build/package checks and publication stop

- `uv sync --locked --extra docs` passed.
- `uv build --wheel --sdist` passed for `latent_anything-1.0.0`; `twine check`
  passed for both distributions.
- Each built archive was installed into a fresh locked base environment.
  Wheel and sdist imports reported version `1.0.0`; both `latent-anything
  --help` smoke runs succeeded and both `uv pip check` runs reported 39
  compatible packages. Temporary build/install environments were removed.
- No `v1.0.0` tag, release-workflow dispatch, GitHub Release, or PyPI upload
  was performed; `CHANGELOG.md` is intentionally unchanged because there was
  no publication.
- The working-tree Sprint 81 plan has Tasks 1–4 and 8 `[x]` and Task 5 `[~]`,
  but those Main-owned status edits are not in `HEAD`. The local main branch
  also contains 24 commits ahead of `origin/main`; I did not stage, commit, or
  push the Main-owned plan or this broader candidate history. CI has not run
  green on the exact release-candidate commit.
- Task 5 remains `[~]`. Before any release: Main must review the ahead
  history/plan state and residual test blockers; a nonpublishing default-branch
  run must prove App-token minting, actor, repository scope, and
  `contents: write`; then the exact candidate must pass CI and all quality/build
  gates. Only after those gates may a separately authorized release dispatch
  proceed.

#### Final graph refresh

- After the final code/format changes, `graphify update .` completed its
  code-only/no-LLM refresh: **16,677 nodes, 39,264 edges, 1,064 communities**.
  `graph.html` is an aggregated view with 1,064 community nodes and 1,861
  cross-community edges; the dated backup is under
  `graphify-out/2026-09-26/`.
- The updater retried 336 uncached code paths; 329 produced zero AST nodes
  (mostly detected JSON inputs). It noted 1,068 saved labels versus 1,064
  communities and renamed 34 communities by hub; I did not run the optional
  LLM `graphify label` step.
- This CLI refresh covers code only; changed docs received no semantic graph
  extraction. `graphify-out/` remains git-ignored.

## Latest correction and handoff (2026-09-26)

- Historical reproducibility: the isolated Sprint 80.25 replay passed all 24
  checks with 0 failures in 2,429.965 seconds (run
  `20260926-115522-629609-4488`). The run removed its temporary clean root.
  Observed current input pins were `pyproject.toml`
  `7050e09260e2f11dcd12b46faf65ddd7f526563d3a78846187b2b1898653913d`,
  `uv.lock`
  `cb6a23d95b73954f212a29e3e0360fc20773ead147c7709c460e4c6e79eb1379`,
  and the 162-file source tree
  `8df33742de6c3072969aff823b514500c4b067592fc4d0ae303638f994fed3b6`.
- The clean replay used a noneditable install with Python 3.13.3,
  `datasets==3.6.0`, `huggingface-hub==0.35.3`,
  `transformers==4.57.6`, and `torch==2.10.0+cpu`; pip reported 64 compatible
  packages. The run's historical evidence is preserved under
  `artifacts/task_80.25_evidence_committed_20260926-115522-629609-4488/`.
- Updated replay pins are covered by
  `uv run --locked --no-sync pytest -q tests/test_sprint80_25_clean_repro.py`:
  7 passed. Added the `ci-proofs` dependency group and enabled it in CI/release
  workflows; `uv sync --locked --extra docs --extra viz --group ci-proofs`
  succeeded. Release-quality observations: Ruff check passed for `src/` and
  `tests/`; formatting check reported 329 files already formatted; Pyright
  reported 0 errors/warnings/information; `mkdocs build --strict` succeeded
  with a nonfatal future-MkDocs-version warning; evidence-ledger validation
  succeeded (107 capabilities); targeted Ruff check and format check passed
  for both modified release scripts.
- The only full-suite run completed with **2,575 passed, 37 skipped, 1 failed,
  39 warnings** in 1,360.89 seconds. The failure was an existing assertion
  pinning the exact spelling of the readiness-command invocation in
  `tests/test_release_workflow_contract.py`; the release workflow now uses
  `uv run --locked --no-sync`. That assertion was removed rather than
  mechanically re-pinned. The affected contract test then passed in a focused
  run (1 passed). The full suite was not rerun after this test-only correction;
  therefore a full-suite pass is not claimed.
- The graph refresh started after these changes completed: **16,693 nodes,
  39,278 edges, 1,091 communities**; `graph.html` aggregates 1,091 community
  nodes and 2,076 cross-community edges. It backed up six semantic/curated
  graph files under `graphify-out/2026-09-26/`. The updater reported 341
  zero-node source files (mostly JSON) and 227 community renames; it recommends
  `graphify label` for changed names, but I did not run the optional LLM step.
- `docs/release-gates.json` still leaves App-token proof pending. The
  nonpublishing proof has not been dispatched because no candidate was pushed;
  exact-commit remote CI, release dispatch, tag, GitHub Release, and PyPI
  publication remain unobserved. No commit or push was made. The current
  worktree plan remains Task 5 `[~]`; I did not edit its Main-owned checkboxes.
  I notified Main that its note at line 25 still says the App ruleset is open,
  contrary to the accepted active-ruleset evidence, and asked Main to correct
  that note or exclude the plan from the candidate.

## Current candidate continuation (2026-09-26)

This entry supersedes the earlier "no commit or push" snapshot above. Main
corrected the Sprint 81 plan's stale ruleset note and approved including that
plan and the relevant Sprint 80 provenance in the candidate. Task 5 remains
`[~]`; Main owns its status and evidence review.

### Candidate audit and scope

- The candidate changes were reviewed for release scope, license, secrets, and
  user-owned scratch. The project license is MIT. The secret scan found no
  private-key PEM header; App/PyPI credential values were not read or emitted.
  The added diagnostic model arrays are the generated synthetic
  `LinearBrightnessPCAAutoencoder`, not external model weights.
- Candidate selection includes the approved Sprint 81 plan correction,
  Task 4/5/8 handoffs, `SECURITY.md` and `docs/SUPPORT_POLICY.md`, permanent
  Sprint 80 tests, and the substantive replay/provenance/diagnostic bundles.
  The proof-80-23 `...-rerun` directory is excluded because it contains only a
  duplicate model-lesion manifest/array and no run record or diagnostic; it is
  preserved untouched.
- `build/`, root `head_init_dump.py` / `head_init_tmp.py`, Sprint 80 smoke and
  v2 proof scripts, `.agents/memory/` edits, and the two M14 failed-attempt
  audit/exit files are user-owned or generated scratch and are not candidate
  paths. They remain untouched. Staging is still pending at this snapshot.
- README and release-note statements were corrected to match observed state:
  the stable-tag ruleset is active, Task 4's package build/install checks are
  complete, and App-token proof plus the first PyPI upload remain pending.
  M14's obsolete instruction to create a ruleset was replaced with the
  historical rejected-default-actor context and the verified owner App
  configuration. No tag, release dispatch, GitHub Release, or upload occurred.

### Local verification

- `uv run --locked --no-sync pytest -q`: **2,576 passed, 37 skipped, 39
  warnings** in 1,845.60 seconds. No test failed.
- `uv run --locked --no-sync ruff check src/ tests/`: passed.
- `uv run --locked --no-sync ruff format --check src/ tests/`: passed; 329
  files already formatted.
- `uv run --locked --no-sync pyright`: **0 errors, 0 warnings, 0 informations**.
- `uv run --locked --no-sync mkdocs build --strict`: succeeded; documentation
  built in 27.56 seconds. The installed theme emitted its informational
  MkDocs 2.0 compatibility warning; strict build had no errors.
- `uv run --locked --no-sync python scripts/validate_evidence_ledger.py`:
  succeeded; inventory 107 capabilities, core 41/63, overall 41/64.
- The exact local release preflight correctly stopped before publication with
  only `release-app-token-proof (status: pending)` as blocker. The proof has
  not yet run from the pushed default branch.
- Broad `git diff --check` exited 2 solely on trailing-CR whitespace reports
  from the excluded M14 audit/exit files. Candidate-only staged whitespace
  verification remains pending; the excluded files were not changed.

### Remote gates

No candidate commit/push, exact-commit CI run, default-branch App proof, or
release action is claimed yet. The audited paths are staged; next steps are to
commit/push without rewriting history, observe CI for the exact commit, run the
App proof, then update the gate record and release notes. Any release dispatch
remains forbidden until both remote proofs pass.

### Audited staging checkpoint

- Staged **218 files** explicitly: the reviewed candidate source/config/docs,
  the safe nonpublishing App-proof workflow, permanent Sprint 80 tests, Task 4/5/8
  handoffs, and selected diagnostic/replay evidence. The staged diff is
  `+32,195 / -785` lines. No broad `git add` was used.
- The staged name inventory contains no `build/`, head-init dumps, temporary
  smoke/v2 proof scripts, `.agents/memory/` edits, M14 failed-attempt logs, or
  incomplete `proof-80-23...-rerun` directory. Those worktree paths remain
  untouched and unstaged.
- `git diff --cached --check` passed for the staged candidate after excluding
  the Task 5 and Task 8 artifact headers, which intentionally use Markdown
  hard-break spaces. No source, test, release-workflow, or other documentation
  whitespace issue remains. The staged candidate has not yet been committed
  or pushed.

## Pushed candidate and App proof checkpoint (2026-09-26)

- Commit `f87726950d5ae59d4287677f026c39e56baef236` committed 218 audited
  files (`+32,211 / -785`) and pushed successfully to `origin/main` as a normal
  fast-forward from `24850717`; no force push was used.
- Push-triggered CI run
  [36247369268](https://github.com/triet4p/latent-anything/actions/runs/36247369268)
  is for exact head SHA `f87726950d5ae59d4287677f026c39e56baef236`. It was still
  `in_progress` when this checkpoint was written; no CI pass is claimed yet.
- Safe proof run
  [36247385348](https://github.com/triet4p/latent-anything/actions/runs/36247385348)
  completed `success` on `main`, event `workflow_dispatch`, head SHA
  `f87726950d5ae59d4287677f026c39e56baef236` (job completed
  2026-09-26T14:06:48Z). Logs verified App ID `5083983` /
  `latent-anything-release-bot`, the selected `triet4p` installation,
  `contents: write`, and an installation token scoped to exactly
  `triet4p/latent-anything`; the run performed no repository write or release
  operation. Secret and token values were masked and not emitted.
- `docs/release-gates.json` now marks the App-token proof ready with that run
  evidence; the pending PyPI first-upload bootstrap remains separately
  recorded. README, release notes, PLAN, and M14 now reflect the verified App
  proof. These documentation/gate changes are not yet committed.
- The App-token action log emitted a nonblocking deprecation warning for its
  `app-id` input (`client-id` is recommended). No workflow change was made on
  the basis of that warning. There has still been no release-workflow dispatch,
  tag, GitHub Release, or PyPI upload.

### Post-proof documentation checkpoint

- After changing the App gate to `ready`, `uv run --locked --no-sync python
  scripts/check_release_readiness.py` passed for the signed-off bounded
  ordinary-DL core. The release job's CI/package checks still remain required.
- A second `uv run --locked --no-sync mkdocs build --strict` after updating
  README, PLAN, M14, release notes, and the Sprint 81 plan succeeded; MkDocs
  reported a 24.32-second build with only its nonblocking MkDocs 2.0 warning.
- The Sprint 81 plan now records the successful App proof and keeps Task 5
  `[~]` pending Main review. The push-triggered exact candidate CI run
  36247369268 had no observed green conclusion at this checkpoint. No release
  dispatch, tag, GitHub Release, or PyPI upload occurred.

### External configuration recheck

- GitHub API revalidated ruleset `24036634`: active `refs/tags/v*`
  creation/update/deletion/non-fast-forward rules, with only App integration
  `5083983` as `always` bypass.
- PyPI GitHub environment `pypi` (`22803163260`) remains restricted to branch
  `main`, requires reviewer `triet4p`, has `can_admins_bypass=false`, and has
  no environment secrets. The repository-level secret listing exposes only
  the required names; secret values were not read.
- PyPI's project JSON endpoint still returns HTTP 404 before the first OIDC
  upload, consistent with the verified Trusted Publisher configuration and
  expected first-project bootstrap. No upload has occurred.

## Candidate CI and release-preparation checkpoint (2026-09-26)

- Push-triggered CI run
  [36247369268](https://github.com/triet4p/latent-anything/actions/runs/36247369268)
  completed successfully for exact commit
  `f87726950d5ae59d4287677f026c39e56baef236` (event `push`, branch `main`).
  The Python 3.12, 3.13, and 3.14 matrix jobs all passed GPT-2 proof-input
  download, evidence-ledger validation, Ruff, format, Pyright, and Pytest.
- `CHANGELOG.md` now has a dated `1.0.0` section. The release-note extraction
  smoke
  `uv run --locked --no-sync python scripts/extract_release_notes.py v1.0.0 --body-file .gh-pages-build/release-body-preview.md`
  succeeded with version `1.0.0`, expected title, `prerelease=false`, and the
  generated body at the requested path.
- The README, PLAN, M14 validation contract, release notes, and Sprint 81 plan
  record the successful App proof and initial candidate CI while explicitly
  stating that the subsequent changelog/documentation commit still needs
  exact-SHA CI. Task 5 remains `[~]`.
- `uv run --locked --no-sync mkdocs build --strict` passed after these
  documentation changes (28.42 seconds); only the existing informational
  Material-for-MkDocs warning was emitted.
- CI annotations reported Node.js 20 deprecation for existing actions and the
  planned `ubuntu-latest` migration; all jobs nevertheless concluded success.
- No release workflow was dispatched; no `v1.0.0` tag, GitHub Release, or
  PyPI upload has occurred. The release remains blocked until the final
  changelog/documentation commit is pushed and its exact-SHA CI plus guarded
  release-workflow gates pass.
- `graphify update .` completed in 139.89 seconds with a code-only/no-LLM
  refresh: 16,693 nodes, 39,278 edges, 1,097 communities, and 2,103
  cross-community edges. `graph.json`, `graph.html`, and `GRAPH_REPORT.md`
  were updated.
- Graphify reported 341 inputs with zero extracted nodes and noted community
  labels need refresh (1,091 saved labels versus 1,097 current communities).
  It requested `graphify label` for labels and a separate semantic update for
  document changes. Those operations were not run; this result is the code
  graph refresh only, not semantic-document extraction.

## Live 1.0.0 publication (2026-09-26)

- Final candidate `a449ca33b4b83ce109c29db7cf471919820b1d56` committed/pushed as
  `docs(release): finalize 1.0.0 candidate changelog and readiness evidence`
  (8 files, `+193 / -72`, staged by exact name; no scratch staged).
- Exact-SHA CI run
  [36250365395](https://github.com/triet4p/latent-anything/actions/runs/36250365395)
  (`workflow_dispatch`, `main`, head `a449ca3`): completed success; all three
  `lint-and-test (3.12/3.13/3.14)` jobs success, each step table showing
  evidence-ledger, Ruff check, Ruff format check, Pyright, and Pytest success.
- Audited release run
  [36251907256](https://github.com/triet4p/latent-anything/actions/runs/36251907256)
  (`workflow_dispatch`, `main`, head `a449ca3`, input `tag=v1.0.0`): completed
  success. All five jobs success: gate-and-build, attest-artifacts,
  create-approved-release-tag, publish GitHub Release, publish PyPI.
  The PyPI job ran only after owner approval of the pending `pypi`
  (`22803163260`) deployment; no protection was altered.
- Tag: remote `refs/tags/v1.0.0` = `3bdb7ff2c5e09def32e39b9b4628750729104b22`,
  annotated dereference `a449ca33b4b83ce109c29db7cf471919820b1d56`.
- GitHub Release `v1.0.0` (`Latent Anything 1.0.0 - Core latent-space framework`,
  non-draft/non-prerelease, `targetCommitish: main`, created `2026-09-26T15:43:40Z`)
  carries 5 assets: wheel (`635199` bytes,
  `sha256:3f7081d4cb8994c6a719d85d51a3c0ccff76171673c5a1dc33737d7770b408ea`),
  sdist (`886293` bytes,
  `sha256:36b6b3950fd791cf9ffa5eeac79050c5052d5b9902d1c98a9a2b7a6d38e8e3a9`),
  `PROVENANCE.json`, `release-body.md`, `SHA256SUMS`. Downloaded
  `PROVENANCE.json` records `tag: v1.0.0`, `commit: a449ca3…`,
  `workflow_run: 36251907256`, `repository: triet4p/latent-anything`, with both
  distribution digests matching `SHA256SUMS`.
- PyPI `https://pypi.org/pypi/latent-anything/json` now returns HTTP 200,
  `version: 1.0.0`, with both filenames and SHA-256 digests byte-identical to
  the GitHub Release assets (verified GH-download vs PyPI JSON match).
- `gh attestation verify --owner triet4p` passed (exit 0) for both downloaded
  distributions. Verification scratch was removed.

When this live-release appendix was first written, the PyPI metadata and documentation still described the pre-publication candidate. That was the state before this reconciliation, not the current status.

## Post-publication documentation reconciliation (2026-09-26)

- `docs/release-gates.json` now records the PyPI Trusted Publisher as active, with project JSON HTTP 200 and activation evidence from successful OIDC release run 36251907256.
- `README.md`, `docs/RELEASE_NOTES_1.0.0.md`, `docs/PLAN.md`, and `docs/M14_REAL_SYSTEM_VALIDATION.md` now describe the published release and distinguish the tagged release commit `a449ca3` from this later documentation update.
- `docs/sprint-plans/sprint-81.md` records the publication and pending Task 5 evidence review without changing any task checkbox. Tasks 6, 7, and 9 remain outside this correction.
- `CHANGELOG.md` already recorded the `1.0.0` release date in the release commit and is unchanged.

Task 5 remains `[~]` pending Main's evidence review.

## Focused verification

- `uv run --locked --no-sync python scripts/check_release_readiness.py` — passed for the signed-off bounded ordinary-DL core.
- `uv run --locked --no-sync pytest -q tests/test_release_readiness.py tests/test_release_workflow_contract.py` — 22 passed.
- `uv run --locked --no-sync mkdocs build --strict` — passed in 43.02 seconds; output included only the informational Material warning.
- No tag, dispatch, release update, or PyPI upload/publication action was performed in this correction.

## Graph update

- `graphify update .` refreshed the code graph; it re-extracted 344/344 code files and reported no code-topology changes (341 zero-node inputs were informational).
- The semantic document merge refreshed the seven changed sources, renamed 13 stale publication-state nodes, added current run/PyPI evidence, and left zero dangling edges. The final graph has 16,699 nodes, 39,288 edges, and 37 hyperedges; all seven sources have semantic manifest hashes.
- `graphify cluster-only .` completed and regenerated `GRAPH_REPORT.md` and `graph.html` for 1,109 communities. The initial relabel pass auto-renamed 646 labels after detecting 1,096 saved labels; the final rerun was stable. No LLM labeling pass was run.
