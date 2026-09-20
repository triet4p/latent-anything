# Sprint 79 task 604 publication evidence

Task 604 remains **open**. This artifact records the historical publication
attempt before the GitHub-only 0.9.0 release contract was adopted: the gate
passed, but the old workflow had no package assets and its PyPI Trusted
Publishing exchange rejected the OIDC claim. PyPI `0.9.0` remains absent by
policy and is not a blocker. The current task is to publish and verify the
exact GitHub Release assets without mutating the tag or claiming task closure.
No product code or Sprint 80/81 scope is changed.


## Candidate, branch, and tag

- Authorized candidate: `2356c92d02022c02be25cd0c79944d07a74b6ca9`
- Candidate parent: `c3fb0852d82dbda1cb424ed618d00e5dbd963477`
- Candidate tree: `e833252570612ab8cf65f80c80e10a15e1092974`
- Candidate subject: `docs: correct 0.9 snapshot digest references`
- Remote branch: `sprint79-local-gate-remediation`
- Candidate branch URL:
  https://github.com/triet4p/latent-anything/tree/sprint79-local-gate-remediation
- Candidate branch was already at the authorized SHA before publication.

The failed old tag was removed once, then recreated as the repository's
lightweight-tag convention:

```text
git tag -d v0.9.0
git push origin :refs/tags/v0.9.0
git tag v0.9.0 2356c92d02022c02be25cd0c79944d07a74b6ca9
git push origin refs/tags/v0.9.0
```

The push output was:

```text
Deleted tag 'v0.9.0' (was 47f4486)
 - [deleted]         v0.9.0
 * [new tag]         v0.9.0 -> v0.9.0
```

Tag provenance verification:

```text
git ls-remote origin refs/tags/v0.9.0
2356c92d02022c02be25cd0c79944d07a74b6ca9 refs/tags/v0.9.0

git cat-file -t v0.9.0
commit

git rev-parse v0.9.0^{commit}
2356c92d02022c02be25cd0c79944d07a74b6ca9

git rev-parse v0.9.0
2356c92d02022c02be25cd0c79944d07a74b6ca9
```

Therefore the exact tag type is **lightweight**, its ref/object is the commit
`2356c92d02022c02be25cd0c79944d07a74b6ca9`, and its commit target is exactly
the authorized candidate.

## Audited release workflow

- Workflow: `Release`
- Workflow file at the tagged candidate:
  https://github.com/triet4p/latent-anything/blob/2356c92d02022c02be25cd0c79944d07a74b6ca9/.github/workflows/release.yml
- Trigger: one push of `refs/tags/v0.9.0`
- Run ID: `35380189555`
- Run URL: https://github.com/triet4p/latent-anything/actions/runs/35380189555
- Event/ref/SHA: `push`, `v0.9.0`,
  `2356c92d02022c02be25cd0c79944d07a74b6ca9`
- Run conclusion: **success**
- Job ID: `105714256827`
- Job: `Release gate and GitHub Release`
- Job conclusion: **success**
- Job interval: `2026-09-18T18:26:31Z` to `2026-09-18T18:32:03Z`

The terminal job/step observation command was:

```text
gh run watch 35380189555 --repo triet4p/latent-anything --interval 30 --exit-status
```

Every job step concluded successfully: Set up job; Checkout; Set up Python;
Install uv; Sync environment; Ruff check; Ruff format check; Pyright; Pytest;
Extract release notes; Create GitHub Release; all post steps; Complete job.
The remote test log recorded:

```text
collected 2259 items / 2 skipped
2194 passed, 67 skipped, 39 warnings in 258.19s (0:04:18)
```

The only annotations were informational Node.js 20 deprecation and a future
`ubuntu-latest` image migration notice; neither changed the successful
conclusion.

## GitHub Release and archive hashes

- Release URL: https://github.com/triet4p/latent-anything/releases/tag/v0.9.0
- API verification:
  `gh api repos/triet4p/latent-anything/releases/tags/v0.9.0`
- Name: `Latent Anything 0.9.0 - Core latent-space framework`
- Tag name: `v0.9.0`
- Draft: `false`
- Prerelease: `false`
- Published: `2026-09-18T18:31:58Z`
- GitHub API `target_commitish`: `main` (the tag's immutable commit
  provenance was independently verified above)
- GitHub Release assets: **none** (`assets: []`).

The historical workflow at the candidate commit only ran validation, extracted
release notes, and called `softprops/action-gh-release`; it did not build,
checksum, attach, or upload Python distribution assets. This is superseded by
the current workflow contract, which builds one deterministic wheel and sdist
after the release gate, emits checksums and provenance, attaches those exact
files plus release notes, and downloads the release assets again to verify their
SHA-256 hashes.

The published release body in that historical run was the extracted `0.9.0`
changelog body. Its candidate-status sentence is retained as historical
evidence; the current release notes define the GitHub-only distribution policy.
## GitHub asset installation evidence

The prior isolated PyPI smoke was intentionally not a successful publication
check: PyPI was absent and is now explicitly deferred for `0.9.0`. The
required verification path is instead a clean environment using the exact
wheel downloaded from the GitHub Release:

```text
gh release download v0.9.0 --repo triet4p/latent-anything --dir release-assets
sha256sum --check release-assets/SHA256SUMS
uv venv .publication-github-smoke --python 3.13
uv pip install --python .publication-github-smoke/Scripts/python.exe \
  release-assets/latent_anything-0.9.0-py3-none-any.whl
```

The smoke must import `latent_anything` and assert
`latent_anything.__version__ == "0.9.0"`. No PyPI installation claim is made;
the GitHub asset and checksum verification are the release evidence.

## Historical package hashes and deferred PyPI state

Historical candidate-local build hashes (not published-file hashes) were:

| File | Candidate-local SHA-256 |
|---|---|
| `latent_anything-0.9.0-py3-none-any.whl` | `18b82bed4520c094c2de41c1f1ab78c1c9a993635d6082371ed2020deac9c117` |
| `latent_anything-0.9.0.tar.gz` | `6e43f91cff8e2d4f8f73f82044e31e0282ac26dc5813bb42168e31019a0d36ad` |

The historical PyPI endpoint returned HTTP `404`; this is expected under the
current deferred, non-gating policy. No PyPI upload command is part of the
audited workflow.

## Cross-source integrity and install smoke

The historical integrity result was **PARTIAL / BLOCKED** because that
workflow produced no package assets. The current workflow closes this evidence
gap by attaching the deterministic build outputs to GitHub Release and
re-downloading every asset for SHA-256 comparison. The clean-install smoke
must use that downloaded GitHub wheel and verify the package version.

## Plan state and final branch

- `docs/sprint-plans/sprint-79.md` task 604 remains `[ ]`; Sprint 79 remains
  open until the exact GitHub Release assets, their hashes, and the clean
  GitHub-asset install are recorded. PyPI absence is expected and non-gating.
- `docs/PLAN.md` Milestone 14 remains `[ ]`; Sprint 79 is not moved to the
  completed-sprints section. Sprint 80/81 scope is unchanged.
- `CHANGELOG.md`, `README.md`, and `docs/MIGRATION.md` define GitHub Release
  assets as the 0.9.0 distribution channel and explicitly defer PyPI.
- The final evidence/planning commit SHA and branch push result are returned
  with delivery; the worktree is verified clean.

## Previous release-path remediation (historical failed publisher exchange)

The following records are retained solely to explain why the OIDC/PyPI path was
removed. They are historical evidence, not a current prerequisite or closure
criterion. The current workflow has no PyPI upload, no OIDC permission, and no
PyPI verification step.

No tag was moved or deleted during the historical remediation. The old repaired
path was dispatched once from the remediation branch and reached the trusted
publisher exchange; PyPI rejected the OIDC claim before any package or GitHub
Release mutation:

- `gate-and-build` now performs the existing lint/type/test gate first, checks
  that the selected ref is exactly `v0.9.0`/`0.9.0` and that the checked-out
  commit is the tag target, builds the wheel and sdist, normalizes sdist
  metadata using the tagged commit epoch, validates archive paths and
  `Name`/`Version` metadata, and writes `SHA256SUMS` plus
  `PROVENANCE.json` containing tag, commit, workflow run, repository, and file
  sizes/hashes.
- The build job uploads the exact archives and provenance as a retained
  `actions/upload-artifact@v4` artifact. A dependent `publish` job downloads
  that artifact, verifies provenance and hashes, then publishes to PyPI before
  updating the GitHub Release. This ordering means a failed test/build cannot
  create or mutate the release, while `skip-existing`, `overwrite_files`, and
  `fail_on_unmatched_files` make authorized reruns safe and explicit.
- PyPI publication uses `pypa/gh-action-pypi-publish@release/v1` with
  `id-token: write` and no long-lived token. After upload, the workflow polls
  PyPI JSON and fails unless both published file digests equal the built
  `SHA256SUMS`; those verification records are also attached to the GitHub
  Release.
- A `workflow_dispatch` input accepts an existing tag so the already-created
  `v0.9.0` Release can be repaired without moving/deleting the tag or creating
  a second release.

Focused dry-run/static proof on the repaired workflow:

```text
python -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml', encoding='utf-8')); print('RELEASE_WORKFLOW_YAML_OK')"
RELEASE_WORKFLOW_YAML_OK

RELEASE_WORKFLOW_STATIC_CONTRACT_OK

SOURCE_DATE_EPOCH=<tagged commit epoch> uv build --wheel --sdist --out-dir <dry-run-dir>
RELEASE_PACKAGE_CONTENT_METADATA_OK
```

Two local builds with the same source epoch produced identical wheel bytes;
normalizing the sdist archive's gzip/tar metadata also produced identical
sdist bytes (`NORMALIZED_SDIST_DETERMINISTIC_OK`). A fresh wheel install in a
new Python 3.13 uv environment printed
`RELEASE_PACKAGE_IMPORT_SMOKE_OK 0.9.0` and asserted
`latent_anything.__version__ == "0.9.0"`. The dry-run directories were
removed. No project-wide lint/test/type/build suite was rerun.

The old dispatch depended on configuring PyPI Trusted Publishing for project
`latent-anything`. That prerequisite was not available, and the publisher
exchange failed before any publication. This historical failure is the reason
the current 0.9.0 contract defers PyPI and uses only the verified GitHub
Release assets.

## Remediation run evidence

- Dispatch command:
  `gh workflow run release.yml --repo triet4p/latent-anything --ref
  sprint79-local-gate-remediation -f tag=v0.9.0`
- Run: https://github.com/triet4p/latent-anything/actions/runs/35383469463
- Run ID: `35383469463`; event `workflow_dispatch`; workflow head
  `b8e31f7ee05e1c4dbef341644b57740f07f563a6`
- Gate job:
  `Release gate and reproducible package build`, job ID `105724866960`,
  conclusion **success**. Every gate/build/provenance step passed, including
  Pytest, archive normalization, package validation, and artifact upload.
- Publish job:
  `Publish verified distributions`, job ID `105726464401`, conclusion
  **failure** at `Publish to PyPI with trusted publishing`; no later PyPI
  verification or GitHub Release update step ran.
- Exact external error:

```text
Trusted publishing exchange failure:
Token request failed: invalid-publisher: valid token, but no corresponding
publisher (Publisher with matching claims was not found).
```

The OIDC claims identify the required configuration:

```text
sub: repo:triet4p/latent-anything:ref:refs/heads/sprint79-local-gate-remediation
repository: triet4p/latent-anything
workflow_ref: triet4p/latent-anything/.github/workflows/release.yml@refs/heads/sprint79-local-gate-remediation
environment: MISSING
```

The immutable build-evidence artifact is
`release-v0.9.0-35383469463`, artifact ID `10563084815`,
https://api.github.com/repos/triet4p/latent-anything/actions/artifacts/10563084815/zip.
Its `PROVENANCE.json` records tag `v0.9.0`, candidate commit
`2356c92d02022c02be25cd0c79944d07a74b6ca9`, repository
`triet4p/latent-anything`, and workflow run `35383469463`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `latent_anything-0.9.0-py3-none-any.whl` | 406993 | `2d709eef8570df2ce3e6c1426823b8cbd7e24e557b27825c231cea2c05d143c8` |
| `latent_anything-0.9.0.tar.gz` | 547990 | `bda07b2911df49a0427bc5886c625384ae7858efa50c597667a027f4780eee83` |

PyPI remained HTTP `404` at
https://pypi.org/pypi/latent-anything/0.9.0/json. The GitHub Release remained
non-draft/non-prerelease with **zero assets**, so no published-file hash or
clean-install smoke can be claimed. Task 604, Sprint 79, and Milestone 14
remain open pending PyPI Trusted Publisher configuration; no token secret or
local upload was attempted.
## Final authorized retry (publisher configuration mismatch)

- Dispatch command (run exactly once):
  `gh workflow run release.yml --repo triet4p/latent-anything --ref
  sprint79-local-gate-remediation -f tag=v0.9.0`
- Run: https://github.com/triet4p/latent-anything/actions/runs/35384704917
- Run ID: `35384704917`; event `workflow_dispatch`; workflow head
  `f856fa2a1567935b780568115a08f73aa4d80b3f`; status **completed /
  failure**.
- Gate job: `Release gate and reproducible package build`, ID
  `105728781246`, URL
  https://github.com/triet4p/latent-anything/actions/runs/35384704917/job/105728781246,
  conclusion **success**. Every gate/build step passed, including Pytest,
  archive normalization, provenance validation, and artifact upload.
- Publish job: `Publish verified distributions`, ID `105731355079`, URL
  https://github.com/triet4p/latent-anything/actions/runs/35384704917/job/105731355079,
  conclusion **failure**. `Publish to PyPI with trusted publishing` failed;
  PyPI verification and GitHub Release update were skipped.

Exact publisher error:

```text
Trusted publishing exchange failure:
Token request failed: the server refused the request for the following reasons:
* invalid-publisher: valid token, but no corresponding publisher
  (Publisher with matching claims was not found)
```

OIDC claims observed by the failed publish step:

```text
sub: repo:triet4p/latent-anything:ref:refs/heads/sprint79-local-gate-remediation
repository: triet4p/latent-anything
repository_owner: triet4p
repository_owner_id: 119602471
workflow_ref: triet4p/latent-anything/.github/workflows/release.yml@refs/heads/sprint79-local-gate-remediation
job_workflow_ref: triet4p/latent-anything/.github/workflows/release.yml@refs/heads/sprint79-local-gate-remediation
ref: refs/heads/sprint79-local-gate-remediation
environment: MISSING
```

The immutable build-evidence artifact is
`release-v0.9.0-35384704917`, artifact ID `10563387928`,
digest
`sha256:57b3edc2ba06d7af5c5838eecfb738c2daff509ce2d3d97a8d1095d1338e010c`,
and API URL
https://api.github.com/repos/triet4p/latent-anything/actions/artifacts/10563387928/zip.
Its `PROVENANCE.json` records tag `v0.9.0`, candidate commit
`2356c92d02022c02be25cd0c79944d07a74b6ca9`, repository
`triet4p/latent-anything`, and workflow run `35384704917`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `latent_anything-0.9.0-py3-none-any.whl` | 406993 | `2d709eef8570df2ce3e6c1426823b8cbd7e24e557b27825c231cea2c05d143c8` |
| `latent_anything-0.9.0.tar.gz` | 547990 | `bda07b2911df49a0427bc5886c625384ae7858efa50c597667a027f4780eee83` |

Artifact `SHA256SUMS` exactly contains the two rows above. The retry
`PROVENANCE.json` has `source_date_epoch=1789728831` and the exact tagged
candidate commit. Tag API verification is
https://api.github.com/repos/triet4p/latent-anything/git/ref/tags/v0.9.0:
the lightweight ref resolves directly to
`2356c92d02022c02be25cd0c79944d07a74b6ca9`.

External publication evidence after the failed job:

- GitHub Release:
  https://github.com/triet4p/latent-anything/releases/tag/v0.9.0
  (`Latent Anything 0.9.0 - Core latent-space framework`, non-draft,
  non-prerelease, published `2026-09-18T18:31:58Z`, `targetCommitish=main`,
  `assets=[]`).
- PyPI JSON: https://pypi.org/pypi/latent-anything/0.9.0/json returned HTTP
  `404`; no public file hashes exist.
- Cross-source integrity: **PARTIAL / BLOCKED**. Candidate/tag/workflow/
  artifact provenance is internally consistent, but no PyPI package or
  GitHub Release package assets exist to compare.

Brand-new public-PyPI smoke (environment removed after the attempt):

```text
uv venv F:/tmp/task604-pypi-smoke-35384704917 --python 3.13
uv pip install --python F:/tmp/task604-pypi-smoke-35384704917/Scripts/python.exe \
  --index-url https://pypi.org/simple latent-anything==0.9.0
No solution found when resolving dependencies:
Because latent-anything was not found in the package registry and you require
latent-anything==0.9.0, we can conclude that your requirements are
unsatisfiable.
```

No import/version output exists because public installation failed before
installation. Task 604, Sprint 79, and Milestone 14 remain open pending a
Trusted Publisher whose claims exactly match the OIDC block above.
## Latest authorized retry (publisher still rejected)

- Dispatch command (run exactly once for this retry):
  `gh workflow run release.yml --repo triet4p/latent-anything --ref
  sprint79-local-gate-remediation -f tag=v0.9.0`
- Run: https://github.com/triet4p/latent-anything/actions/runs/35398189493
- Run ID: `35398189493`; event `workflow_dispatch`; workflow head
  `c5fb429a17cb11c0998e9ae359590f027c6c6605`; status **completed /
  failure**.
- Gate job: `Release gate and reproducible package build`, ID
  `105771753652`, URL
  https://github.com/triet4p/latent-anything/actions/runs/35398189493/job/105771753652,
  conclusion **success**. All gate/build/provenance steps passed, including
  Pytest and immutable artifact upload.
- Publish job: `Publish verified distributions`, ID `105773500066`, URL
  https://github.com/triet4p/latent-anything/actions/runs/35398189493/job/105773500066,
  conclusion **failure** at `Publish to PyPI with trusted publishing`;
  PyPI hash verification and GitHub Release update were skipped.

Exact new external error:

```text
Trusted publishing exchange failure:
Token request failed: the server refused the request for the following reasons:
* invalid-publisher: valid token, but no corresponding publisher
  (Publisher with matching claims was not found)
```

The failed step rendered the same claims despite the publisher being recreated:

```text
sub: repo:triet4p/latent-anything:ref:refs/heads/sprint79-local-gate-remediation
repository: triet4p/latent-anything
repository_owner: triet4p
repository_owner_id: 119602471
workflow_ref: triet4p/latent-anything/.github/workflows/release.yml@refs/heads/sprint79-local-gate-remediation
job_workflow_ref: triet4p/latent-anything/.github/workflows/release.yml@refs/heads/sprint79-local-gate-remediation
ref: refs/heads/sprint79-local-gate-remediation
environment: MISSING
```

The immutable build-evidence artifact is
`release-v0.9.0-35398189493`, artifact ID `10569961784`, digest
`sha256:3d5e0693d05045e53f58bd46ec7c555b6d30f3d5b4814d30f1ff8e7aa488d75b`,
and API URL
https://api.github.com/repos/triet4p/latent-anything/actions/artifacts/10569961784/zip.
Its `PROVENANCE.json` records tag `v0.9.0`, candidate commit
`2356c92d02022c02be25cd0c79944d07a74b6ca9`, workflow run `35398189493`, and
`source_date_epoch=1789728831`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `latent_anything-0.9.0-py3-none-any.whl` | 406993 | `2d709eef8570df2ce3e6c1426823b8cbd7e24e557b27825c231cea2c05d143c8` |
| `latent_anything-0.9.0.tar.gz` | 547990 | `bda07b2911df49a0427bc5886c625384ae7858efa50c597667a027f4780eee83` |

The GitHub Release remained non-draft/non-prerelease with `assets=[]` at
https://github.com/triet4p/latent-anything/releases/tag/v0.9.0, and PyPI JSON
https://pypi.org/pypi/latent-anything/0.9.0/json still returned HTTP `404`.
No package mutation occurred; cross-source package integrity remains
**PARTIAL / BLOCKED**. Task 604, Sprint 79, and Milestone 14 remain open.
## Superseded branch-dispatch attempt

The final branch-dispatched attempt was superseded by the required
default-branch integration correction and was canceled before publication:

- Run: https://github.com/triet4p/latent-anything/actions/runs/35512923025
- Run ID: `35512923025`; workflow head
  `71340b1aca2d6540bd72a7ba2106ee7e96d20c61`.
- Gate job: `Release gate and reproducible package build`, ID
  `106083905310`, canceled during `Pytest` after the branch-dispatch
  correction; no build artifact or publication mutation occurred.
- Publish job: `Publish verified distributions`, ID `106084643928`, canceled
  before starting.
- This cancellation is **superseded integration work**, not a product or test
  failure. Task 604, Sprint 79, and Milestone 14 remain open pending the
  reviewed workflow integration on the repository default branch.
