# Sprint 79 task 604 publication evidence

## Outcome

Task 604 is **complete**. The exact merged main commit passed the audited
tag-triggered release workflow, which created the annotated `v0.9.0` tag and
published the five required GitHub Release assets. The workflow gate/build and
publish jobs both passed, and the assets were downloaded independently for
cross-source hash and provenance verification. PyPI remains explicitly
deferred and non-gating for `0.9.0`; no PyPI upload or installation claim is
part of this evidence.

No product runtime code changed. Sprint 80/81 scope remains unchanged.

## Candidate, branch, and annotated tag

- Merged release commit: `75341e4292fcdf1703186c58b626666b4a923c19`
- Parent commits: `e662654775a1f1b85116e8b18a09f3dbdfe441f8` and
  `de4f4ece600a2c53e91d087f6541627d15902d9b`
- Release tree: `de61a4b2c7b1ee8dcd09dec054eb9195d4bfc782`
- Tag: `v0.9.0`
- Tag object: `8e346926dd088af56a06396655cab8c05d530e58`
- Tag type: annotated
- Peeled tag target: `75341e4292fcdf1703186c58b626666b4a923c19`
- Tag message: `v0.9.0 pre-stable baseline`
- Tag URL: https://github.com/triet4p/latent-anything/releases/tag/v0.9.0

The remote and local provenance checks were:

```text
git ls-remote origin refs/tags/v0.9.0
8e346926dd088af56a06396655cab8c05d530e58 refs/tags/v0.9.0

git ls-remote origin refs/tags/v0.9.0^{}
75341e4292fcdf1703186c58b626666b4a923c19 refs/tags/v0.9.0^{}

git cat-file -t v0.9.0
tag

git rev-parse v0.9.0^{commit}
75341e4292fcdf1703186c58b626666b4a923c19
```

## Audited release workflow

- Workflow: `Release`
- Run ID: `35516933525`
- Run URL: https://github.com/triet4p/latent-anything/actions/runs/35516933525
- Event/ref/SHA: `push`, `v0.9.0`,
  `75341e4292fcdf1703186c58b626666b4a923c19`
- Run conclusion: **success**
- Gate/build job: `Release gate and reproducible package build`
- Gate/build job URL:
  https://github.com/triet4p/latent-anything/actions/runs/35516933525/job/106094349102
- Gate/build conclusion: **success**
- Publish job: `Publish verified GitHub Release`
- Publish job URL:
  https://github.com/triet4p/latent-anything/actions/runs/35516933525/job/106095232699
- Publish conclusion: **success**

Every gate/build step passed: tag/SHA provenance, environment sync, Ruff
check/format, Pyright, Pytest, release-note extraction, deterministic wheel
and sdist build, sdist normalization, package validation, checksums,
provenance, and immutable build-evidence upload. Every publish step passed:
artifact download, provenance/checksum verification, idempotent GitHub Release
creation/update, and post-upload asset download/hash verification.

The retained workflow artifact is:

- Name: `release-v0.9.0-35516933525`
- Artifact ID: `10606937525`
- Download URL:
  https://api.github.com/repos/triet4p/latent-anything/actions/artifacts/10606937525/zip
- Contents: the exact five files listed below.

## GitHub Release metadata and exact assets

- Release URL: https://github.com/triet4p/latent-anything/releases/tag/v0.9.0
- Name: `Latent Anything 0.9.0 - Core latent-space framework`
- Tag name: `v0.9.0`
- Draft: `false`
- Prerelease: `false`
- Published: `2026-09-20T14:42:00Z`
- GitHub API `target_commitish`: `main`
- Asset count: **5**

The workflow artifact and independently downloaded GitHub Release directory
each contained exactly these files. Local and remote bytes matched for every
file:

| File | Bytes | SHA-256 |
|---|---:|---|
| `latent_anything-0.9.0-py3-none-any.whl` | 407224 | `a3667986eb01f96154958d7105e922abc424280aad7ad609fb2da45427300a85` |
| `latent_anything-0.9.0.tar.gz` | 549080 | `529121041a6268ad89a43cd2b3baa1762bfb9133a648784bd39a1a4d22bfa26f` |
| `SHA256SUMS` | 200 | `8515945050b619a6b37f79d71d94c2d7055e62b7d4164c25fe1f21e61c126db8` |
| `PROVENANCE.json` | 601 | `0b50fead137e509a2e31bf014709b65fa4e860b9479dbed19383e7c64e39b5af` |
| `release-body.md` | 3413 | `375abe62c4b43a9f405b2ea58e9abc9b4b21067eb2127580468cfee676452dd5` |

GitHub's asset API reported the same SHA-256 digest for each asset. The
workflow's `SHA256SUMS` matches the wheel and sdist bytes, and
`PROVENANCE.json` records:

```json
{
  "tag": "v0.9.0",
  "commit": "75341e4292fcdf1703186c58b626666b4a923c19",
  "workflow_run": "35516933525",
  "repository": "triet4p/latent-anything"
}
```

Its `files` entries match the wheel/sdist names, byte sizes, and hashes above.

## Clean GitHub-wheel installation smoke

The release assets were downloaded into a new isolated environment. The wheel
was installed by local file path with `--no-deps`, then dependencies were
resolved only from the existing local uv cache with `--offline`; no PyPI
fallback was permitted:

```text
uv venv .release-verification/venv --python 3.13
uv pip install --python .release-verification/venv/Scripts/python.exe \
  --no-deps .release-verification/release/latent_anything-0.9.0-py3-none-any.whl
uv pip install --python .release-verification/venv/Scripts/python.exe \
  --offline .release-verification/release/latent_anything-0.9.0-py3-none-any.whl
RELEASE_GITHUB_WHEEL_IMPORT_OK 0.9.0
```

The final command imported `latent_anything` and asserted
`latent_anything.__version__ == "0.9.0"`.

## Cross-source integrity and plan state

Integrity result: **PASS**. The annotated tag, merged commit, workflow
provenance, retained workflow artifact, GitHub Release asset bytes, API
digests, checksums, and clean local-file installation all agree. PyPI remains
absent by deliberate policy and is not a release gate.

- `docs/sprint-plans/sprint-79.md` task 604 is `[x]`; Sprint 79 publication
  and closure evidence are complete.
- `docs/PLAN.md` records the completed Sprint 79/0.9.0 publication while
  Milestone 14 remains open for its unchanged Sprint 80 and Sprint 81 work.
- `CHANGELOG.md`, `README.md`, and `docs/MIGRATION.md` identify the verified
  GitHub Release assets as the 0.9.0 distribution channel and defer PyPI.
- Closure branch/PR evidence is recorded separately; no release tag mutation,
  release rerun, or task closure is performed by the closure PR.

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
- The historical build job uploaded exact archives and provenance as a
  retained artifact. Its dependent publish job then attempted the old PyPI
  path before updating the GitHub Release. This ordering was superseded by
  the current GitHub-only publish job, which uploads and verifies the exact
  five assets without PyPI.
- The old PyPI publication used `pypa/gh-action-pypi-publish@release/v1` with
  `id-token: write`; that path and permission were removed by the reviewed
  workflow change. The historical polling records below are retained only as
  evidence for why PyPI is deferred.
- The historical `workflow_dispatch` input accepted an existing tag. The
  current successful publication was the single authorized tag-triggered run
  recorded above.

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

The historical PyPI endpoint returned HTTP `404`, and that historical GitHub
Release had zero assets. This is superseded by the successful tag-triggered
run and five-asset release documented above. Task 604, Sprint 79 publication,
and the GitHub Release evidence are now complete; PyPI remains non-gating.
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

The historical public-install attempt failed because PyPI was intentionally
absent at that time. It is superseded by the successful clean GitHub-wheel
installation smoke recorded above; task 604 and Sprint 79 are complete.
The historical GitHub Release had no assets and the historical cross-source
result was **PARTIAL / BLOCKED**. That record is superseded by the successful
five-asset publication and **PASS** integrity result above. Task 604 and
Sprint 79 are complete; Milestone 14 remains open only for Sprint 80/81.
This canceled dispatch is superseded integration work, not a product or test
failure. The later default-branch tag-triggered run completed successfully;
task 604 and Sprint 79 are complete, while Sprint 80/81 remain unchanged.
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

The historical GitHub Release had no assets and the historical cross-source
result was **PARTIAL / BLOCKED**. That record is superseded by the successful
five-asset publication and **PASS** integrity result above. Task 604 and
Sprint 79 are complete; Milestone 14 remains open only for Sprint 80/81.
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
This cancellation is superseded integration work, not a product or test
failure. The later default-branch tag-triggered run completed successfully;
task 604 and Sprint 79 are complete, while Sprint 80/81 remain unchanged.
