# Sprint 79 task 604 publication evidence

## Outcome

Task 604 remains **open**. The authorized candidate was retagged exactly and
the audited `Release` workflow completed successfully, creating a public
GitHub Release. The required PyPI publication did not occur: the version JSON
still returns HTTP 404, the audited workflow has no build/upload step, and no
publication credentials or alternate publication route is available in this
environment. Because the external PyPI step failed, Sprint 79 and Milestone 14
remain open. No product code or Sprint 80/81 scope was changed.

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

The workflow at the candidate commit only runs validation, extracts release
notes, and calls `softprops/action-gh-release`; it does not build, checksum,
attach, or upload a wheel/sdist. Consequently there is no workflow-produced
package artifact path or workflow artifact hash to claim. GitHub's generated
source archives were downloaded only as release-page verification (they are
not release assets and are not Python distribution artifacts):

| URL | Bytes | SHA-256 |
|---|---:|---|
| https://github.com/triet4p/latent-anything/archive/refs/tags/v0.9.0.tar.gz | 12737606 | `9033b9ae30d48a7855266c22e2c5f266ebfa75aec5e7a36ae5cced681c6c4f6f` |
| https://github.com/triet4p/latent-anything/archive/refs/tags/v0.9.0.zip | 13942160 | `e6e87f333fb18307a14cbdef8c7f75c6fbecc8c8f0f52bfe102a1e96e6dcb70d` |

The published release body is the extracted `0.9.0` changelog body. Its
historical candidate-status sentence still says that no tag/release exists;
the current immutable release/tag state above and this evidence supersede
that stale pre-publication sentence.

## Package hashes and PyPI state

Historical candidate-local build hashes (not published-file hashes and not
recomputed or uploaded here) were:

| File | Candidate-local SHA-256 |
|---|---|
| `latent_anything-0.9.0-py3-none-any.whl` | `18b82bed4520c094c2de41c1f1ab78c1c9a993635d6082371ed2020deac9c117` |
| `latent_anything-0.9.0.tar.gz` | `6e43f91cff8e2d4f8f73f82044e31e0282ac26dc5813bb42168e31019a0d36ad` |

Verification commands and results:

```text
curl -sS -o pypi-0.9.0.json -w '%{http_code}' \
  https://pypi.org/pypi/latent-anything/0.9.0/json
404
```

- PyPI project/version URL:
  https://pypi.org/project/latent-anything/0.9.0/
- PyPI JSON URL: https://pypi.org/pypi/latent-anything/0.9.0/json
- PyPI files and hashes: **none; version endpoint is HTTP 404**.
- No `UV_PUBLISH_*`, `TWINE_*`, or `PYPI_*` environment credentials were
  present; `gh secret list --repo triet4p/latent-anything` returned no secrets.
- No PyPI upload command was run. The audited workflow contains no PyPI
  publication step.

## Cross-source integrity and install smoke

Integrity result: **PARTIAL / BLOCKED**. The exact candidate commit is proven
through the remote lightweight tag, workflow checkout log, successful workflow
run, and GitHub Release tag. There is no workflow-produced wheel/sdist,
no GitHub Release package asset, and no PyPI file to hash or compare. Thus a
cross-source package-hash/provenance match cannot be established and no
published-package claim is made.

The required brand-new isolated PyPI smoke was attempted:

```text
rm -rf .publication-pypi-smoke
uv venv .publication-pypi-smoke --python 3.13
uv pip install --python .publication-pypi-smoke/Scripts/python.exe \
  latent-anything==0.9.0
rm -rf .publication-pypi-smoke
```

It failed as expected from the missing publication:

```text
No solution found when resolving dependencies:
Because latent-anything was not found in the package registry and you require
latent-anything==0.9.0, we can conclude that your requirements are
unsatisfiable.
```

No published clean-install import/version output exists. The earlier
candidate-local wheel/sdist smokes remain historical only: each imported
`latent_anything` and asserted `latent_anything.__version__ == "0.9.0"`, but
they do not satisfy published-distribution evidence.

## Plan state and final branch

- `docs/sprint-plans/sprint-79.md` task 604 remains `[ ]`; Sprint 79 remains
  open because PyPI publication and published-package smoke failed with the
  exact HTTP 404/absent-publication blocker above.
- `docs/PLAN.md` Milestone 14 remains `[ ]`; Sprint 79 is not moved to the
  completed-sprints section. Sprint 80/81 scope is unchanged.
- `CHANGELOG.md` and `docs/MIGRATION.md` record the partial state: the exact
  GitHub Release exists, while PyPI publication remains pending.
- Evidence/planning commit: `811def8` (`docs(release): record 0.9.0
  publication blocker`). The final metadata commit SHA and branch push result
  are returned with delivery; the worktree is verified clean.

## Release-path remediation (not executed)

The existing GitHub Release remains unchanged and no tag was moved or deleted
during this remediation. The audited path was repaired in
`.github/workflows/release.yml` but was **not triggered**:

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

External prerequisite before dispatching this repaired path: configure PyPI
Trusted Publishing for project `latent-anything` with owner `triet4p`,
repository `latent-anything`, workflow filename
`.github/workflows/release.yml`, and no environment (or an explicitly
configured matching GitHub environment). Because the immutable tag points to
the pre-remediation workflow, the first authorized retry must dispatch the
repaired workflow from the branch carrying this fix while selecting the
existing tag:

```text
gh workflow run release.yml --repo triet4p/latent-anything \
  --ref sprint79-local-gate-remediation -f tag=v0.9.0
```

This command was **not** run. Until that publisher is configured, the workflow
must not be dispatched; task 604/Sprint 79/Milestone 14 remain open.
