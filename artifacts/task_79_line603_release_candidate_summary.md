# Sprint 79 `0.9.0` release-candidate summary (source-committed, NOT tagged/published)

## Candidate status

- This is a **source-committed release candidate only**. No `v0.9.0` tag, GitHub
  Release, or PyPI publication was created, and none is claimed.
- Proposed tag after review: **`v0.9.0`** (expected GitHub Release title:
  `Latent Anything 0.9.0 - Core latent-space framework`, non-prerelease).
- Sprint 79 publication task (line 604) remains **pending** until this candidate
  passes evidence review with no actionable finding.
- Sprint 80 (diagnostic depth) and Sprint 81 (`1.0.0` publication) links resolve:
  `docs/sprint-plans/sprint-80.md`, `docs/sprint-plans/sprint-81.md`.

## Version consistency

- Authoritative current version: **`0.9.0`** in `pyproject.toml`,
  `src/latent_anything/__init__.py` (`__version__`), and `uv.lock`.
- Current compatibility snapshot: `artifacts/api_freeze_snapshot_0.9.0.json`; file
  SHA-256 `2d32c2955826d2fb83133575e91daef0749aabad015c6f2a553a44b6c02344df`.
  `uv run python scripts/api_freeze_snapshot.py --check` reports the frozen API
  digest `048ac553adabb11c24d3e1f4d86e0c6d469df6590064014c915d6a08c7d53c26`
  and exits 0.
- Historical evidence preserved unchanged: `v0.1.0-beta.1` tag,
  `CHANGELOG.md` `0.1.0-beta.1` section,
  `artifacts/api_freeze_snapshot_0.1.0b1.json` (SHA-256
  `48d64721b73a9d0c9e73da4a41940008c70dfa7841e500bc11bc8dcd22ddf7f6`),
  and `artifacts/release_notes_0.1.0-beta.1.md`.
- Compatibility aliases retained: all 18 rows verified by focused suites; the
  `method_a`/`method_b` construction warning now explicitly defers removal past
  `0.9.0` pending a separate reviewed migration decision.

## Changed paths

- Planning/ADR revision committed with the candidate: `.agents/memory/decisions.md`,
  `docs/PLAN.md`, `docs/sprint-plans/sprint-79.md`,
  `docs/sprint-plans/sprint-80.md`, `docs/sprint-plans/sprint-81.md` (new).
- Version surfaces: `pyproject.toml`, `src/latent_anything/__init__.py`,
  `uv.lock`, `tests/test_latent_anything/test_package.py`.
- Contract/docs: `scripts/api_freeze_snapshot.py`,
  `src/latent_anything/_api_freeze_runtime.py`,
  `src/latent_anything/registry_aliases.py`,
  `tests/test_api_freeze_snapshot.py`, `tests/test_api_compatibility.py`,
  `docs/MIGRATION.md`, `docs/API_COMPATIBILITY.md`, `docs/API_REFERENCE.md`,
- Release content: `CHANGELOG.md` (`## [0.9.0] - 2026-09-17`),
  `artifacts/release_notes_0.9.0.md` (byte-mirror of the extractor body; 3355
  bytes, SHA-256
  `841060f08f97b8b9e8730e8a8ab5c10047583163cd1fe2e23dafba1449b5ace0`),
  `artifacts/api_freeze_snapshot_0.9.0.json` (new current snapshot).
- Sprint 79 plan: line 603 checked by this candidate commit; line 604 left
  pending.

## Honest 0.9.0 scope (no 1.0 claim)

- Accepted: 205 runtime / 202 canonical exports, 32 registry entries, 5 plugin
  groups, 12 profiles, 5 CLI commands, 28 config schemas, 9 sync/async pairs,
  7 exceptions, versioned serialization envelopes.
- Preserved negative evidence: **41/63 core and 41/64 scoped overall** ledger
  counts; M14 13 accepted / 2 partial / 1 pending / 7 blocked; TCAV/SAE/steering/
  manifold-geometry failures carried as Sprint 80 diagnostic-claim blockers.
- Ceiling: **16 GiB**; **OpenVLA excluded** (historical D0, >=24 GiB BF16
  unavailable); no named 3DGS checkpoint; no real-policy/LeRobot overhead claim.

## Release-gate evidence (executed 2026-09-18; pre-normalization candidate)

- Locked environments: `uv sync --locked` exited 0; `uv sync --locked --extra
  docs` exited 0. The docs extra is the project optional extra, not a dependency
  group.
- Static gates: `uv run ruff check src tests scripts` exited 0 (`All checks
  passed!`); `uv run ruff format --check src tests scripts` exited 0 (`432 files
  already formatted`); `uv run pyright` exited 0 (`0 errors, 0 warnings,
  0 informations`).
- Contract gates: `uv run python scripts/api_freeze_snapshot.py --check` exited
  0 for the pre-normalization candidate with frozen digest
  `d0495cd85fb78b9d2eb9e53bb00052b91f2c5a9cf951ddbc0798c9710d38cee8`; this
  historical result is superseded by the corrected current normalized digest
  `048ac553adabb11c24d3e1f4d86e0c6d469df6590064014c915d6a08c7d53c26`.
  `uv run pytest tests/test_api_freeze_snapshot.py
  tests/test_api_compatibility.py -q` passed **14 tests**; the evidence-ledger
  validator exited 0 with `errors: []`, coverage **41/63 core** and **41/64
  overall**.
- Full test gate: `uv run pytest -v` passed **2215**, skipped **46**, with **39
  warnings**, in **872.54s** (`0:14:32`).
- Documentation gate: `uv run --project latent-anything-theory --group dev
  mkdocs build --strict` exited 0 and built `.gh-pages-build`. The upstream
  Material-for-MkDocs 2.0 warning is informational and did not fail strict mode.
- Packaging gate: `uv build --wheel --sdist --out-dir .release-gate-dist-090`
  exited 0. Built files (recorded before cleanup) were:
  `latent_anything-0.9.0-py3-none-any.whl`, **407162 bytes**, SHA-256
  `18b82bed4520c094c2de41c1f1ab78c1c9a993635d6082371ed2020deac9c117`; and
  `latent_anything-0.9.0.tar.gz`, **565106 bytes**, SHA-256
  `6e43f91cff8e2d4f8f73f82044e31e0282ac26dc5813bb42168e31019a0d36ad`.
- Wheel smoke: a new `.release-gate-wheel-env-090` was created, the wheel was
  installed with `uv pip install`, and
  `./.release-gate-wheel-env-090/Scripts/python.exe -c "import
  latent_anything; ..."` printed the installed site-packages path and
  `version= 0.9.0`; the assertion `latent_anything.__version__ == "0.9.0"`
  passed.
- Sdist smoke: independently, a new `.release-gate-sdist-env-090` was created,
  the sdist was installed with `uv pip install`, and the same import/version
  assertion passed, printing its separate site-packages path and
  `version= 0.9.0`.

All disposable release, build, and site outputs were removed before commit.
The final candidate commit SHA is returned with delivery; this summary is part of
that commit.

## Initial Ruff gate failure and correction scope

- The initial Ruff report recorded **68 errors** (`I001`, `F401`, `F841`, and
  `E501`) across the M14 evidence scripts, runtime SAE evaluation, and M14
  contract test. No behavior or release scope change was required.
- The correction is formatting-only plus removal of the reported unused imports
  and local assignments, covering exactly:
  `scripts/m14_l02_metrics.py`, `scripts/m14_l05_density.py`,
  `scripts/m14_l06_sae.py`, `scripts/m14_l07_interventions.py`,
  `scripts/m14_l09_diffusers_vae.py`, `scripts/m14_l11_gpt2.py`,
  `scripts/m14_l14_tokenized.py`, `scripts/m14_l15_transitions.py`,
  `scripts/m14_l16_mpc.py`, `scripts/m14_l16_planning.py`,
  `scripts/m14_l22_runtime.py`, `scripts/validate_evidence_ledger.py`,
  `src/latent_anything/sae_evaluation.py`,
  `tests/test_scripts/test_validate_evidence_ledger.py`, and
  `tests/test_m14_validation_contract.py`.

- The post-commit gate identified residual import ordering in
  `scripts/m14_l05_density.py` and `scripts/m14_l14_tokenized.py`, plus formatter
  drift in those files, `scripts/m14_l06_sae.py`,
  `scripts/m14_l07_interventions.py`, `scripts/m14_l09_diffusers_vae.py`,
  `scripts/m14_l11_gpt2.py`, and `tests/test_m14_validation_contract.py`.
  The follow-up correction reapplies Ruff's formatter to exactly those six paths
  and Ruff's import fixes to the two ordering paths, without behavior changes.
- Task 604 remains **pending**; no version, tag, publication, or release-scope
  change is included.

Audited precedents and packaging/audit evidence:
`artifacts/task_sprint79_release_audits_summary.md`,
`artifacts/task_79_line602_rc_evidence_report.md`.

## Rollback / stop conditions

- Stop before tag/publication if: any gate above fails; the candidate review
  raises an actionable finding; `api_freeze_snapshot.py --check` drifts; any
  alias is missing; historical beta artifacts differ; or a `v0.9.0` tag/release/
  package already exists unexpectedly.
- Rollback: do not push; delete any accidental local `v0.9.0` tag immediately
  (`git tag -d v0.9.0`), record the incident, and re-review.

## Proof no publication occurred

- `git tag -l "v0.9.0" "0.9.0"` returns empty; only historical `v0.1.0-beta.1`
  exists.
- No `dist/` output remains; disposable `.release-gate-dist-090`,
  `.release-gate-wheel-env-090`, `.release-gate-sdist-env-090`, `build/`, and
  `.gh-pages-build/` outputs were removed before commit. No push, tag, GitHub
  Release, PyPI publish, or publication workflow command was run.
- The candidate commit contains source, docs, snapshot, release notes, and this
  evidence summary only.
