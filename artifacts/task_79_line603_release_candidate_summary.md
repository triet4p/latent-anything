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
- Current compatibility snapshot: `artifacts/api_freeze_snapshot_0.9.0.json`,
  SHA-256 `d0495cd85fb78b9d2eb9e53bb00052b91f2c5a9cf951ddbc0798c9710d38cee8`,
  `--check` clean.
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
  `docs/M14_REAL_SYSTEM_VALIDATION.md`, `docs/sprint-plans/sprint-78.md`,
  `README.md`.
- Release content: `CHANGELOG.md` (`## [0.9.0] - 2026-09-17`),
  `artifacts/release_notes_0.9.0.md` (byte-mirror of the extractor body),
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

## Expected artifacts after review (not yet created)

- `git tag v0.9.0` pushed through `.github/workflows/release.yml`, producing a
  GitHub Release from the `## [0.9.0]` changelog section.
- Clean-checkout `latent_anything-0.9.0-py3-none-any.whl` and
  `latent_anything-0.9.0.tar.gz` with recorded sizes/hashes.
- Clean-environment wheel/sdist install/import evidence and post-publication
  install/link/plugin/example checks.

## Release-gate commands (final validation; NOT run in this pass)

```text
uv sync --locked
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pyright
uv run pytest -v
uv run --project latent-anything-theory --group dev mkdocs build --strict
uv run python scripts/validate_evidence_ledger.py --json
uv run python scripts/api_freeze_snapshot.py --check
uv run pytest tests/test_api_freeze_snapshot.py tests/test_api_compatibility.py -q
uv build --wheel --sdist --out-dir .release-gate-dist
```

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
- No `dist/` output was created in this pass; no push, release, or publish
  command was run.
- This candidate commit contains source, docs, snapshot, changelog, and release
  notes only.
