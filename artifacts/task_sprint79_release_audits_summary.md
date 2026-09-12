# Sprint 79 release and audit gate summary

## Scope and authority

This artifact covers only Sprint 79's release-gate task: **Run
unit/property/integration tests plus strict docs, packaging, security, license,
and dependency audits**. Commands reuse the repository's existing CI/release
contracts (`.github/workflows/ci.yml`, `.github/workflows/release.yml`,
`mkdocs.yml`, `pyproject.toml`, and the project `uv` lockfile); no parallel gate
convention was introduced.

The local gate ran on Windows `win32`, AMD64, Python 3.13.3, with the checked-in
locked environment. Tool versions were:

- uv 0.9.7
- Ruff 0.15.17
- Pyright 1.1.410 (strict mode from `pyproject.toml`)
- pytest 9.1.0
- MkDocs 1.6.1
- pip-audit 2.10.1
- pip-licenses 5.5.5

## Gate results

Every required gate completed successfully after the one source-format
correction recorded below:

| Area | Exact command | Result |
|---|---|---|
| Locked environment | `uv sync --locked` | Exit 0 on retry; first attempt hit a transient Windows access-denied lock on `yaml/_yaml.cp313-win_amd64.pyd` (exit 2), then completed after the lock cleared. |
| Full tests | `uv run pytest -v` | **2209 passed, 46 skipped, 39 warnings**, exit 0, corrected rerun 615.71s (10:15). The initial pre-correction run also passed in 1200.50s. Unit, property-based, and integration test files were discovered through the configured `tests/` testpaths. |
| Ruff lint | `uv run ruff check src tests scripts` | **All checks passed**, exit 0. |
| Ruff format | `uv run ruff format --check src tests scripts` | **418 files already formatted**, exit 0. |
| Strict typing | `uv run pyright` | **0 errors, 0 warnings, 0 informations**, exit 0. |
| Strict docs | `uv run --project latent-anything-theory --group dev mkdocs build --strict` | **Documentation built**, exit 0, 167.97s build time. |
| Evidence ledger | `uv run python scripts/validate_evidence_ledger.py --json` | Exit 0; `errors: []`; live coverage 36/63 core (57.142857%) and 36/65 overall (55.384615%). |

The pytest run retained its configured skipped tests rather than weakening or
reclassifying network/large-download behavior. Warnings were existing
Convergence/Deprecation/UserWarning output; no test failed.

## Packaging evidence

Build command:

```text
uv build --wheel --sdist --out-dir .release-gate-dist
```

Both artifacts built successfully. Setuptools emitted its existing advisory
that the TOML-table form of `project.license` is deprecated; this is a warning,
not a build failure, and the package metadata/license contract was separately
checked.

| Artifact | Size | SHA-256 |
|---|---:|---|
| `latent_anything-0.1.0b1-py3-none-any.whl` | 404872 bytes | `458d8889d4eb8d9c217bedb01163e9b0b268b06b5f843fdeb41efe69c3a4ba2f` |
| `latent_anything-0.1.0b1.tar.gz` | 561365 bytes | `c5607e8051da5605c692fb33dc28ad2f548fd70c545ed48bb99ab6975b4fb91d` |

Each artifact was installed into its own fresh Python 3.13.3 uv environment:

- `uv venv .release-gate-wheel-env --python 3.13` then `uv pip install
  --python .release-gate-wheel-env/Scripts/python.exe
  .release-gate-dist/latent_anything-0.1.0b1-py3-none-any.whl`; import and
  metadata assertion returned `WHEEL_INSTALL_IMPORT_OK`.
- `uv venv .release-gate-sdist-env --python 3.13` then `uv pip install
  --python .release-gate-sdist-env/Scripts/python.exe
  .release-gate-dist/latent_anything-0.1.0b1.tar.gz`; import and metadata
  assertion returned `SDIST_INSTALL_IMPORT_OK`.
- `uv pip check --python .release-gate-wheel-env/Scripts/python.exe` and the
  corresponding sdist command both returned **All installed packages are
  compatible**.

The project metadata check returned `PROJECT_LICENSE_AND_PYTHON_CONTRACT_OK`
for MIT licensing and `>=3.12,<3.15` Python support.

## Dependency, security, and license audits

Dependency lock and export checks:

- `uv lock --check` — exit 0, `Resolved 272 packages in 11ms`.
- `uv export --locked --no-dev --no-hashes --output-file
  .release-gate-requirements.txt` — exit 0, 147 package entries emitted.
- `uv pip check` passed for both fresh wheel and sdist environments.

The first attempted `uvx pip-audit --locked --format json --output
.release-gate-pip-audit.json .` was not a valid audit for this repository:
`pip-audit` reported `no lockfiles found in .` (exit 1) because the project
uses `uv.lock`, not a pip lockfile. This failed attempt is retained here rather
than hidden. The corrected audit targeted the fully installed clean wheel
runtime environment:

```text
uvx pip-audit --path .release-gate-wheel-env/Lib/site-packages --format json --output .release-gate-pip-audit-wheel.json
```

It returned **No known vulnerabilities found**. The JSON contained 40 audited
dependencies and 0 vulnerabilities; output SHA-256:
`1ad2e8c11388c52f3aa45c5291973ae956991697ee961b5dcc8bbd8de9bfaf80`.

License audit command:

```text
uvx pip-licenses --python .release-gate-wheel-env/Scripts/python.exe --from mixed --format json --with-urls --output-file .release-gate-licenses-wheel.json
```

It returned exit 0 with 39 package rows and 0 missing/unknown license values;
output SHA-256:
`a495dea1aeb8ecc77bbdcb4dd1f49d468b3f4ce4fa82398992b734d9eae1e43b`.
The project metadata/license assertion also returned
`PROJECT_LICENSE_AND_PYTHON_CONTRACT_OK`.

## Correction and focused reruns

The first strict Ruff lint run found one real `SIM102` violation introduced by
the Sprint 79 Integrated Gradients provenance propagation in
`scripts/_m14_l04_envelope.py`. The nested condition was combined into one
explicit condition. Focused regression verification then returned:

```text
uv run pytest tests/test_m14_l04_integrated_gradients_handler.py -q
22 passed in 22.95s
```

The corrected file was formatted with `uv run ruff format
scripts/_m14_l04_envelope.py`; the complete Ruff lint and format gates were
rerun and passed as recorded above. No strictness was disabled and no failure
was converted to a warning.

## Scope and cleanup

No project-wide gate was skipped. Network-marked/model-download tests remained
at their configured skipped status; this gate does not claim real-model or
CUDA evidence. The repository-wide release, publication, tag, and deployment
operations were not run. Temporary build/install environments and audit output
files were used only for this verification and are removed after the evidence
is committed; only this English summary is retained.
