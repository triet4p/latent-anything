# Sprint 79 clean-environment matrix summary

## Scope and authority

This artifact covers only the Sprint 79 task **Build clean environments for
base, each optional extra, and supported combined extras on every supported
Python/platform tier**. The matrix is derived from the existing contracts; no
second compatibility matrix was introduced:

- `pyproject.toml` requires Python `>=3.12,<3.15`, yielding Python 3.12,
  3.13, and 3.14.
- `.github/workflows/ci.yml` and `.github/workflows/optional-extras.yml`
  declare the CI platform as `ubuntu-latest`. The remote GitHub Actions run
  below supplies the required Ubuntu evidence; the local Windows execution is
  supplemental.
- `pyproject.toml` declares these 12 optional profiles, in declaration order:
  `docs`, `diffusers`, `transformers`, `diffusers-full`, `3d`, `lerobot`,
  `lerobot-diffusion`, `lerobot-smolvla`, `viz`, `tracking-mlflow`,
  `tracking-wandb`, and `tracking`.
- Supported combined profiles are the declared combined extras
  `diffusers-full`, `lerobot-diffusion`, `lerobot-smolvla`, and `tracking`.
  The `tool.uv.conflicts` declarations remain unchanged: LeRobot profiles are
  not co-installed with the legacy `transformers`/`diffusers-full` profiles.
  No invalid conflict combination was run or marked supported.

The repaired workflow now has a clean base job and resolves all 12 profiles
across all three Python versions. Existing optional compatibility test jobs
retain their prior 3.12/3.13 scope; the clean-environment jobs provide the
3.14 resolution/import coverage. Every CI job starts from a fresh runner and
uses the locked resolver.

## Local execution

Local execution was performed on Windows `win32`, `AMD64`, Python 3.13.3, with
Python 3.12.12 and 3.14.0 interpreters selected by uv. Every lane used a fresh
temporary virtual environment through `UV_PROJECT_ENVIRONMENT`; temporary
environments were removed after each lane. Resolution/install used:

```text
uv sync --locked --python <3.12|3.13|3.14> [--extra <profile>]
```

Runtime model/data acquisition was disabled for the smoke phase with
`HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`WANDB_MODE=disabled`. The base smoke imported `latent_anything` and asserted
that no optional provider modules (`diffusers`, `transformers`, `tokenizers`,
`gsplat`, `lerobot`, `mlflow`, `wandb`, `plotly`, `kaleido`, `ipywidgets`, or
`anywidget`) leaked into `sys.modules`. Each profile smoke ran
`uv run --locked --extra ${{ matrix.extra }} python -c "import latent_anything"`
so it could not re-resolve or execute outside the selected profile; no model,
data, CUDA, or provider operation was invoked.

## Results

All locally reachable Windows lanes resolved, installed, imported, and passed
base isolation checks:

| Python | Base | `docs` | `diffusers` | `transformers` | `diffusers-full` | `3d` | `lerobot` | `lerobot-diffusion` | `lerobot-smolvla` | `viz` | `tracking-mlflow` | `tracking-wandb` | `tracking` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.12.12 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 3.13.3 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 3.14.0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

Total local result: **39/39 lanes passed** (3 base lanes plus 36 profile
lanes). No implicit model/data download occurred. The remote Ubuntu run below
also passed all required clean-base and resolve-extra lanes.

## Ubuntu GitHub Actions evidence

The corrected workflow was dispatched on the dedicated review branch
`sprint79-local-gate-remediation` at commit
`5911d0096decb92269d2273c50d6de812c08dc5a`:

- Run ID: `34681280312`
- URL: https://github.com/triet4p/latent-anything/actions/runs/34681280312
- Platform: `ubuntu-latest`
- Workflow conclusion: **success**
- Required lanes: **39/39 success** — `clean-base` for Python 3.12, 3.13,
  and 3.14, plus `resolve-extra` for all 12 profiles on each Python version.
- The run was waited to settlement with `gh run watch 34681280312
  --repo triet4p/latent-anything --exit-status`; no required lane failures
  occurred.

The individual required job conclusions were all `success`:

| Python | `clean-base` | `docs` | `diffusers` | `transformers` | `diffusers-full` | `3d` | `lerobot` | `lerobot-diffusion` | `lerobot-smolvla` | `viz` | `tracking-mlflow` | `tracking-wandb` | `tracking` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.12 | success | success | success | success | success | success | success | success | success | success | success | success | success |
| 3.13 | success | success | success | success | success | success | success | success | success | success | success | success | success |
| 3.14 | success | success | success | success | success | success | success | success | success | success | success | success | success |

No failure logs exist for the required lanes because every required job
concluded successfully.

The workflow matrix was checked against its source definitions. The local
39-lane run used the exact locked command pattern above, and the remote
`resolve-extra` lanes used:

```text
uv run --locked --extra ${{ matrix.extra }} python -c "import latent_anything"
```

The workflow matrix was checked against its source definitions and the local
39-lane run used the exact locked command pattern above. The final local
summary was:

```text
{'total': 39, 'passed': 39, 'failed': []}
```

The post-review workflow static assertion command was:

```text
uv run python -c "...assert the resolve-extra smoke uses
uv run --locked --extra ${{ matrix.extra }} ...; assert the clean-base leak
list includes ipywidgets and anywidget..."
```

It returned:

```text
WORKFLOW_LOCKED_PROFILE_SMOKE_AND_BASE_LEAK_CHECK_OK
```

Additional checks:

- `uv lock --check` — **exit status 0** (`Resolved 272 packages in 4ms`).
- `uv run python -c "...parse pyproject.toml..."` — **`EXTRAS_12_CONFLICTS_6_OK`**; all 12 profiles and six declared conflicts were confirmed unchanged.

The repository-wide project test, lint, type, docs, packaging, and release
suites were intentionally not run for this atomic task.
