# Sprint 81 Task 4 — clean candidate package build and install matrix

**Task:** Build wheel/sdist from a clean checkout and verify installed imports, plugin discovery, and lightweight examples in base and optional-extra environments.

**Plan status:** Task 4 remains `[~]`; Main owns the plan checkbox and evidence review. This handoff is not a review verdict and does not mark the task complete.

## Outcome

Built the unpublished `1.0.0` candidate wheel and source distribution twice from independent clean worktrees of candidate revision `9d9c613393976e5cd96da1ba79e38f97a4729f8d`. Both builds produced byte-identical wheel and normalized sdist checksums. Archive safety, file inventory, `1.0.0` metadata, advertised extras, console entry point, and runtime import version all passed. Fresh locked base environments installed the wheel and sdist; the README quick start passed from each installed artifact. A separately installed plugin was discovered and constructed through the installed wheel. Compatible optional profiles and the LeRobot / LeRobot Diffusion profiles installed and passed import/dependency checks.

No package/source fix was needed. No tag, push, release workflow, upload, or publication was performed. External stable-tag-ruleset and PyPI Trusted Publisher prerequisites remain pending.

## Candidate checkout and build evidence

The `1.0.0` Task 1–3 candidate changes were staged by explicit path and committed locally as `chore(release): prepare 1.0.0 candidate`; the commit contains those candidate metadata, release-gate, documentation, API-snapshot, and focused-contract changes. It does not include Task 4 results or change the Sprint 81 plan. Commit revision:

```text
9d9c613393976e5cd96da1ba79e38f97a4729f8d
```

Two detached worktrees were created at that exact revision. Before each build, `git status --short --branch` returned only `## HEAD (no branch)` and `git rev-parse HEAD` returned the revision above. The unrelated dirty/untracked files in the original project worktree were not staged by path selection and were left untouched.

Build environment: Windows 11 x64; CPython `3.13.3`; uv `0.9.7 (0adb44480 2025-10-30)`. `SOURCE_DATE_EPOCH=1790373545` is the candidate commit timestamp. The isolated PEP 517 build environment resolved `setuptools==84.0.0` from `setuptools>=75`. For both clean worktrees, `VIRTUAL_ENV`, `PYTHONPATH`, and the original project `.venv\Scripts` PATH entry were removed; uv selected `C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe`. Python downloads were disabled and uv ran offline using the available build dependency cache.

Exact build invocations (one per clean worktree; output directories differ):

```text
uv build --wheel --sdist --out-dir <worktree>/dist-task4-{c,d} --clear --no-create-gitignore --python 3.13.3 --offline -vv
```

The environment supplied `SOURCE_DATE_EPOCH=1790373545`. Each sdist was normalized using the same algorithm as `.github/workflows/release.yml`: gzip and tar member timestamps set to the commit epoch, uid/gid set to zero, owner/group names cleared, and PAX headers cleared. This normalization was applied to both independent build outputs before hashing.

| Artifact | Size | SHA-256 | Repeated clean build |
|---|---:|---|---|
| `latent_anything-1.0.0-py3-none-any.whl` | 634,812 bytes | `a612f319a1a6dc7bfee211ac93486ec793a1dde6968103ebd2aae1b291b03e7a` | identical |
| `latent_anything-1.0.0.tar.gz` | 771,817 bytes | `f9379e49264908d8b379a95b9f3ea79a7d1f71571c4dc9d3eaa84badede2bf4c` | identical after normalization |

Archive validation used Python `zipfile`, `tarfile`, `email`, and `hashlib`:

- Each output directory contained exactly the wheel and sdist named above; no `0.9.0` artifact was present.
- Wheel CRC validation passed; the wheel contains 168 members and exactly one `.dist-info/METADATA` with `Name: latent-anything`, `Version: 1.0.0`, and `Requires-Python: <3.15,>=3.12`.
- Wheel metadata advertises all 12 extras declared in candidate `pyproject.toml`. `entry_points.txt` contains `latent-anything = latent_anything.cli:main`; required package and integration modules are present.
- The sdist contains 278 members under its single `latent_anything-1.0.0/` root, including `PKG-INFO`, `pyproject.toml`, and `src/latent_anything/__init__.py`. Its package metadata is `latent-anything` version `1.0.0`.
- Both archives have no absolute or parent-traversal paths. Installed runtime and distribution metadata versions were also `1.0.0` in every smoke environment. Historical `0.9.0` references in release documentation remain intentional; they did not contaminate built package metadata or filenames.

## Locked clean install and smoke results

Each environment was new, outside the candidate checkout, created with `uv sync --locked --no-default-groups --no-install-project --python 3.13.3` and an explicit `UV_PROJECT_ENVIRONMENT` path. Dependencies came from candidate `uv.lock` (272 packages resolved for the project lock); the wheel or sdist was then installed separately with `uv pip install --python <environment-python> --no-deps <artifact>`. `PYTHONNOUSERSITE=1` was set and smoke processes ran with no source-checkout `PYTHONPATH`. Imports reported their module path under the target environment’s `Lib/site-packages`, not the checkout.

| Fresh environment | Locked profile / artifact | Observed result |
|---|---|---|
| Base wheel | Base lock; built wheel | Base sync installed 38 dependencies. Wheel install succeeded. `uv pip check` checked 39 installed packages: all compatible. |
| Base sdist | Base lock; normalized sdist installed with `uv pip install --offline --python <sdist-python> --no-deps <sdist>` | PEP 517 built and installed the sdist successfully. `uv pip check` checked 39 packages: all compatible. |
| Standard optional profiles | `docs`, `diffusers-full`, `viz`, `tracking`; built wheel | Locked sync installed 183 dependencies. Wheel install succeeded. `uv pip check` checked 184 packages: all compatible. |
| LeRobot | `lerobot`; built wheel | Locked sync installed 84 dependencies. Wheel install and LeRobot module imports succeeded. `uv pip check` checked 85 packages: all compatible. |
| LeRobot Diffusion | `lerobot-diffusion`; built wheel | Locked sync installed 88 dependencies. Wheel install and integration imports succeeded. `uv pip check` checked 89 packages: all compatible. |

The README Quick Start was executed as written (seed 42, 12 four-dimensional random points, PCA to two dimensions) from the installed wheel in the base and standard optional environments and from the installed sdist in the second base environment. Each printed runtime/distribution version `1.0.0` and projection shape `(12, 2)`. The module and `.dist-info` paths pointed into the respective fresh environment. The installed wheel’s `latent-anything.exe --help` exited 0 and displayed the package CLI commands and documented aliases.

The plugin guide’s installed-distribution path was exercised with the checked-in hello-plugin fixture:

```text
uv pip install --offline --no-build-isolation --target <base-env>/plugin-target --no-deps tests/fixtures/sprint73_hello_plugin
```

A clean child process had only the plugin target on `PYTHONPATH`; it imported the framework from the base environment’s installed wheel. `list_entry_points()` found one `hello-world` declaration without importing `latent_anything_hello`; explicit `load_entry_points()` imported and loaded it with no issues. Building `ObjectSpec(kind="adapter", name="hello-world", params={"prefix": "hi"})` returned `"hi:world"`. The entry reported group `latent_anything.adapter`, distribution `latent-anything-hello-plugin` version `0.1.0`, external source, target `latent_anything_hello:HelloAdapter`, and plugin API version `1`.

The standard optional environment successfully imported `mkdocs`, `mkdocs_jupyter`, `diffusers`, `transformers`, `tokenizers`, `plotly`, `kaleido`, `ipywidgets`, `anywidget`, `mlflow`, `wandb`, `latent_anything.integrations.diffusers_vae`, `latent_anything.integrations.transformer_lm`, both tracking recorder modules, and `latent_anything.visualization`. Locked versions observed: `mkdocs-material 9.7.7`, `mkdocs-jupyter 0.26.3`, `diffusers 0.39.0`, `transformers 4.57.6`, `tokenizers 0.22.2`, `plotly 6.9.0`, `kaleido 1.3.0`, `ipywidgets 8.1.8`, `anywidget 0.11.0`, `mlflow 3.14.0`, and `wandb 0.28.1`.

The LeRobot environment imported `lerobot` `0.6.1` and `latent_anything.integrations.lerobot`, `lerobot_act`, and `lerobot_dataset`. The LeRobot Diffusion environment imported `lerobot` `0.6.1`, `diffusers` `0.39.0`, and `latent_anything.integrations.lerobot_diffusion`. No model/checkpoint, dataset, simulation, or GPU workload was fetched or run.

## Optional-extra coverage

For each optional profile below, the exact resolution check was:

```text
uv sync --locked --dry-run --no-default-groups --no-install-project --python 3.13.3 --extra <extra>
```

All 12 individual checks exited 0 and reported `Resolved 272 packages`; these dry runs are resolution evidence, not install/runtime success.

| Extra | Install/import status |
|---|---|
| `docs` | Installed and imported in the standard optional environment. |
| `diffusers` | Dependencies installed and imported through `diffusers-full`. |
| `transformers` | Dependencies installed and imported through `diffusers-full`. |
| `diffusers-full` | Installed; `diffusers`, `transformers`, `tokenizers`, and package integration import passed. |
| `3d` | Lock resolution passed only. No install/import/runtime claim: this machine is Windows and CPU-only; no CUDA/3D workload was attempted. |
| `lerobot` | Installed; LeRobot and package integration imports passed. |
| `lerobot-diffusion` | Installed; LeRobot, Diffusers, and package integration imports passed. |
| `lerobot-smolvla` | Lock resolution passed only. No install or runtime claim: this candidate explicitly keeps SmolVLA blocked/non-gating, and its profile includes the Linux-only Libero simulation extra. No model or simulation data was downloaded. |
| `viz` | Installed and imported in the standard optional environment. |
| `tracking-mlflow` | Dependencies installed and imported through `tracking`. |
| `tracking-wandb` | Dependencies installed and imported through `tracking`. |
| `tracking` | Installed; both MLflow and W&B integrations imported. |

No all-extras sync was attempted: the candidate deliberately declares incompatible LeRobot-vs-legacy-Transformer/Diffusers profile pairs. The compatible standard extras were installed together; LeRobot profiles were installed in separate environments. `3d` and `lerobot-smolvla` were not misreported as install/runtime passes based on successful dry-run resolution.

## Scope, graph, and open findings

- No Task 4 source or project-document change was needed; only this required handoff artifact was created. Therefore no Graphify update was run. The existing project graph is present at `graphify-out/graph.json`; Task 3’s handoff records its prior candidate refresh. The Task 4 handoff itself was not added to the graph.
- No project-wide tests, formatters, linters, docs builds, release-readiness dispatch, tag creation, push, upload, or publication were run.
- The build emitted a non-fatal Setuptools deprecation warning for the table form of `project.license`; Setuptools says the metadata should migrate to an SPDX string and `license-files` before `2027-02-18`. Both artifacts still built and validated.
- Reproducibility evidence is scoped to this Windows x64 / CPython 3.13.3 / uv 0.9.7 build with `setuptools==84.0.0` and the recorded commit epoch. Other operating systems and Python minors were not built. Candidate `pyproject.toml` allows `setuptools>=75`; this report records the actual resolver result, not a cross-time or cross-platform backend guarantee.
- The stable-tag ruleset and PyPI Trusted Publisher (`pypi`) remain pending in `docs/release-gates.json`. This validation did not change those external gates and does not authorize publication.
- Task 4 remains `[~]` pending Main’s evidence review.
