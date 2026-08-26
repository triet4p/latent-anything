# Sprint 78.29 — RFC0001 canonical public symbols

## Scope and decision

This atomic task adds the RFC0001 canonical names `AnalysisMethod`,
`Intervention`, and `InterventionPipeline` without removing or warning on the
beta names. The implementation keeps the existing frozen definitions and
assigns exact aliases:

| Canonical name | Beta compatibility name | Definition module | Identity result |
|---|---|---|---|
| `AnalysisMethod` | `Method` | `latent_anything.methods.protocols` | `AnalysisMethod is Method` |
| `Intervention` | `BMethod` | `latent_anything.methods.b_protocols` | `Intervention is BMethod` |
| `InterventionPipeline` | `ManipulationPipeline` | `latent_anything.manipulation_pipeline` | `InterventionPipeline is ManipulationPipeline` |

Canonical imports are available from the top-level package and the existing
`methods`/`pipeline` facades. No wrapper subclass, import-time warning,
registry-kind change, schema change, version bump, alias removal, or change to
`method_a`/`method_b` and `KIND_METHOD_A`/`KIND_METHOD_B` was made.

## Deterministic inventory

The Sprint 78.28 baseline had 202 top-level exports and top-level `__all__`
order SHA-256
`4a101a3a30687437958c6d504f9741f962813daf46ee4322c3f122bd1bf3f8e6`.
After this additive change:

- top-level exports: **205** (the original 202 names remain the exact prefix);
- top-level `__all__` order SHA-256:
  `0652dc0ba91671e40705d2455a164caecd34382256e1094bfb2144a0785aa2ab`;
- `latent_anything.methods.__all__`: **10**, with the original eight names
  preserved as the prefix; order SHA-256:
  `2ac06a70c39953270c6cb028df342e10892f0237829c74abc129245287fc8f2e`;
- canonical signature/module/name rows (ordered canonical names), using
  `name|obj.__module__|obj.__name__|inspect.signature(obj)`, are:

  ```text
  AnalysisMethod|latent_anything.methods.protocols|Method|(*args, **kwargs)
  Intervention|latent_anything.methods.b_protocols|BMethod|(*args, **kwargs)
  InterventionPipeline|latent_anything.manipulation_pipeline|ManipulationPipeline|(method: 'BMethod', adapter: 'FlatBatchDecodableAdapter | None' = None) -> 'None'
  ```

  Their ordered-row SHA-256 is
  `d3b8108fcd558dc27922555ba0fa14faff6fe28dc67ce7c562634710e36bcd86`.
  The prior full runtime identity scan digest was
  `2c5429b14fc9b86d222f86a2b7c2c40967766c1a685ab12af96af24e88fe7c2c`;
  existing beta rows remain unchanged.

## Compatibility tests

`tests/test_api_surface.py` now snapshots the 205-name top-level order and
10-name methods order and verifies:

- top-level, methods-facade, protocol-module, manipulation-module, and
  pipeline-facade imports;
- exact object identity for all three canonical/legacy pairs;
- unchanged module and class names, constructor/method signatures, and
  pickle round trips;
- runtime-checkable Protocol conformance of `PCA` and `Lerp` through the new
  canonical names;
- unchanged registry-kind snapshot (`adapter`, `analysis`, `intervention`).

Focused command:

```text
uv run pytest tests/test_api_surface.py tests/test_latent_anything/test_registry.py tests/test_portable.py tests/test_run_record.py tests/test_artifact_store.py -q
89 passed in 19.84s
```

## Gates and graph

- `uv run ruff check src tests`: PASS.
- `uv run ruff format --check src tests`: PASS (`251 files already formatted`).
- `uv run pyright`: PASS (`0 errors, 0 warnings, 0 informations`).
- Authoritative CI-equivalent environment: `uv sync --locked --extra viz`
  completed successfully. It changed only the active virtual environment
  profile (51 packages removed and 11 installed); no repository implementation
  files changed.
- Full `uv run pytest -q` under the locked `viz` profile: PASS (`1546 passed,
  36 skipped, 39 warnings`). This is exactly the Sprint 78.28 baseline of
  `1545 passed, 36 skipped` plus the one new RFC0001 API test. The earlier
  `1504 passed, 45 skipped` run used the `docs` profile left active by task
  78.27 and is not CI-comparable.
- Focused API/registry/portable/run-record/artifact suite after profile sync:
  PASS (`89 passed in 9.75s`).
- A repository-root `uv run ruff check .` / `ruff format --check .` also scans
  bundled skill sources and theory notebooks outside the package/test scope;
  it reports 1920 pre-existing findings and 106 notebook formatting candidates.
  No unrelated files were changed; the authoritative project gate is the clean
  `src` + `tests` scope above.
- Final `git diff --check`: PASS (only the repository's existing LF/CRLF
  normalization warnings are emitted).
- Final graphify update: **10,769 nodes / 20,754 edges / 930 communities**;
  topology includes the additive canonical export edges and no new cycle or
  dependency direction.

No external model, network, CUDA, commit, push, tag, or release action is in
scope.

Sprint 78 remains pre-stable: package metadata stays `0.1.0b1`; aliases remain
available through the beta window and no `0.9.0` tag or publication is created.
