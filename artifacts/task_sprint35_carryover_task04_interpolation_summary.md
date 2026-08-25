# Sprint 35 Carryover Task C35.4 — Local Interpolation Evidence

**Status:** Complete — final interpolation and declared Milestone 8 evidence scope closed on 2026-08-26.

## Predeclared acceptance

- Use the verified local `stabilityai/sd-vae-ft-mse` snapshot at revision
  `31f26fdeee1355a5c34592e401dd41e45d25a493`, safetensors SHA-256
  `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`,
  334,643,276 bytes, MIT model-card provenance, and locked Diffusers 0.39.0,
  safetensors 0.8.0, Torch 2.10.0+cpu.
- Use distinct deterministic sklearn-digits endpoints (`digit_0`, `digit_1`)
  converted to identical float32 NCHW inputs in `[-1, 1]`; coefficients are
  exactly seven increasing values from 0.0 through 1.0.
- Preserve endpoint order and identity, validate finite shape/dtype/range,
  compare endpoint reconstruction, and reject latent/decoded collapse using
  explicit minimum L2 movement thresholds (1e-3 endpoint and 1e-4 adjacent).
- Require deterministic JSON content and PNG digests, matching JSON/PNG
  dimensions (2100×360), local-only safetensors/no remote code/zero socket
  attempts, and CPU bounds of 60 seconds/2 GiB RSS.
- This is quantitative reconstruction/interpolation evidence, not a claim of
  perceptual quality or a general diffusion-pipeline interpolation contract.

## Implementation and focused validation

- Replaced the old download-prone script with a local snapshot lane that
  validates provenance/hash/config, denies sockets during adapter load, records
  movement/resource metrics, and writes corresponding JSON/PNG artifacts.
- Added adversarial tests for reversed endpoints, collapsed paths, provenance,
  hash/coefficient tampering, and missing PNG output.
- `uv run pytest tests/test_diffusers_vae_interpolation.py -q -m 'not large_download'`
  — **5 passed, 1 deselected**.
- Scoped Ruff check/format — passed.
- Strict Pyright was clean after exposing the script validation helper as a
  named utility; the cached real lane remains to be run.

The first reproducibility test run found that calling
`torch.set_num_interop_threads(1)` during a second in-process evidence run
raises after parallel work has started. The lane now tolerates only that known
already-configured condition and still raises unrelated Torch configuration
errors; the focused real lane is rerun after this remediation.

## Cached real-checkpoint evidence

- `uv run --offline --extra diffusers python scripts/diffusers_vae_interpolation.py`
  — passed; generated the canonical JSON/PNG pair.
- Model provenance: `stabilityai/sd-vae-ft-mse` at revision
  `31f26fdeee1355a5c34592e401dd41e45d25a493`, MIT, safetensors SHA-256
  `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`.
- Coefficients: `[0.0, 0.16666667, 0.33333334, 0.5, 0.66666669,
  0.83333331, 1.0]`; endpoint order is digit 0 → digit 1.
- Latent shape/dtype: `(7, 4, 4, 4)` float32; decoded shape/dtype:
  `(7, 3, 32, 32)` float32. Endpoint latent L2 **8.3936653**, minimum
  adjacent latent L2 **1.3989439**, endpoint decoded L2 **51.7986336**, and
  minimum adjacent decoded L2 **9.1108503**; all exceed the declared gates.
- Endpoint reconstruction consistency: max absolute error **0.0**, RMSE
  **0.0**. Latent/decoded values are finite; input range is `[-1, 1]`.
- PNG: 2100×360 RGBA, SHA-256
  `65245a9c171106dc33ce63c3687738721985917aafa7fec9f4751355e3cbfe40`;
  JSON deterministic content SHA-256
  `b3887f0e4b13e5942011275dac77da3d7f92bfb8203d67a46f119d61abeaf0dd`.
- Local-only: zero network attempts, no remote code; runtime **3.5648021 s**
  and peak RSS **1,150,750,720 bytes**, within 60-second/2-GiB bounds.
- `LATENT_ANYTHING_RUN_REAL_CHECKPOINT=1 uv run --offline --extra diffusers
  pytest tests/test_diffusers_vae_interpolation.py -q` — **6 passed** in
  17.69 seconds, including reproducible second-run digest checks.
- Scoped Ruff check/format and strict Pyright — passed (0 errors, warnings,
  and informations).

## Graph refresh after C35.4 evidence

- Command: `graphify update .`
- Result: exit 0; **10,159 nodes, 19,717 edges, and 880 communities**, with
  50 known zero-node JSON/source warnings. The topology is graph navigation
  metadata, not an interpolation quality metric.

## Closure reconciliation

- Updated `docs/PLAN.md` and `docs/sprint-plans/sprint-35.md` to close the
  interpolation gate and Milestone 8's declared bounded evidence scope.
- Updated the rendered and typed evidence ledgers, `CHANGELOG.md`,
  `docs/DIFFUSERS_INTEGRATION.md`, README current-status wording, and the
  historical C35.3 artifact to distinguish the now-superseded pending state.
- Focused post-reconciliation checks: evidence validator passed with 107
  capabilities (core 25/63, overall 25/65); `git diff --check` passed.
- Latest graph snapshot before this closure-record update: **10,162 nodes,
  19,720 edges, and 895 communities**, with 50 known zero-node warnings.

## Final closure gates and integrated audit

- `uv run pytest -q` — **1,501 passed, 34 skipped, 39 warnings** in 218.44
  seconds. The count includes the six interpolation tests and the prior
  Sprint34–77 suite.
- `uv run python scripts/validate_evidence_ledger.py` — passed; 107
  capabilities, core 25/63, overall 25/65.
- `uv run --extra docs mkdocs build --strict` — passed; documentation built in
  32.64 seconds. Only the upstream Material/MkDocs 2.0 compatibility warning
  was emitted.
- `git diff --check` — passed with ordinary LF-to-CRLF working-copy warnings.
- Final audit found no current stale Sprint35/Milestone8 caveats, no tracked
  checkpoint weights or secrets, and no untracked generated output beyond the
  intended JSON/PNG/artifact files and expected ignored MkDocs/graphify state.
  Sprint34 held-out evidence, Sprint35 fidelity/interpolation evidence, and
  Sprint73–77 closure artifacts/plans are mutually consistent for their
  declared scopes. Stable release/Milestone14 work was not started.

The interpolation carryover and Milestone8's declared bounded evidence scope
are closed. The result remains D2/local CPU evidence and does not claim
perceptual quality, CUDA, hosted integrations, or a stable release.
