# Sprint 35 Carryover Task C35.3 — Fidelity Closure Reconciliation

**Status:** Complete — fidelity gate closed at the C35.3 historical boundary; C35.4 subsequently closed interpolation.

## Scope

Reconcile the project plan, Sprint 35 plan, typed/rendered evidence ledger,
changelog, and append-only lesson log after C35.2's successful local-only
revision-pinned fidelity run. At the C35.3 starting point, this task closed
only the fidelity gate; the real-checkpoint interpolation artifact was pending
and Milestone 8 was partial before C35.4.

## Truthful evidence

- Checkpoint: `stabilityai/sd-vae-ft-mse`, revision
  `31f26fdeee1355a5c34592e401dd41e45d25a493`, MIT model-card provenance.
- Safetensors: 334,643,276 bytes; SHA-256
  `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`.
- Locked runtime: Diffusers 0.39.0, safetensors 0.8.0, Torch 2.10.0+cpu.
- Direct `AutoencoderKL` and `DiffusersAutoencoderKLAdapter` parity is exact
  (zero maximum absolute error) for deterministic mean and independently
  seeded posterior-sample latent/decode outputs.
- Local-only loading made zero network attempts and used no remote code;
  runtime was 2.7973485 seconds and peak RSS was 1,446,883,328 bytes within
  the 60-second/2-GiB bounds.

## Reconciliation changes

- Marked C35.3 and the direct-backend fidelity task complete in the Sprint 35
  plan while leaving interpolation pending at that historical boundary.
- Marked only the Sprint 35 fidelity carryover gate complete in `docs/PLAN.md`;
  Milestone 8 was partial because interpolation was open at that boundary.
- Promoted the existing `THY-T02-AUTOENCODER` ledger item to D2 with the
  Diffusers source/tests, fidelity benchmark/config, and JSON artifact.
- Added rendered ledger, changelog, and lesson-log statements preserving the
  exact revision/hash and the eval-mode/local-generator root-cause lesson.

## Focused validation

- `uv run python scripts/validate_evidence_ledger.py` — passed; 107
  capabilities, core 25/63, overall 25/65.
- `git diff --check` — passed (only normal Git LF-to-CRLF working-copy
  warnings were emitted for pre-existing and edited text files).

## Graph refresh after reconciliation

- Command: `graphify update .`
- Result: exit 0; rebuilt graph with **10,128 nodes, 19,660 edges, and 882
  communities**. It reported 49 known zero-node JSON/source warnings and
  refreshed `graphify-out` successfully. The changed community count is a
  graph extraction result, not a product-evidence metric.

## Closure-gate evidence

- `uv run --offline --extra diffusers python scripts/diffusers_vae_fidelity.py`
  — passed; regenerated `artifacts/diffusers_vae_fidelity.json` with exact
  zero-error mean and seeded-sample parity, zero network attempts, runtime
  2.7973485 seconds, and peak RSS 1,446,883,328 bytes.
- `LATENT_ANYTHING_RUN_REAL_CHECKPOINT=1 uv run --offline --extra diffusers
  pytest tests/test_diffusers_vae.py tests/test_diffusers_vae_fidelity.py -q`
  — **7 passed** in 13.37 seconds.

The remaining repository gates are recorded after they run. Interpolation is
not run.

The first full default pytest run completed with one regression-test failure:
the Pyright cleanup renamed a monkeypatch lambda parameter but did not accept
the production function's keyword argument. The test seam is being corrected
to accept ignored keyword arguments without changing production behavior.

During the scoped lint gate, Ruff found one line-length violation in the
evidence script and one unused lambda parameter in the adapter regression
test. Both were corrected without changing runtime semantics; the scoped
Ruff/format commands are rerun below.

Ruff format then normalized the evidence script; this mechanical formatting
change is included in the closure diff and followed by another graph refresh.

Strict Pyright then found only annotation/test-private-access issues in the
new lane (Diffusers' dynamic export, literal mode narrowing, and a deliberate
private backend cache assertion). The harness now uses an explicit `Any` cast
at the optional dynamic boundary, a typed mode tuple, and a line-scoped
private-use suppression for that regression assertion. No runtime behavior or
public library boundary changed.

## Final closure validation

- `uv run pytest -q` — **1,496 passed, 33 skipped, 39 warnings** in 156.93
  seconds. The first attempt exposed and the focused test then corrected the
  monkeypatch keyword-argument issue described above; this is the successful
  rerun.
- `uv run --extra docs mkdocs build --strict` — passed; documentation built
  in 198.24 seconds. The only warning was the upstream Material for MkDocs
  MkDocs 2.0 compatibility notice.
- Scoped Ruff check/format and strict Pyright — passed; Pyright reported 0
  errors, 0 warnings, 0 informations.
- `uv run python scripts/validate_evidence_ledger.py` — passed; 107
  capabilities, core 25/63, overall 25/65.
- `git diff --check` — passed with ordinary LF-to-CRLF working-copy warnings.

## Final graph refresh and worktree audit

- Latest graph snapshot before this final artifact record: **10,133 nodes,
  19,665 edges, and 908 communities**; 49 known zero-node JSON/source
  warnings. The final refresh after recording this result completed with exit
  0 and reported the graphify watch notice that the saved 906-label community
  set differs from the current 908 communities (144 renamed hubs); no product
  evidence is affected. Graph outputs are expected dirty generated state and
  are not product evidence.
- `git status --short --branch` remains intentionally dirty on `main` with
  the cumulative Sprint 34–77 remediation/delivery diff. No tracked weights,
  `.bin`, or safetensors files were found; the ignored local checkpoint is not
  part of the worktree. At the time of this C35.3 fidelity-only audit, no
  interpolation JSON/PNG artifact had yet been generated; C35.4 subsequently
  produced the canonical pair and superseded that pending status.
- The MkDocs `.gh-pages-build` output is generated/ignored. No credentials or
  secrets were added by this Sprint 35 closure.

## Final status

The revision-pinned fidelity gate was **closed by C35.3**. C35.4 subsequently
closed the interpolation gate and the declared Milestone 8 evidence scope;
neither task claims perceptual quality, broader real-image quality, CUDA, or a
stable release.
