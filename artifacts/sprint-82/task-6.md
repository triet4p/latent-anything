# Sprint 82 Task 6 — Intervene and compare

**Task status:** `[~]` (Main owns evidence review and status change)

## Summary

Added an intervention-and-comparison guide for the published `latent-anything==1.0.0` API. It distinguishes stateless `Lerp`, contrast-fitted `SteeringVector`, flat-batch `ActivationPatch`, identity-bound `SubspaceProjection` project/remove/transfer, and `InterventionPipeline`. It explains the difference between returning an edited latent, the released adapter-mediated encode/patch/decode path, and injecting an activation during a model's forward execution; documents exact `FlatBatchDecodableAdapter` shape expectations and coordinate-identity responsibilities; and lays out identity/null, randomized, off-target, and positive/restore controls alongside observational-versus-causal claim boundaries.

The runnable CPU example uses a fixed 2-D orthogonal linear adapter. It exercises all five API paths, compares a zero-delta identity patch with a nonzero intervention, checks a zero-strength steering control, and measures a predeclared off-target feature. Its limits explicitly exclude conclusions about nonlinear models, arbitrary adapters, and causal model behavior.

## Files changed

- `docs/public/intervene-compare.md` — new task-oriented intervention/comparison guide, runnable CPU example, controls, evidence boundaries, and release/specialist links.
- `docs/public/index.md` — linked the intervention/comparison route from the landing page.
- `mkdocs-api.yml` — added the page to the API site's navigation.
- `graphify-out/graph.json` — scoped deterministic graph merge for the new guide and updated landing page.
- `graphify-out/manifest.json` — refreshed AST stamps for those two Markdown sources.
- `artifacts/sprint-82/task-6.md` — this evidence record.

No package API, release tag, theory-site configuration/content, sprint-plan status, or public deployment was changed.

## Verification

- **Runnable CPU example — PASS.** Executed the Python block extracted from the guide with:

  ```text
  uv run --locked python -c "from pathlib import Path; p=Path('docs/public/intervene-compare.md').read_text(encoding='utf-8'); b=p.split(chr(96)*3+'python',1)[1].split(chr(96)*3,1)[0]; exec(b, {'__name__':'__main__'})"
  ```

  Exit status 0. Observed:

  ```text
  Lerp midpoint decoded: [0.5, 10.0]
  Steering strength=0.5 decoded: [0.5, 12.0]; strength=0 unchanged: True
  Subspace remove decoded rows: [[0.0, 12.0], [0.0, 12.0]]
  ActivationPatch latent delta: [0.8, -0.6]
  ActivationPatch identity max |delta|: 0.000
  ActivationPatch changed rows: [[1.0, 12.0], [3.0, 12.0]]
  Off-target feature max |delta|: 0.000
  ```

  The first run used an exact floating-point equality for the off-target residual and failed on round-trip numeric residue. The example now uses `np.isclose(..., atol=1e-12)`; the rerun passed and produced the output above.

- **Strict API-guide build — PASS.** Ran after the final page edits:

  ```text
  uv run --locked --extra docs mkdocs build --strict --config-file mkdocs-api.yml
  ```

  MkDocs built `.api-pages-build/` successfully with exit status 0. The Material for MkDocs 2.0 announcement banner appeared; there were no MkDocs build errors.

- **Local HTTP and Chromium surface — PASS.** Served the final built site on `127.0.0.1:8771`. Fetching `http://127.0.0.1:8771/latent-anything/api/intervene-compare/` returned HTTP `200` and contained the guide title. In Chromium, opened the local landing page, clicked the visible `Intervene and compare representations` navigation link, and confirmed route `/latent-anything/api/intervene-compare/`, title `Intervene and compare representations - Latent Anything API Guide`, and H1 `Intervene on latent representations and compare outcomes`. The rendered article contained the expected example and identity/off-target controls; Chromium reported no console errors. An early raw-response substring probe for one nested-list output line was too specific and returned false; rendered article-text checks confirmed the expected result. The tab and temporary HTTP server were closed afterward.

- **Public deployment — not attempted.** Local build/HTTP/browser checks do not establish that the intended public `/api/` URL is deployed; publication remains Task 9 scope.

## Graph update

**Scoped deterministic AST merge — PASS.** Used the interpreter recorded in `graphify-out/.graphify_python` and `graphifyy 0.9.68`. A temporary-copy dry run passed before the actual update. The only extraction inputs were `docs/public/intervene-compare.md` and `docs/public/index.md`; extraction used `extract(..., root=root, parallel=False)`. No semantic extraction, full-corpus graph build, community reclustering, report regeneration, or visualization regeneration was run.

The merge used `build_merge([extraction], graph_path=..., root=root, dedup=False)` and `to_json` with existing community labels and `built_at_commit`, then `save_manifest(..., kind='ast', root=root, scan_corpus=None)`. Dry-run and post-write checks confirmed that all unrelated graph node records and unrelated links were preserved attribute-for-attribute, all graph metadata/flags and all hyperedges were unchanged, and every unrelated manifest entry remained unchanged.

- Extraction: **19 nodes / 25 edges**.
- Graph: **16,853 nodes / 39,463 links / 38 hyperedges** → **16,866 / 39,478 / 38**.
- Preserved unrelated records: **16,847 nodes and 39,454 links**.
- Manifest entries: **1,828 → 1,829**.
- `docs/public/index.md` AST hash: `54bd61b4ce3985899fbf1e07277234aa` (semantic hash empty).
- `docs/public/intervene-compare.md` AST hash: `219fc0f26270264ab9e331dd8035cbf2` (semantic hash empty).
- `graphify-out/GRAPH_REPORT.md`, `graphify-out/graph.html`, and `.graphify_labels.json` were intentionally not regenerated or changed; this was a scoped merge, not a reclustering run.

## Evidence limits and open findings

- The example validates declared API operations and flat-batch shapes on one deterministic CPU linear fixture only. It is not evidence for nonlinear models, production model-forward hooks, arbitrary model behavior, or causal use of a feature.
- The bounded encoder and transformer intervention/control examples are linked to the existing AI engineer guide and remain limited to their frozen model, dataset, task, and comparison protocols.
- No task-level implementation blocker remains. Public publication/availability is not claimed and belongs to the later site-publication task. This artifact records evidence for Main's review; Task 6 remains `[~]`.
