# Sprint 82 Task 5 — Inspect and explain representations

**Task status:** `[~]` (Main owns evidence review and status change)

## Summary

Added a question-led guide for choosing among PCA/UMAP projections, labeled linear and nonlinear probes, KMeans clustering, GMM-based density/OOD scoring, SAE feature evaluation, TCAV, activation-space Integrated Gradients, and temporal trajectory methods. It explains inputs and fitting, labels and controls, result semantics, and limits without duplicating the full API inventory. The page distinguishes API capability from the bounded diagnostic evidence and explicitly marks unexecuted advanced method paths as conditional guidance.

The CPU example uses the local scikit-learn digits 0/1 subset. It exercises a PCA projection, held-out `LinearProbe`, and KMeans with post-fit adjusted-Rand comparison; it is not a general model or performance validation.

## Files changed

- `docs/public/inspect-explain.md` — new task-oriented guide, decision table, method assumptions/result reading/controls/limits, CPU example, and release/specialist links.
- `docs/public/index.md` — linked the guide from the inspection/explanation route.
- `mkdocs-api.yml` — added the guide to the API site's navigation.
- `graphify-out/graph.json` — scoped deterministic AST merge for the new guide and updated landing page.
- `graphify-out/manifest.json` — refreshed AST stamps for those two Markdown sources.
- `artifacts/sprint-82/task-5.md` — this evidence record.

No package API, release tag, theory-site configuration, sprint status, or deployment was changed.

## Verification

- **Runnable CPU example — PASS.** Extracted and executed the Python block from `docs/public/inspect-explain.md` with:

  ```text
  uv run --locked python -c "from pathlib import Path; p=Path('docs/public/inspect-explain.md').read_text(encoding='utf-8'); b=p.split(chr(96)*3+'python',1)[1].split(chr(96)*3,1)[0]; exec(b, {'__name__':'__main__'})"
  ```

  Exit status 0. Observed:

  ```text
  digits 0/1: 360 examples, 64 input features
  PCA: coordinates=(360, 2), explained_variance=0.593
  LinearProbe: held-out accuracy=1.000
  KMeans: sizes=[182, 178], ARI=0.978
  ```

  Assertions require two real classes, the projected shape and nontrivial explained variance, held-out probe accuracy of at least 0.95, and cluster agreement above chance (`ARI > 0.50`). The guide states that PCA uses the full subset only for a descriptive projection and that KMeans ARI is post-fit descriptive agreement on the same rows, not held-out prediction.

- **Strict API-guide build — PASS.** Ran `uv run --locked --extra docs mkdocs build --strict --config-file mkdocs-api.yml`; MkDocs built `.api-pages-build/` successfully with exit status 0. The Material for MkDocs 2.0 announcement banner appeared; the build completed without MkDocs build errors.

- **Local HTTP and Chromium surface — PASS.** Served the final guide at port `8768`. An HTTP fetch returned status `200` for `http://127.0.0.1:8768/latent-anything/api/inspect-explain/` and the response contained the guide heading and release-source links. In Chromium, opened the landing page, clicked the visible `Inspect and explain representations` route, and confirmed URL `/latent-anything/api/inspect-explain/`, title `Inspect and explain representations - Latent Anything API Guide`, and H1 `Choose a method to inspect or explain a representation`. The decision table and rendered CPU example were visually inspected.

- **Public deployment — not attempted.** The local smoke does not establish that the intended public `/api/` URL is deployed or available.

## Graph update

**Scoped deterministic AST update — PASS.** Used the graph interpreter recorded in `graphify-out/.graphify_python` and installed `graphifyy 0.9.68`. Only `docs/public/inspect-explain.md` and `docs/public/index.md` were extraction inputs; no semantic extraction or reclustering was run. The temporary-copy dry run passed before the staged graph and manifest were committed.

The merge used deterministic `extract(..., parallel=False)`, `build_merge([extraction], graph_path=..., root=root, dedup=False)`, and `to_json` with the existing `built_at_commit` and community labels, followed by `save_manifest(..., kind='ast', root=root, scan_corpus=None)`. All unrelated graph node records and links were preserved attribute-for-attribute; graph metadata, flags, and all hyperedges were unchanged.

- Graph counts: **16,837 nodes / 39,446 links / 38 hyperedges** → **16,853 / 39,463 / 38**.
- Preserved unrelated records: **16,831 nodes and 39,438 links**.
- Manifest entries: **1,827 → 1,828**; all unrelated entries were unchanged.
- `docs/public/index.md` AST hash: `4fbe4d1eaed41232d5228764b4dad574` (semantic hash empty).
- `docs/public/inspect-explain.md` AST hash: `006b24829c069ed3281f521cafbb38f4` (semantic hash empty).
  A final scoped AST refresh after adding release-source links left graph counts at 16,853 nodes / 39,463 links / 38 hyperedges and the manifest at 1,828 entries.
- `graphify-out/GRAPH_REPORT.md` and `graphify-out/graph.html` were intentionally not regenerated; this was a scoped merge, not a reclustering run.

## Evidence limits and open findings

- Only PCA, the linear probe, and KMeans on the bundled digits 0/1 subset were executed. UMAP, MLP probes, GMM density/OOD evaluation, SAE feature evaluation, TCAV, Integrated Gradients, and temporal methods are documented as conditional paths and were not run or represented as verified examples.
- The page describes the two accepted Sprint 80 diagnoses and separately frozen transformer stability supplement as bounded, model/task-specific evidence. It does not equate projection, probe separability, or method availability with causal explanation or task-performance validation. Arbitrary model/dataset and GPU/CUDA claims remain out of scope; SmolVLA remains blocked and non-gating.
- No task-level implementation blocker remains. Public Pages deployment and its HTTP verification belong to the later sprint publication task. This artifact is evidence for Main's review, not a verdict; Task 5 remains `[~]`.
