# Sprint 82 Task 2 — Latent primitives and generic AE

**Task status:** `[~]` (Main owns the evidence review and status change)

## Summary

Added an English `1.0.0` public guide for `LatentSpace`, `LatentValue`, `Trajectory`, and the structural `ModelAdapter` / `DecodableAdapter` / `FlatBatchDecodableAdapter` capability split. The guide explains flat and structured point shapes, leading batch/sequence axes, caller-supplied coordinate identity and incompatible checkpoints, and a runnable CPU adapter around a pre-existing user-owned model object. It explicitly says the fixed example weights are illustrative and untrained, and makes no arbitrary-autoencoder diagnostic claim. The landing route and API-guide navigation now point to the created page.

## Files changed

- `docs/public/latent-primitives.md` — new shapes, provenance, capability, runnable example, limits, and frozen-contract guide.
- `docs/public/index.md` — added the latent primitives guide to the encoder route.
- `mkdocs-api.yml` — added the created page to API-guide navigation.
- `graphify-out/graph.json` — scoped deterministic indexing of the Task 2 guide and updated landing page.
- `graphify-out/manifest.json` — added the guide entry and refreshed the landing-page AST hash with a scoped manifest update.
- `artifacts/sprint-82/task-2.md` — this evidence record.

## Verification

- **Runnable page example — PASS.** Extracted the Python block from `docs/public/latent-primitives.md` and executed that exact source with `uv run --locked python -c "exec(<extracted block>)"`; exit status 0. Observed:

  ```text
  encode: (3, 2); first=[1.0, 2.0]
  LatentValue: (3, 2); identity=example:tiny-linear-ae:encoder::tiny-linear-ae::demo-v1
  leading axes: (1, 3, 2); batch_shape=(1, 3)
  Trajectory: (3, 2); decoded: (3, 3)
  protocols: True True True
  same shape, different checkpoint: arithmetic rejected
  ```

- **Strict API-guide build — PASS.** Ran from the repository root:

  ```text
  uv run --locked --extra docs mkdocs build --strict --config-file mkdocs-api.yml
  ```

  MkDocs 1.6.1 / Material 9.7.7 built `.api-pages-build/` successfully in 0.68 seconds with exit status 0. The Material MkDocs 2.0 announcement banner appeared; no strict-mode build errors or warnings were reported.

- **Internal route and rendered-page smoke — PASS.** Served the guide locally with `uv run --locked --extra docs mkdocs serve --config-file mkdocs-api.yml --dev-addr 127.0.0.1:8765` (HTTP 200, local canonical route `http://127.0.0.1:8765/latent-anything/api/`). In Chromium, opened the landing page, clicked the visible `Latent primitives and adapters` route, and confirmed the destination URL `/latent-anything/api/latent-primitives/`, title `Latent primitives and adapters - Latent Anything API Guide`, and H1 `Latent primitives and generic autoencoder adapters`. The rendered page contained the structured-shape table and expected example output; a viewport screenshot showed the page, navigation, and table of contents. The preview was stopped after the smoke. This is local output only, not a public deployment check.

## Graph update

**Scoped deterministic update — PASS.** Read-only inspection of installed `graphifyy 0.9.32` confirmed that `graphify.extract.extract()` dispatches `.md` inputs to the deterministic `extract_markdown()` extractor, then applies canonical repo-relative file IDs and `_origin: ast`. I ran it with only `docs/public/latent-primitives.md` and `docs/public/index.md`, an explicit repository root, and a temporary AST cache outside the repository. It emitted 15 nodes and 15 edges without semantic extraction or a corpus scan. `graphify.build.build_merge([extraction], graph_path=..., root=..., dedup=False)` replaced only these two source files' AST contributions; `graphify.export.to_json()` atomically wrote the merged graph. Existing community metadata on stable index node IDs and the original build stamp were retained.

**Observed graph counts:** 16,810 nodes / 39,418 edges / 38 hyperedges before; 16,819 nodes / 39,427 edges / 38 hyperedges after. The graph is undirected and non-multigraph: the two reciprocal Markdown `references` links coalesce to one edge. All 16,804 unrelated node records and 39,413 unrelated edge records remain attribute-for-attribute identical; all hyperedges and graph metadata are unchanged.

**Manifest and derived outputs:** Used `graphify.detect.save_manifest(kind="ast", root=..., scan_corpus=None)` with only the two Markdown paths. The manifest grew from 1,824 to 1,825 entries: `docs/public/latent-primitives.md` was added with AST hash `2f2c2cfdbb2444352032e03d9ed06623`, and `docs/public/index.md` was refreshed with AST hash `b98e795d8eddb30f7f82cbc4dfb4fa48`; both have an empty semantic hash. All 1,823 unrelated entries were preserved exactly; post-update SHA-256 is `647e6523bbf142681f093c8de4cca36f5ee454495a23486884e4cf6fd1d3ea4e`. `.graphify_labels.json`, `graph.html`, and `GRAPH_REPORT.md` were left unchanged; no reclustering or relabeling was run. The 1,059 existing communities and labels remain intact. The six existing index nodes retain community 353; the nine new guide nodes have no community assignment until a later clustering run.

`mkdocs-api.yml` was not extracted: the installed deterministic dispatcher returns no extractor for it, and it is not a recognized package manifest. Its existing manifest entry was intentionally not stamped as AST-processed; all unrelated entries were preserved.

**Indexed source nodes** (`source_file`, node ID, label, source location):

| Source file | Node ID | Label | Location |
| --- | --- | --- | --- |
| `docs/public/latent-primitives.md` | `docs_public_latent_primitives` | `latent-primitives.md` | L1 |
| `docs/public/latent-primitives.md` | `docs_public_latent_primitives_latent_primitives_and_generic_autoencoder_adapters` | Latent primitives and generic autoencoder adapters | L1 |
| `docs/public/latent-primitives.md` | `docs_public_latent_primitives_when_to_use_this_api` | When to use this API | L3 |
| `docs/public/latent-primitives.md` | `docs_public_latent_primitives_prerequisites` | Prerequisites | L7 |
| `docs/public/latent-primitives.md` | `docs_public_latent_primitives_shapes_and_coordinate_meaning` | Shapes and coordinate meaning | L17 |
| `docs/public/latent-primitives.md` | `docs_public_latent_primitives_coordinate_identity_and_provenance` | Coordinate identity and provenance | L35 |
| `docs/public/latent-primitives.md` | `docs_public_latent_primitives_adapter_capabilities` | Adapter capabilities | L41 |
| `docs/public/latent-primitives.md` | `docs_public_latent_primitives_runnable_cpu_example_wrap_an_existing_model_object` | Runnable CPU example: wrap an existing model object | L53 |
| `docs/public/latent-primitives.md` | `docs_public_latent_primitives_contract_and_next_steps` | Contract and next steps | L176 |
| `docs/public/index.md` | `docs_public_index` | `index.md` | L1 |
| `docs/public/index.md` | `docs_public_index_latent_anything_1_0_0_api_guide` | Latent Anything 1.0.0 API Guide | L1 |
| `docs/public/index.md` | `docs_public_index_choose_a_route` | Choose a route | L8 |
| `docs/public/index.md` | `docs_public_index_start_from_the_released_contract` | Start from the released contract | L16 |
| `docs/public/index.md` | `docs_public_index_page_writing_conventions` | Page-writing conventions | L28 |
| `docs/public/index.md` | `docs_public_index_build_and_preview_locally` | Build and preview locally | L41 |

**Remaining findings:** none for this graph update. `graph.json` is current for the two Task 2 Markdown sources; aggregate HTML/report and community assignments were intentionally not rebuilt.
## Limits

The example demonstrates declared NumPy CPU shapes, adapter delegation, structural protocol conformance, inspection, and rejection of arithmetic across two declared weight revisions. Its fixed matrices are not trained weights or evidence of reconstruction quality. It does not validate arbitrary autoencoders, datasets, or hardware. Public publication and theory-site deployment were not attempted.
