# Sprint 82 Task 4 — Deep-learning hidden-state path

**Task status:** `[~]` (Main owns evidence review and status change)

## Summary

Added an English task guide for carrying decoder-free deep-model activations through the released `1.0.0` `ModelAdapter` contract. The guide distinguishes the fixed-random `HiddenStateAdapter` from a user-owned adapter for real model outputs and from the optional, version-specific `TransformerLMIntegration`; it documents sequence/layer axes, model/layer/revision identity, controlled capture, optional Transformers dependencies, and limits. A deterministic two-layer CPU Torch fixture exercises a user-owned decoder-free adapter without model downloads or training. The guide links the pinned encoder-v3 and transformer target-evidence-v2 proof scripts and accurately separates the independently gated transformer stability supplement.

No source API, frozen package surface, release tag, theory-site configuration, or sprint status was changed.

## Files changed

- `docs/public/hidden-state-path.md` — new task-oriented hidden-state guide, runnable CPU example, shape/provenance/capture guidance, optional integration boundary, and evidence links.
- `docs/public/index.md` — added the decoder-free hidden-state route to the landing page.
- `mkdocs-api.yml` — added the guide to the API site's actual navigation.
- `graphify-out/graph.json` — scoped deterministic AST refresh for the new guide and updated landing page.
- `graphify-out/manifest.json` — AST hashes refreshed for those same two Markdown files.
- `artifacts/sprint-82/task-4.md` — this evidence record.

## Verification

- **Runnable CPU example — PASS.** Extracted and executed the first Python block in `docs/public/hidden-state-path.md` with:

  ```text
  uv run --locked python -c 'from pathlib import Path; p=Path("docs/public/hidden-state-path.md").read_text(encoding="utf-8"); b=p.split("```python", 1)[1].split("```", 1)[0]; exec(b, {"__name__": "__main__"})'
  ```

  Exit status 0. Observed:

  ```text
  input_ids=(2, 3); selected hidden=(2, 3, 4)
  point=(4,); leading_axes=(2, 3)
  layer=encoder.layers.0; revision=fixed-fixture-v1
  first token hidden=[0.7616000175476074, 0.0, 0.0, 0.7616000175476074]
  ```

  This exercises the adapter protocol, selected-layer capture, NumPy boundary, `LatentValue` point/leading-axis shape, and declared layer/revision identity on the fixed fixture only. It is not evidence of general model behavior or feature quality.

- **Strict API-guide build — PASS.** Final run: `uv run --locked --extra docs mkdocs build --strict --config-file mkdocs-api.yml`. MkDocs built `.api-pages-build/` successfully with exit status 0 and no strict-mode build warnings/errors. The Material for MkDocs 2.0 announcement banner was emitted.

- **Local HTTP and Chromium route — PASS.** Started the dedicated local preview with `uv run --locked --extra docs mkdocs serve --config-file mkdocs-api.yml --dev-addr 127.0.0.1:8766`. HTTP fetches returned 200 for `/latent-anything/api/` and `/latent-anything/api/hidden-state-path/`. In Chromium, clicked the visible `Decoder-free hidden-state paths` navigation item and confirmed URL `/latent-anything/api/hidden-state-path/`, title `Decoder-free hidden-state paths - Latent Anything API Guide`, H1 `Adapt decoder-free models and inspect hidden states`, and rendered CPU example/evidence content. Captured a screenshot of the rendered page and closed the tab and preview server. This is local verification only; it does not publish `/api/`.

## Graph update

**Scoped deterministic update — PASS.** Used installed `graphifyy 0.9.32` and only `docs/public/hidden-state-path.md` plus `docs/public/index.md` as extraction inputs. The temporary-copy dry run passed before writing the graph and manifest:

```text
extract(paths, root=root, cache_root=<temporary directory>, parallel=False)
build_merge([extraction], graph_path=graph_path, root=root, dedup=False)
to_json(..., built_at_commit=original['built_at_commit'], community_labels=labels)
save_manifest({'document': [hidden_state_path, index]}, kind='ast', root=root, scan_corpus=None)
```

The deterministic extraction emitted 16 nodes and 19 edges. Counts changed from 16,827 nodes / 39,435 links / 38 hyperedges to 16,837 nodes / 39,446 links / 38 hyperedges. All 16,821 nodes and 39,428 links unrelated to these two sources were preserved attribute-for-attribute; all hyperedges, graph metadata, direction/multigraph flags, and the original `built_at_commit` were unchanged. Existing node attributes were restored with `setdefault`, preserving the landing page's existing community metadata.

The manifest now has 1,827 entries. All 1,825 unrelated entries were preserved exactly; both changed sources have empty semantic hashes:

| Source | AST hash |
| --- | --- |
| `docs/public/hidden-state-path.md` | `a3676c3b8b0e991c73b68bbb24040f71` |
| `docs/public/index.md` | `cfb393e0391e80e18ae9ebc662d30d13` |

Post-update SHA-256: `graphify-out/graph.json` `11d10e508a7607288854e0d54a15956a6bcfd3bf15afa1de2113286aef0b1321`; `graphify-out/manifest.json` `5e02c867c97211bf28714cc1f6b10eb8863c67904e4158c0f58e9a9ab0628b2c`.

Indexed hidden-state page headings include `When to use this API`, `Choose the right route`, `Prerequisites`, `Shapes, axes, and coordinate identity`, `Runnable CPU example: adapt one named hidden layer`, `Capture deliberately`, `Bounded diagnostic examples—not general model validation`, and `Contract and next steps`. This was a scoped update, not reclustering: new page nodes have no community assignment, and `graphify-out/GRAPH_REPORT.md` / `graphify-out/graph.html` were intentionally not regenerated.

## Evidence boundaries and open findings

- `TransformerLMIntegration` is absent from the frozen `1.0.0` public API surface. The guide identifies it only as a version-specific optional integration; it does not treat the Transformers extra or current source implementation as a stable import contract.
- The pinned encoder-v3 and GPT-2/Wikitext target-evidence-v2 proof replays were not rerun here. Their inputs, commands, bounded results, and limitations are linked to the accepted AI engineer guide and existing reports; the transformer replay requires the exact local model/data cache and is heavyweight. The separately frozen stability supplement remains distinct from the target-evidence-v2 record and is described as non-causal, same-source-corpus evidence.
- No task-level implementation blocker remains. Public `/api/` deployment is not claimed; publication and theory-site preservation remain under the sprint's later deployment task.
- The task remains `[~]` pending Main's evidence review; this artifact is not a review verdict.
