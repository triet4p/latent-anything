# Sprint 82 Task 3 — VAE paths

**Task status:** `[~]` (Main owns the evidence review and status change)

## Summary

Added an English public guide for training the released `VAE` and `ConvVAE`, then separating `fit()` from deterministic posterior-mean `encode()` and `decode()`. It includes a small runnable CPU example with explicit shape and repeatability assertions, plus an optional pinned Diffusers `AutoencoderKL` seam that distinguishes NCHW arrays from the adapter's NHWC `LatentValue` and documents symmetric checkpoint scaling.

The optional example requires the `diffusers` extra and the already-cached `stabilityai/sd-vae-ft-mse` revision `31f26fdeee1355a5c34592e401dd41e45d25a493`. It sets `HF_HUB_OFFLINE=1` because the adapter's `from_pretrained()` call does not itself set `local_files_only=True`. The optional model example was not run and no model files were acquired. D1/D2 evidence is scoped to the referenced test and existing benchmark artifacts; the page makes no general pretrained-quality claim.

## Files changed

- `docs/public/vae-paths.md` — task-oriented built-in VAE/ConvVAE guide, runnable CPU code, optional offline Diffusers example, shapes/scaling/sampling, evidence boundaries, and release/specialist links.
- `docs/public/index.md` — added the VAE task route and updated the local-page/navigation convention.
- `mkdocs-api.yml` — added the page to API-guide navigation after creating its source.
- `graphify-out/graph.json` — scoped deterministic Markdown AST update for the guide and updated landing page.
- `graphify-out/manifest.json` — scoped AST stamps for those same two Markdown files.
- `artifacts/sprint-82/task-3.md` — this evidence record.

## Verification

- **Runnable built-in CPU example — PASS.** Extracted the first Python block from `docs/public/vae-paths.md` and ran it with `uv run --locked python -c "from pathlib import Path; p=Path('docs/public/vae-paths.md').read_text(encoding='utf-8'); b=p.split(chr(96)*3+'python',1)[1].split(chr(96)*3,1)[0]; exec(b, {'__name__':'__main__'})"`; exit status 0. Observed:

  ```text
  VAE: input=(6, 3), latent_mean=(6, 2), decoded=(6, 3); encode deterministic
  ConvVAE: input=(4, 1, 8, 8), latent_mean=(4, 3), decoded=(4, 1, 8, 8); encode deterministic
  ```

- **Strict API-guide build — PASS.** Ran `uv run --locked --extra docs mkdocs build --strict --config-file mkdocs-api.yml`. MkDocs built `.api-pages-build/` successfully in 0.36 seconds with exit status 0. The Material for MkDocs 2.0 announcement banner was emitted; the strict build reported no build errors.

- **Local HTTP and browser surface — PASS.** Served the site with `uv run --locked --extra docs mkdocs serve --config-file mkdocs-api.yml --dev-addr 127.0.0.1:8765`. HTTP status was 200 for both `http://127.0.0.1:8765/latent-anything/api/` and `http://127.0.0.1:8765/latent-anything/api/vae-paths/`. In Chromium, the visible `VAE training and pretrained AutoencoderKL` route was found and clicked; the destination title was `VAE training and pretrained AutoencoderKL - Latent Anything API Guide`, H1 was `Train a VAE or connect a pretrained AutoencoderKL`, and the rendered optional section showed the offline guard and non-executed code. This verifies the local surface only; it does not claim the public `/api/` site is deployed.

- **Optional pretrained checkpoint — intentionally not executed.** The guide supplies the exact model/revision, weight digest and size, required extra, offline guard, and expected axes, while requiring the matching cache to pre-exist. No checkpoint fetch or Diffusers smoke was attempted.

## Graph update

**Scoped deterministic update — PASS.** Used installed `graphifyy 0.9.32` and only `docs/public/vae-paths.md` and `docs/public/index.md` as extraction inputs:

```text
extract(paths, root=root, cache_root=<temporary directory>, parallel=False)
build_merge([extraction], graph_path=graph_path, root=root, dedup=False)
to_json(..., built_at_commit=original['built_at_commit'], community_labels=labels)
save_manifest({'document': [vae_paths, index]}, kind='ast', root=root, scan_corpus=None)
```

A temporary-copy dry run passed first. The final deterministic extraction produced 14 AST nodes and 15 edges. Graph counts changed from 16,819 nodes / 39,427 links / 38 hyperedges to 16,827 nodes / 39,435 links / 38 hyperedges. All 16,813 unrelated node records and 39,421 unrelated link records were preserved attribute-for-attribute; all 38 hyperedges and graph metadata, including the original `built_at_commit`, were unchanged. Prior non-structural node attributes were restored with `setdefault`, preserving the landing page's existing community 353.

The manifest grew from 1,825 to 1,826 entries. `docs/public/index.md` now has AST hash `af147b321250fc787bc6d686fadd9c16`; `docs/public/vae-paths.md` was added with AST hash `4af94e4153890de76785774a67964684`. Both have empty semantic hashes, and all 1,824 unrelated manifest entries were preserved exactly. Post-update SHA-256: `graph.json` `4a8a1d0fd94ecc0559701559dd85b9c84cf4ccace27ebce2dd2478537bcedacb`; `manifest.json` `32e423c4c36689def867ae0520f094aa8bfa7a4cd36b36b0ed8947b014741a09`.

Indexed sources include:

| Source | Node ID | Label | Location |
| --- | --- | --- | --- |
| `docs/public/vae-paths.md` | `docs_public_vae_paths` | `vae-paths.md` | L1 |
| `docs/public/vae-paths.md` | `docs_public_vae_paths_train_a_vae_or_connect_a_pretrained_autoencoderkl` | Train a VAE or connect a pretrained AutoencoderKL | L1 |
| `docs/public/vae-paths.md` | `docs_public_vae_paths_when_to_use_this_api` | When to use this API | L3 |
| `docs/public/vae-paths.md` | `docs_public_vae_paths_prerequisites` | Prerequisites | L9 |
| `docs/public/vae-paths.md` | `docs_public_vae_paths_choose_the_built_in_model_by_input_shape` | Choose the built-in model by input shape | L25 |
| `docs/public/vae-paths.md` | `docs_public_vae_paths_runnable_cpu_example_fit_encode_and_decode` | Runnable CPU example: fit, encode, and decode | L36 |
| `docs/public/vae-paths.md` | `docs_public_vae_paths_optional_seam_pretrained_diffusers_autoencoderkl` | Optional seam: pretrained Diffusers AutoencoderKL | L87 |
| `docs/public/vae-paths.md` | `docs_public_vae_paths_evidence_boundaries` | Evidence boundaries | L126 |
| `docs/public/index.md` | `docs_public_index` | `index.md` | L1 |
| `docs/public/index.md` | `docs_public_index_latent_anything_1_0_0_api_guide` | Latent Anything 1.0.0 API Guide | L1 |
| `docs/public/index.md` | `docs_public_index_choose_a_route` | Choose a route | L8 |
| `docs/public/index.md` | `docs_public_index_start_from_the_released_contract` | Start from the released contract | L16 |
| `docs/public/index.md` | `docs_public_index_page_writing_conventions` | Page-writing conventions | L28 |
| `docs/public/index.md` | `docs_public_index_build_and_preview_locally` | Build and preview locally | L41 |

The new VAE guide nodes have no community assignment because this was a scoped update, not a reclustering run. `graphify-out/GRAPH_REPORT.md` and `graphify-out/graph.html` were intentionally not regenerated, following Task 2's precedent; their aggregate view may not yet include this page.

## Evidence limitations and open findings

- `artifacts/conv_vae_heldout_benchmark.json` records the accepted D2 held-out digits lane (1,437 train / 360 held out, reconstruction MSE 0.1717 versus 0.2359 zero baseline; the train-pixel-mean diagnostic is stronger at 0.0731). This remains bounded to the compact 8×8 grayscale model and that split.
- `docs/DIFFUSERS_INTEGRATION.md` distinguishes D1 fake-backend coverage from D2 pinned, cached local CPU fidelity/interpolation artifacts. These do not establish perceptual quality or a general diffusion-pipeline claim.
- `docs/VAE_EXPLANATION_BENCHMARK.md` remains the specialist contract for controlled explanation claims; this page's usage smoke does not meet that benchmark by itself.
- No implementation blocker remains. The local pinned Diffusers example is deliberately unexecuted, and public Pages deployment/public HTTP availability remains outside this task.
