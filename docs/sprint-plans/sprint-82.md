# Sprint 82 Plan

Status: In progress

## Sprint Goal

Publish a task-oriented, English-language guide to the released Latent Anything `1.0.0` API, with reproducible AE/VAE/hidden-state entry paths, clear method-selection and evidence boundaries, a discoverable API coverage map, and a public GitHub Pages site that preserves the existing theory site.

## Publication contract

- The package documentation lives at `https://triet4p.github.io/latent-anything/api/`; the existing theory site retains its root URL, content, and deployment path. A single deploy must publish both trees atomically; never overwrite the theory tree with an API-only `gh-pages` push. Public availability requires a successful authorized deployment and an HTTP/browser check, not merely a local build.
- Pages and examples use the published `1.0.0` contract (`artifacts/api_freeze_snapshot_1.0.0.json`), not later source-only APIs. No changes to the immutable `v1.0.0` package or `docs-v1.0.0-r1` tag. Installation examples target public PyPI; optional extras are explicit.
- Usage examples distinguish API capability from independently validated model/data claims. The ordinary-DL encoder and transformer cases remain the bounded validated diagnostic claims; SmolVLA is blocked and non-gating, and GPU/CUDA and arbitrary-model diagnostic claims are excluded. `DiagnosticWorkflow` is not a stable public import.
- Each instructional page answers when to use the API, prerequisites, input/output shapes and semantics, a runnable example with observable result, limits/failure modes, and links to the frozen API/reference where appropriate. Keep examples small and CPU-first; separately pin/cache any heavyweight optional model.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done. Only Main marks [x] after a passing evidence review. Each task writes `artifacts/sprint-82/task-<N>.md`; Main runs the final differential review after all nine evidence gates.

- [x] **Task 1 — Public-guide foundation:** Establish the `/api/` guide source/build configuration and user-facing landing/navigation page, using a local strict MkDocs build, without changing theory deployment. Record the publication boundary and page-writing conventions.
- [x] **Task 2 — Latent primitives and generic AE:** Document `LatentSpace`, `LatentValue`, `Trajectory`, adapter protocols, and a small user-owned AE wrapper; show shapes, provenance/coordinate identity, decode capability differences, and a runnable CPU path from encode to inspection.
- [x] **Task 3 — VAE paths:** Document trainable `VAE`/`ConvVAE` and the optional pretrained Diffusers VAE seam, including fit versus encode/decode, latent mean versus sample, scaling/axis conventions, optional dependencies, and bounded evidence; run the CPU local example and any feasible optional smoke without network acquisition.
- [x] **Task 4 — Deep-learning hidden-state path:** Document adapting a decoder-free hidden-state/transformer model and distinguish ordinary API usage from the pinned encoder and transformer diagnostic evidence. Link executable frozen examples; verify a lightweight CPU hidden-state path without implying arbitrary-model auto-diagnosis or promoting an internal coordinator.
- [x] **Task 5 — Inspect and explain:** Create a question-to-method guide covering projection/PCA/UMAP, probes, clustering/density, SAE/TCAV/Integrated Gradients and temporal analysis; show representative runnable CPU workflows, required labels/controls, output interpretation and limits, with links for specialist methods.
- [x] **Task 6 — Intervene and compare:** Document `Lerp`, steering, activation patching, subspace projection/removal and intervention pipelines; show one runnable control-aware example and explicitly distinguish latent edits, model-forward interventions, flat/structured adapter requirements, and causal versus observational claims.
- [x] **Task 7 — Compose, persist and extend:** Document the analysis/intervention/rollout pipeline differences, `ObjectSpec`/registry/plugin usage, runtime batching/cache/profiling/streaming/async, run records and portable results. Provide runnable bounded examples plus links to existing plugin/storage/runtime specialist guides; preserve serialization trust boundaries.
- [x] **Task 8 — Reference and specialist navigation:** Create the release-snapshot-derived 1.0 API coverage map and method-selection index. Classify all 211 canonical-stable top-level entries by instructional entry, supporting API, or specialist reference; account for submodule adapter/method exports, configs/results, CLI, optional extras and sync/async without duplicating signatures. Link advanced geometry/world-model/3D/tracking routes and explicitly identify excluded/blocked support claims.
- [~] **Task 9 — Publish and verify the site:** Integrate the new guide into a safe single `gh-pages` deployment that retains the existing theory site at `/` and serves guide pages at `/api/`; wire change detection and strict documentation checks, link entry points from README/docs, smoke the built pages and links, and record public deployment plus HTTP/browser checks if repository authorization is available. If publication access is unavailable, document the exact blocker and do not claim the site is live.

## Notes / Blockers

The existing `mkdocs.yml` builds `latent-anything-theory` at the repository Pages root; `.github/workflows/deploy-latent-anything-theory.yml` deploys `.gh-pages-build` with `force_orphan: true` on `theory-v*` tags or manual dispatch. A separate docs-only workflow targeting the same branch would erase the theory site. Task 9 must consolidate or otherwise prove preservation before any deploy. Do not edit GitHub Pages settings or force-push without established authorization. Publication success is not inferred from workflow configuration alone.

A task-level evidence review is required before assigning the next task. All flow workers/reviewers are blocking; use the default `task` worker unless a concrete correctness-invariant escalation justifies `hard-task`. No sprint-wide validation until all task edits have landed; run the strict gate and on-site smoke at closure. Pending external publication remains explicitly blocked rather than silently downgraded to local-only completion.
