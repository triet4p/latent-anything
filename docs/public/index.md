# Latent Anything 1.0.0 API Guide

Use this guide to choose a starting point for working with model representations, inspect or change latents, or connect analyses to your own code. It documents the **published `1.0.0` API**, not unreleased additions from the current source tree.

!!! note "Publication status"
    This guide is published at [the `/api/` URL](https://triet4p.github.io/latent-anything/api/) alongside the existing theory site at the Pages root. A local build does not publish changes or verify the public site; use the combined deployment workflow to publish both trees atomically.

## Choose a route

- **I have an encoder and want to represent or inspect its outputs.** Start with the [latent primitives and generic autoencoder guide](latent-primitives.md), then use the [frozen API reference](https://github.com/triet4p/latent-anything/blob/main/docs/API_REFERENCE.md) for the stable surface and the [AI engineer guide](https://github.com/triet4p/latent-anything/blob/main/docs/AI_ENGINEER_GUIDE.md) for the bounded encoder example and its limits.
- **I want to train a VAE or use a pretrained Diffusers VAE.** Start with [VAE paths](vae-paths.md) for built-in `VAE`/`ConvVAE`, fit-versus-encode semantics, and the optional `AutoencoderKL` seam; consult the [frozen API reference](https://github.com/triet4p/latent-anything/blob/v1.0.0/docs/API_REFERENCE.md) for the released core surface and the [Diffusers integration guide](https://github.com/triet4p/latent-anything/blob/main/docs/DIFFUSERS_INTEGRATION.md) for its pinned-cache evidence path.
- **I have a decoder-free model and want to inspect a hidden layer.** Start with [decoder-free hidden-state paths](hidden-state-path.md) for adapting a real hidden output, shapes, identity, and capture limits; the [AI engineer guide](https://github.com/triet4p/latent-anything/blob/main/docs/AI_ENGINEER_GUIDE.md) links the separate bounded encoder-v3 and transformer target-evidence-v2 diagnostics.
- **I need to inspect, explain, compare, or intervene on representations.** Start with [inspect and explain representations](inspect-explain.md) to choose among projections, probes, clustering/OOD scores, sparse features, attribution, and temporal analysis; use the [frozen API reference](https://github.com/triet4p/latent-anything/blob/v1.0.0/docs/API_REFERENCE.md) for released methods and the [API compatibility ledger](https://github.com/triet4p/latent-anything/blob/main/docs/API_COMPATIBILITY.md) for spelling and migration compatibility.
- **I want to interpolate, steer, remove a subspace, or patch a representation.** Start with the [intervention and comparison guide](intervene-compare.md) for latent edits, flat-batch adapter requirements, identity/off-target controls, and causal-claim boundaries; use the bounded [AI engineer diagnoses](https://github.com/triet4p/latent-anything/blob/main/docs/AI_ENGINEER_GUIDE.md) for model-specific intervention and run-comparison evidence.
- **I need to compose, persist, or extend a pipeline.** Start with [compose, persist, and extend](compose-persist-extend.md) for pipeline lifecycles, config, runtime boundaries, run records, and portable results; use the [pipeline reference](https://github.com/triet4p/latent-anything/blob/main/docs/PIPELINES.md), [plugin author guide](https://github.com/triet4p/latent-anything/blob/main/docs/PLUGIN_AUTHOR_GUIDE.md), and [portable-artifact guide](https://github.com/triet4p/latent-anything/blob/main/docs/PORTABLE_ARTIFACTS.md) for specialist detail.
- **I need a release-wide lookup or specialist route.** Use the [method-selection index and 1.0.0 API coverage map](api-reference-index.md) to find the canonical root export, distinguish worked examples from supporting/reference APIs, and route geometry, 3D, world-model, planning, discrete-latent, and tracking questions to their authoritative guides.

## Start from the released contract

Install the released package from PyPI, using Python `>=3.12,<3.15`:

```bash
python -m pip install "latent-anything==1.0.0"
```

The [published API-freeze snapshot](https://github.com/triet4p/latent-anything/blob/v1.0.0/artifacts/api_freeze_snapshot_1.0.0.json) is the authority for the `1.0.0` surface. The `v1.0.0` release tag is immutable; later documentation or source-only changes on `main` must not be presented as part of that release. Read the [support, versioning, and migration policy](https://github.com/triet4p/latent-anything/blob/main/docs/SUPPORT_POLICY.md) for compatibility commitments, optional-dependency boundaries, and artifact migration rules.

An API capability does not establish that an arbitrary model, dataset, or device has been validated. The current diagnostic evidence is limited to the ordinary-DL encoder and transformer cases. SmolVLA is **BLOCKED** and non-gating; arbitrary-model diagnostic claims and GPU/CUDA claims are excluded. `DiagnosticWorkflow` is not a stable public import. Treat the [AI engineer guide](https://github.com/triet4p/latent-anything/blob/main/docs/AI_ENGINEER_GUIDE.md) as evidence for its named frozen cases only.

## Page-writing conventions

Each instructional page should answer these questions, in this order:

1. **When should I use this API?** State the task and what it does not solve.
2. **What do I need first?** Name the supported Python range, base install, and any explicitly required optional extra or pinned model/data input.
3. **What goes in and comes out?** Show input/output shapes, axis meaning, identity or provenance requirements, and decode/intervention semantics that affect the result.
4. **How do I run it?** Give a small CPU-first example with an observable expected result; pin or separately cache heavyweight optional models.
5. **Where can it fail, and what is not established?** Explain limits and failure modes, and distinguish API capability from evidence for specific models, datasets, or hardware.
6. **Where is the released contract?** Link to the frozen API reference or snapshot and relevant specialist/support documentation.

Keep new guide pages under `docs/public/`. Use relative Markdown links only for pages that exist in that directory, so MkDocs resolves them within the eventual `/api/` tree. Link to existing repository references outside `docs/public/` with their explicit GitHub URLs. Add a page to `nav` in `mkdocs-api.yml` after its source exists, and link it from this landing page when it adds a real task-oriented route.

## Build and preview locally

From the repository root, build the guide with the locked documentation dependencies:

```bash
uv run --locked --extra docs mkdocs build --strict --config-file mkdocs-api.yml
```

The generated site is written to `.api-pages-build/`, separate from the theory site's `.gh-pages-build/`. To preview locally instead, run:

```bash
uv run --locked --extra docs mkdocs serve --config-file mkdocs-api.yml
```

A successful local build verifies this source/configuration only. It does not publish the guide, alter the theory deployment, or establish that the intended Pages URL is serving it.
