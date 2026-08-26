# Sprint 78 Atomic Task 78.17 — Transformer Runtime SRP Closure

Status: complete (pure internal refactor and test-only failure-path coverage; no changelog entry).

## Responsibility map

- `src/latent_anything/integrations/transformer_lm.py` remains the stable public facade. It owns request/result dataclasses, validation, model provenance, latent-space descriptors, high-level orchestration, public NumPy result conversion, and compatibility wrappers for logit-lens helpers.
- `src/latent_anything/_transformer_runtime.py` owns the model-bound runtime lifecycle: tokenization, device placement, native forward execution, intervention callback construction, `ActivationCaptureSession` scope, native hidden-state extraction, final-logit conversion, lens/rank intermediate assembly, and read-only NumPy boundaries.
- `src/latent_anything/_transformer_backend.py` owns lazy optional Transformers loading and tokenization.
- `src/latent_anything/_transformer_analysis.py` owns pure logit-lens, probability, top-token, and rank calculations.

The dependency direction is facade → runtime/backend/analysis. No generic generative-model Protocol, public torch/Transformers type, new dependency cycle, network model acquisition, or remote CUDA execution was introduced.

## Metrics

| Surface | Baseline | Final |
| --- | ---: | ---: |
| `integrations/transformer_lm.py` LOC / AST | 799 / 2,478 | 674 / 1,832 |
| `TransformerLMIntegration` LOC | 461 | 337 |
| `generate` LOC | 191 | 67 |
| `_transformer_runtime.py` LOC / AST | — | 163 / 1,028 |

The facade and generation method are materially smaller while retaining a cohesive public orchestration boundary; runtime-specific torch and hook concerns are isolated in the private module.

## Compatibility and parity evidence

- `generate` signature, public dataclass fields/module identities, provenance, and result schemas remain unchanged.
- Layer selection/indexing, native `output_hidden_states=True`, direct final-layer-normalization/LM-head lens, top-k tokens, rank trajectories, logits, hidden states, and read-only NumPy arrays remain equivalent on the fake backend.
- Intervention direction/strength/token-index semantics, seeded behavior, hook order, and cleanup on both success and model-forward failure are covered.
- Lazy optional imports and existing error boundaries remain isolated; no real model or network lane was run.

## Validation

- Focused transformer suite: `39 passed`.
- Focused transformer/TCAV/IG/probe/SAE/config/registry suite: `270 passed, 5 skipped, 19 warnings`.
- Full default pytest: `1534 passed, 36 skipped, 39 warnings`.
- Ruff check: pass.
- Ruff format check: pass.
- Strict Pyright on `src` and `tests`: `0 errors, 0 warnings, 0 informations`.
- Final `git diff --check`: pass (normal LF/CRLF conversion warnings only).
- Final graphify: `10,568 nodes / 20,472 edges / 936 communities`; known warning: 50 JSON files produce zero graph nodes and remain absent from the code graph.

## Review verdict

PASS. No source changes outside 78.17, model download, network access, remote CUDA, commit, or push was performed.
