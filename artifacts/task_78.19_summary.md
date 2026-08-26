# Sprint 78 Atomic Task 78.19 — SmolVLA Query Runtime SRP Closure

Status: complete (pure internal runtime extraction and compatibility snapshot coverage; no changelog entry).

## Responsibility map

- `src/latent_anything/integrations/lerobot_smolvla.py` remains the concrete public `SmolVLAPolicyAdapter` facade. It owns public schemas, policy validation, preprocessing, queue-aware official `select_action` orchestration, postprocessing, provenance/result assembly, and compatibility wrappers.
- `src/latent_anything/_lerobot_smolvla_runtime.py` owns the SmolVLA-specific model-bound query session: ordered hook registration/removal, capture parsing, camera/token offsets, four seam callbacks, additive action-expert intervention, fixed-noise conversion, and raw query result assembly.
- Existing `_lerobot_smolvla_loader.py` and `_lerobot_smolvla_metrics.py` retain checkpoint construction and measurement/report responsibilities from task 78.18.

The runtime is intentionally SmolVLA-specific. No ACT/Diffusion sharing, cross-policy Protocol, generic hook abstraction, LeRobot reimplementation, or public torch surface was introduced. The public facade delegates to the private runtime while preserving the private `_SmolVLAHookSession` compatibility alias.

## Metrics

| Surface | Baseline | Final |
| --- | ---: | ---: |
| `integrations/lerobot_smolvla.py` LOC / AST | 1,000 / 5,245 | 717 / 2,977 |
| `SmolVLAPolicyAdapter` LOC / methods | 330 / 10 | 224 / 10 |
| `select_action` LOC | 165 | 59 |
| `_lerobot_smolvla_runtime.py` LOC / AST | — | 282 / 1,535 |

The facade and public action method are materially smaller; lifecycle-specific hook/capture/intervention responsibilities now have one focused owner.

## Compatibility and parity evidence

- Public `select_action` signature, factory/result schemas, dataclass fields/module identities, and private hook-session alias are covered by snapshots.
- Hook registration order remains vision → language → state → action expert, with exact camera names, prefix offsets, denoising-step indices, capture call indices, and provenance metadata.
- Queue hits still execute no model query and produce no captures. Fixed NumPy noise, postprocessing, action arrays, and strength-zero bit identity remain unchanged.
- Expert direction shape validation, bounded intervention semantics, action-expert dimension errors, unknown-location errors, and failure-path hook removal remain covered.
- No real checkpoint, network, or CUDA lane was run; those remain marked opt-in lanes.

## Validation

- Focused SmolVLA/LeRobot bridge/benchmark/recorder suite: `61 passed, 3 skipped`.
- Full default pytest: `1535 passed, 36 skipped, 39 warnings`.
- Ruff check: pass.
- Ruff format check: pass.
- Strict Pyright on changed source/tests: `0 errors, 0 warnings, 0 informations`.
- Final `git diff --check`: pass (normal Git LF/CRLF conversion warnings only).
- Final graphify: `10,620 nodes / 20,610 edges / 933 communities`; known warning: 50 JSON files produce zero graph nodes and remain absent from the code graph.

## Review verdict

PASS. No model download, network access, remote CUDA, commit, or push was performed.
