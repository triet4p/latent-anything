# Sprint 78 Atomic Task 78.18 — SmolVLA Adapter SRP Refactor

Status: complete (pure internal responsibility extraction and API/result snapshot coverage; no changelog entry).

## Responsibility map

- `src/latent_anything/integrations/lerobot_smolvla.py` remains the concrete public `SmolVLAPolicyAdapter` facade. It owns checkpoint/result dataclasses, public metadata and latent-space descriptors, policy validation, the official `preprocess -> select_action -> postprocess` lifecycle, four capture seams, queue-aware action selection, additive expert intervention, and exception-safe hook scope.
- `src/latent_anything/_lerobot_smolvla_loader.py` owns pinned SmolVLA configuration, dataset metadata resolution, official `LeRobotAPI.make_policy` / `make_pre_post_processors` construction, camera rename mapping, device override, and provenance context assembly.
- `src/latent_anything/_lerobot_smolvla_metrics.py` owns policy-specific intervention measurement/report assembly: induced action direction, on/off-target decomposition, representation drift, first-step drift, prompt sensitivity, camera-order sensitivity, and NumPy metric conversion.

The split is SmolVLA-specific; no cross-policy Protocol, generic helper abstraction, LeRobot reimplementation, or ACT/Diffusion behavior was introduced. Runtime dependencies remain one-way from the facade to the private loader/metrics modules and the existing LeRobot bridge.

## Metrics

| Surface | Baseline | Final |
| --- | ---: | ---: |
| `integrations/lerobot_smolvla.py` LOC / AST | 1,000 / 5,245 | 879 / 4,177 |
| `SmolVLAPolicyAdapter` LOC / methods | 330 / 10 | 330 / 10 |
| Largest adapter method (`select_action`) | 165 LOC | 165 LOC |
| `_lerobot_smolvla_loader.py` | — | 72 LOC |
| `_lerobot_smolvla_metrics.py` | — | 171 LOC |

The public adapter lifecycle remains intact while checkpoint loading and measurement/report concerns are physically separated; no public/private compatibility seam was removed.

## Compatibility and parity evidence

- Public exports, `SmolVLAPolicyAdapter.select_action` and `load_smolvla_policy` signatures, dataclass fields/module identity, checkpoint/provenance metadata, and NumPy result boundaries are covered by an API/result schema snapshot.
- Offline fake-policy parity covers official preprocessor/policy/postprocessor flow, camera rename mapping, four capture locations and token metadata, action queue hits (no query/no captures), fixed-noise actions, strength-zero bit identity, bounded direction validation, additive intervention, hook cleanup after policy failure, and measurement metrics.
- Loader parity retains pinned revisions, official factory calls, dataset stats, device override, and raw LeRobot policy ownership. Real checkpoint/network/CUDA lanes remain opt-in and were not run.

## Validation

- Focused SmolVLA/LeRobot bridge/benchmark/recorder suite: `61 passed, 3 skipped`.
- Full default pytest: `1535 passed, 36 skipped, 39 warnings`.
- Ruff check: pass.
- Ruff format check: pass.
- Strict Pyright on changed source/tests: `0 errors, 0 warnings, 0 informations`.
- Final `git diff --check`: pass (normal Git LF/CRLF conversion warnings only).
- Final graphify: `10,593 nodes / 20,529 edges / 932 communities`; known warning: 50 JSON files produce zero graph nodes and remain absent from the code graph.

## Review verdict

PASS. No model download, network access, remote CUDA, commit, or push was performed.
