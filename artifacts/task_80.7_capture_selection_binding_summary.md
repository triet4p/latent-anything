# Task 80.7 — Generalize Capture Selection and Axis Metadata

## Status

**Complete.** Sprint 80 tasks 80.1–80.7 are marked `[x]`; tasks 80.8–80.29 remain pending. Historical Sprint 79 evidence was not changed.

## Summary of Work

Added `src/latent_anything/_capture_binding.py`, the smallest shared binding layer that turns a domain `CaptureSelection` plus frozen-manifest source identity into deterministic capture identities and typed axis/provenance metadata, reusing the existing `CapturedActivation`, `LatentValue`, and `Trajectory` primitives with no parallel representation container and no second configuration/serialization path.

- `bind_selection(selection, manifest, request_id)` builds a plan-level `BoundCapture` (deterministic plan identity over capture id, representation identity, manifest/model/dataset/split/revision, and sorted requested-axis selection/identity; no observed dtype/shape/device yet).
- `resolve_capture` / `resolve_captures` bind observed `CapturedActivation` records in forward execution order to `LatentValue` instances carrying complete provenance (identity, model/dataset/split/revision, capture point, ordering, dtype/shape/device, axis details, absent axes). The tensor passes straight into `LatentValue`, which owns its single copy; the binder performs no additional copy.
- `bound_trajectory(bound, value, axis=...)` projects a resolved 2D value onto an explicit `token`/`time` axis; every other axis name is an explicit non-applicability rejection, never an empty success.
- One architecture-neutral rule assigns indices from array rank alone: feature is always the last axis; sample/slice share batch axis 0; at most one of token/time occupies the sequence axis; layer/module/checkpoint are selection (non-array) axes with `axis_index=None`. No architecture-specific branching in the binder.
- Supported axes: sample, slice, token, time, checkpoint, layer, module, feature. Unknown axes, duplicate/ambiguous axes (`layer`+`module`, two sequence axes), missing required identity, undeclared-axis requests, incompatible shape/axis metadata, mixed-shape/dtype batches, duplicate capture points, empty capture lists, provenance mismatches (representation identity, source version, metadata-vs-values shape), and ordering violations all reject fail-closed via `CaptureBindingError`.
- The reviewed 80.6 nine-name top-level surface is preserved: no new top-level re-export was added (`ResultStatus` remains `diagnostics`-only).

## Both Core Selection Shapes (Same Generic Path)

- Encoder bottleneck: axes `(sample, feature, slice)` against `sprint80-core-encoder-autoencoder-collapse-v1`; resolved `(360, 4)` with indices `{sample: 0, feature: 1, slice: None}`; absent axes `['checkpoint', 'layer', 'module', 'time', 'token']`.
- Transformer hidden states: axes `(slice, token, feature, layer)` against `sprint80-core-transformer-hidden-state-probe-v1`; resolved `(4, 16, 768)` with indices `{slice: 0, token: 1, feature: 2, layer: None}`; absent axes include `sample`, `time`, `checkpoint`, `module`.
- Deterministic identity: plan identity is canonical-JSON SHA-256 over sorted axis entries plus frozen manifest provenance; permutation of requested axes yields the same plan identity; concrete identity additionally binds (location, call_index, shape, dtype).
- Provenance completeness: every resolved `LatentValue.metadata` carries `capture_identity`, `capture_id`, `source_representation_identity`/`representation_identity`, `manifest_id`, `request_id`, `model_id`, `model_revision` (+`model_version`/`revision` aliases), `dataset_id`, `dataset_revision`, `dataset_split`, `split_identity`, `axes`, `axis_details`, `absent_axes`, `manifest_axes`, `capture_point`, `ordering`, `dtype`, `shape`, `device`; `LatentValue.identity` is non-empty through the existing coordinate-identity path.

## Validation Evidence

Focused validation only (no formatter, linter, type checker, packaging, docs build, or project-wide suite was run):

```text
uv run pytest tests/test_capture_binding.py -q
7 passed in 3.04s

uv run pytest tests/test_capture_binding.py tests/test_diagnostics_request.py tests/test_sprint80_core_manifests.py -q
26 passed in 3.08s

uv run python scripts/sprint80_task80_7_smoke.py
PASS encoder 1d72f227dec7 sample/feature/slice -> (360, 4)
PASS transformer ba15bd7ded35 slice/token/feature/layer -> (4, 16, 768)
PASS deterministic-plan 36d21be40e25
PASS absent-axes encoder=['checkpoint', 'layer', 'module', 'time', 'token']
PASS trajectory token(16,768)
REJECT duplicate-ambiguous: ambiguous axes: 'layer' and 'module' name the same seam
REJECT unsupported-axis: unsupported axis: 'head'
REJECT provenance-mismatch: provenance mismatch: selection representation_identity 'openai-community-gpt2:hidden-states:layers0-11:dim768' != manifest 'conv_vae_8x8:bottleneck-mu:latent_dim=4'
REJECT incompatible-shape: incompatible shape: rank 1 carries feature only, not sample/sequence axes
exit code 0

uv run python -c "import latent_anything as la; ..."
9 ['CaptureSelection', 'ComparisonRequest', 'ControlSelection', 'DiagnosticRequest', 'DiagnosticRequestError', 'DiagnosticResult', 'DiagnosticSelection', 'InterventionRequest', 'OutputSelection']
False
```

The nine-name check confirms the exact 80.6 top-level surface is intact and `ResultStatus` is not a top-level export.

## Affected Claims

- Both core models use one architecture-neutral binding path with deterministic identity and complete provenance through the existing representation primitives.
- Applicable axes align (feature-last, batch/sequence from rank); absent axes are explicit (`absent_axes`) rather than fabricated empty success.
- Invalid/ambiguous metadata rejects fail-closed; manifest thresholds were not changed.
- No 80.8+ orchestration, detector, localization, or execution claim is made by this task.

## Negative Results and Limitations

- Rank-3 transformer values must be sliced to one sample via the existing `LatentValue` indexing before `bound_trajectory`; the binder never squeezes silently (rank-3 direct trajectory binding rejects).
- `feature` is the only axis allowed to default when absent from the manifest (`all-features`); every other undeclared requested axis rejects as a provenance mismatch rather than guessing.
- `sample`+`slice` legitimately share batch axis 0; `layer`+`module` is ambiguity and rejects because they name the same capture seam.
- Synthetic NumPy captures stand in for real model/dataset bytes in tests/smoke; real benchmark execution remains an 80.23/80.24 concern.
- Canonical identity uses the shared `canonical_json` contract (`sort_keys`, `separators=(",", ":")`, `allow_nan=False`); non-JSON metadata values reject at identity time rather than hashing approximately.

## Files Modified

- `src/latent_anything/_capture_binding.py` — new private binder (`AxisBinding`, `BoundCapture`, `bind_selection`, `resolve_capture`, `resolve_captures`, `bound_trajectory`, `CaptureBindingError`, `SUPPORTED_AXES`); no public export changes.
- `tests/test_capture_binding.py` — 7 consumer-observable tests: shared generic path + alignment + provenance, deterministic/canonical identity, explicit absent axes + trajectory gating, ordering + mixed-shape rejection, duplicate/ambiguous/unknown rejection, identity/provenance rejection, empty/duplicate-point/incompatible-rank rejection.
- `scripts/sprint80_task80_7_smoke.py` — committed direct smoke for both core selection shapes, deterministic identity, provenance, absent axes, token trajectory, and four representative rejections.

## Evidence-Review Readiness

Ready: focused binding tests prove both core selections traverse one generic path with deterministic identity and complete provenance through `LatentValue`/`Trajectory`; the committed smoke script reproduces both shapes plus ambiguity, unsupported-axis, provenance-mismatch, and incompatible-shape rejections; the nine-name 80.6 surface is verified unchanged.

## Graph Refresh Record

Final task-level step, run after implementation, focused validation, artifact, and status:

```text
graphify update .
Re-extracting code files in . (no LLM needed)...
  AST extraction: 100/257 uncached files (38%) [8 workers]
  AST extraction: 200/257 uncached files (77%) [8 workers]
  AST extraction: 257/257 uncached files (100%) [8 workers]
  warning: 248 source file(s) produced zero nodes and are absent from the graph: hooks.json, evals.json, evals.json, act_policy_representation_benchmark.json, anisotropy_benchmark.json (+243 more). A re-run will retry them (empties are no longer cached); if it persists, please report the file(s) (#1666).
[graphify watch] community set changed since labeling (1025 saved labels, 1058 communities now; renamed 178 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) -> 2026-09-21/
Graph has 13973 nodes (above 5000 limit). Building aggregated community view...
graph.html written (aggregated: 1058 community nodes, 1281 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt: 13973 nodes, 29033 edges, 1058 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
exit code 0
```

Index/freshness evidence: `graphify-out/graph.json` (2026-09-21 04:32:53 +0000) is newer than `src/latent_anything/_capture_binding.py` (2026-09-21 04:28:35 +0000), `tests/test_capture_binding.py` (2026-09-21 04:30:18 +0000), `scripts/sprint80_task80_7_smoke.py` (2026-09-21 04:31:03 +0000), and the artifact summary (2026-09-21 04:31:56 +0000). Symbol indexing: `graphify query "_capture_binding bind_selection"` resolves `bind_selection()`, `resolve_capture()`, `BoundCapture`, `CaptureBindingError`, `CapturedActivation`, `CaptureSelection`, `LatentValue`, `Trajectory`, and `LatentSpace` with the new module and test file indexed.
