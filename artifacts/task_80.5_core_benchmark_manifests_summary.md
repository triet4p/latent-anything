# Task 80.5 — Predeclare the Two Core Benchmark Manifests

## Status

**Complete.** Sprint 80 tasks 80.1–80.5 are marked `[x]`; tasks 80.6–80.29 remain pending. Historical Sprint 79 evidence was not changed.

## Summary of Work

Added exactly two immutable, machine-readable benchmark manifests conforming to the frozen 80.2 schema, reusing stable taxonomy identifiers and existing pinned revisions. No benchmark was run and no evidence or results were generated.

- Encoder/autoencoder: `artifacts/benchmark_manifest_sprint80_encoder_autoencoder_v1.json` (`sprint80-core-encoder-autoencoder-collapse-v1`). ConvVAE bottleneck collapse/rank-loss probe on sklearn digits heldout data. Pins ConvVAE configuration `conv-vae-8x8-latent4-seed0-epochs5`, sklearn digits `scikit-learn==1.9.0` with `default_rng(42)` 1437/360 train/heldout split, bottleneck-mu representation identity, healthy-counterexample plus benign-low-variance negative plus null-shuffle controls, independent seeds, bootstrap uncertainty, two predeclared thresholds, expected feature-axis localization, and an applicable causal expectation with explicit success/falsification rules.
- Transformer hidden-state: `artifacts/benchmark_manifest_sprint80_transformer_hidden_state_v1.json` (`sprint80-core-transformer-hidden-state-probe-v1`). Leakage-safe separability/probe probe on pinned `openai-community/gpt2` revision `e7da7f221d5bf496a48136c0cd264e630fe9fcc8` with WikiText-2 config `wikitext-2-raw-v1` at revision `f776294184f13b8ff2337b3841cf9269a6216d1e`. Uses the existing validation selection (3760 official rows, 2048 selected, max 128 tokens/row, blank rows dropped with hashes retained), native `output_hidden_states` capture, layer/token/slice axes, capacity plus label-randomization plus non-separable-negative plus split-swap controls, independent seeds, bootstrap uncertainty, two predeclared thresholds, expected earliest-passing-layer localization, and an applicable activation-patching causal expectation with explicit falsification rules.

Both manifests are `status=predeclared`, `commitment.locked=true`, and carry canonical SHA-256 commitment digests. Encoder digest `b4c49c890cf34cc0530d499c2b82a09057f1f1ebf5b9d9a3843200760b137dd3`; transformer digest `c4773238ddf4d8854cb40ad26811b7b204cab8210ff9fe45f79e83e12da4675a`. They declare acceptance semantics only and encode no observed metric values or outcomes. Added `tests/test_sprint80_core_manifests.py` and committed smoke script `scripts/sprint80_task80_5_smoke.py`.

## Model/Dataset/Revision Choices and Rationale

- Encoder case reuses the existing CPU ConvVAE 8x8 lane and sklearn digits heldout split already proven in `artifacts/conv_vae_heldout_benchmark.json` (1437 train / 360 heldout via `default_rng(42)`), keeping the smallest defensible ordinary-DL autoencoder case inside the permanent 16 GiB ceiling.
- Transformer case reuses the pinned GPT-2 identity from `src/latent_anything/integrations/transformer_lm.py` (`openai-community/gpt2` @ `e7da7f221d5bf496a48136c0cd264e630fe9fcc8`, 12 layers, hidden dim 768) and the pinned WikiText-2 identity from `scripts/m14_l04_wikitext_manifest.py` (`Salesforce/wikitext`, `wikitext-2-raw-v1` @ `f776294184f13b8ff2337b3841cf9269a6216d1e`, 8192/2048 selection, 128 tokens/row), restricted here to the validation selection for a leakage-safe heldout probe lane.
- Collapse/rank-loss covers the encoder case; separability/probe-leakage covers the transformer case. Both causal expectations are applicable with explicit falsification rules so later 80.23/80.24 proofs are binary.

## Validation Evidence

Focused validation only (no formatter, linter, type checker, packaging, documentation build, or project-wide suite was run):

```text
uv run pytest tests/test_sprint80_core_manifests.py -q
5 passed in 3.01s

uv run pytest tests/test_sprint80_core_manifests.py tests/test_benchmark_manifest.py -q
13 passed in 2.99s

uv run python scripts/sprint80_task80_5_smoke.py
PASS encoder sprint80-core-encoder-autoencoder-collapse-v1
PASS transformer sprint80-core-transformer-hidden-state-probe-v1
REJECT post-hoc-mutation: manifest digest does not match canonical content
exit code 0

uv run python -m json.tool artifacts/benchmark_manifest_sprint80_encoder_autoencoder_v1.json > NUL
exit code 0
uv run python -m json.tool artifacts/benchmark_manifest_sprint80_transformer_hidden_state_v1.json > NUL
exit code 0
```

## Affected Claims

- Both core benchmark inputs are validator-clean immutable predeclared manifests with pinned revisions/splits, defect/counterexample declarations, thresholds, controls, expected localization, and causal success/falsification rules.
- Later 80.23/80.24 proofs have binary, leakage-safe acceptance inputs, particularly for the transformer probe lane.
- No workflow, detector, evidence, or result claim is made by this task.

## Negative Results and Limitations

- Placeholder SHA-256 digests for model/data artifacts are derived from pinned identity strings, not from fetched model weights or dataset bytes; they bind the declared identities but do not prove byte-level asset availability.
- Threshold values are predeclared judgment choices, not calibrated from Sprint 80 workflow evidence; 80.23/80.24 must execute them as written or fail explicitly.
- The transformer lane declares inference-only capture within 16 GiB but was not executed here; runtime feasibility remains a proof-task concern.
- No observed results are encoded; any future result must arrive through the workflow and 80.4 validator, never by editing these manifests.

## Graph Refresh Record

Final task-level step, run after implementation, tests/smoke, artifact, and status:

```text
graphify update .
Re-extracting code files in . (no LLM needed)...
  AST extraction: 100/256 uncached files (39%) [8 workers]
  AST extraction: 200/256 uncached files (78%) [8 workers]
  AST extraction: 256/256 uncached files (100%) [8 workers]
  warning: 248 source file(s) produced zero nodes and are absent from the graph: hooks.json, evals.json, evals.json, act_policy_representation_benchmark.json, anisotropy_benchmark.json (+243 more). A re-run will retry them (empties are no longer cached); if it persists, please report the file(s) (#1666).
[graphify watch] community set changed since labeling (1003 saved labels, 1043 communities now; renamed 177 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) -> 2026-09-21/
Graph has 13829 nodes (above 5000 limit). Building aggregated community view...
graph.html written (aggregated: 1043 community nodes, 1226 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt: 13829 nodes, 28602 edges, 1043 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
exit code 0
```

Index/freshness evidence: `graphify-out/graph.json` (2026-09-21 10:24:08 +0700) is newer than both manifests (2026-09-21 10:22:25 +0700). Graphify indexes code AST nodes, so JSON manifest identifiers return 0 hits by design (`grep -c` for each manifest id: 0); the code-side manifest test module is indexed with 77 hits for `test_sprint80_core_manifests`. The authoritative manifest binding proof is the validator-clean focused suite and smoke scenario above, not graph text search.

## Evidence-Review Readiness

Ready: focused manifest tests prove both files validate, carry distinct core identities and taxonomy bindings, declare leakage-safe transformer controls, and reject post-hoc mutation and incomplete thresholds; the committed smoke script reproduces the clean-plus-tamper-rejection scenario.
