# Task 80.4 — Implement the Report Validator

## Status

**Complete.** Sprint 80 tasks 80.1–80.4 are marked `[x]`; tasks 80.5–80.29 remain pending. Historical Sprint 79 evidence was not changed.

## Summary of Work

Added `src/latent_anything/_diagnostic_validator.py`, an independent validator boundary that consumes externally supplied machine-readable reports plus externally supplied evidence, control outcomes, and artifacts. It first reuses the frozen construction checks without changing their semantics (`validate_report_shape`, `validate_manifest`, `validate_taxonomy`/`evaluate_claim`), then enforces the 80.4 boundary:

- taxonomy references resolve (manifest metric families are known families; report metric/control references are declared in the manifest);
- evidence is complete (missing taxonomy evidence maps, unknown evidence identifiers, incomplete required evidence, and unresolvable evidence/artifact references reject);
- required controls are referenced by the report, every referenced control is declared, required controls have recorded passed/failed outcomes, and failed controls block supported statistical, intervention, observation, explanation, and causal conclusions;
- artifacts verify (missing payloads, missing digests for artifact-only references, malformed digests, and digest mismatches reject);
- provenance agrees (every capture repeats the manifest model revision, dataset split identity, and representation identity);
- conclusion strength matches evidence (supported explanations/causal results require complete family evidence, applicable capture axes, measurement support, passed controls, no admitted missing evidence, and intervention support plus an applicable causal expectation for causal results).

Explicit unsupported conclusions, inconclusive results, declared non-applicable families, and falsified causal results are preserved without promotion. No 80.5 manifest, report rendering, orchestration, or detector logic was added.

## Files Modified

- `src/latent_anything/_diagnostic_validator.py` — independent validator boundary described above.
- `tests/test_diagnostic_validator.py` — focused consumer-observable pass/fail tests for each acceptance-critical rejection.
- `scripts/sprint80_task80_4_smoke.py` — committed executable smoke scenario (valid pass plus missing-control, provenance-mismatch, and overclaim rejections).
- `docs/sprint-plans/sprint-80.md` — marked task 80.4 complete only.

## Validation Evidence

Focused validation only (no formatter, linter, type checker, packaging, documentation build, or project-wide suite was run):

```text
uv run pytest tests/test_diagnostic_validator.py -q
15 passed in 3.03s

uv run pytest tests/test_representation_taxonomy.py tests/test_benchmark_manifest.py tests/test_diagnostic_report.py -q
21 passed in 3.29s

uv run python scripts/sprint80_task80_4_smoke.py
PASS validator-clean
REJECT missing-controls: required control control-extra is missing
REJECT provenance-mismatch: provenance mismatch for captures[0].model_revision
REJECT overclaim: claim explanation-1 is stronger than its evidence supports
exit code 0
```

The committed smoke script validates an externally supplied valid fixture, then independently rejects an unreferenced required control, a changed capture model revision, and an explanation whose only support is an observation reference. Digest-mismatch and missing-payload artifact rejections are covered by focused tests. Re-run evidence for this correction round:

```text
uv run pytest tests/test_diagnostic_validator.py tests/test_representation_taxonomy.py tests/test_benchmark_manifest.py tests/test_diagnostic_report.py -q
36 passed in 3.15s
```

## Affected Claims

- An externally supplied bounded report with complete taxonomy evidence, passed controls, verified artifacts, matching provenance, and measurement-backed conclusions passes.
- Missing controls, failed required controls, mismatched provenance, artifact digest failures, incomplete evidence, unknown taxonomy references, and overclaimed conclusions are rejected independently of report construction.
- Non-applicable, unsupported, inconclusive, and falsified results remain explicit and non-promotable.

## Negative Results and Limitations

- The validator checks declared evidence/control/artifact/provenance semantics, not the scientific correctness of detector metric values or estimator quality.
- Artifact verification is SHA-256 over caller-supplied payloads; it does not fetch, decode, or re-execute artifacts.
- Statistical repetition counts and control randomization quality beyond a recorded passed/failed outcome are not re-audited.
- No 80.5 manifest or later-sprint workflow evidence was created; evidence-review readiness is limited to the focused validator tests, predecessor coherence checks, and committed smoke script above.

## Evidence-Review Readiness

Ready: the focused validator suite covers the valid path plus every named rejection; predecessor contract tests remain green; the committed smoke script demonstrates one clean case and missing-controls, provenance-mismatch, and overclaim rejections.

## Graph Refresh Record

Final task-level step, run after all corrections and bookkeeping:
```text
graphify update .
Re-extracting code files in . (no LLM needed)...
  AST extraction: 100/250 uncached files (40%) [8 workers]
  AST extraction: 200/250 uncached files (80%) [8 workers]
  AST extraction: 250/250 uncached files (100%) [8 workers]
  warning: 246 source file(s) produced zero nodes and are absent from the graph: hooks.json, evals.json, evals.json, act_policy_representation_benchmark.json, anisotropy_benchmark.json (+241 more). A re-run will retry them (empties are no longer cached); if it persists, please report the file(s) (#1666).
[graphify watch] community set changed since labeling (1047 saved labels, 1003 communities now; renamed 158 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) -> 2026-09-21/
Graph has 13806 nodes (above 5000 limit). Building aggregated community view...
graph.html written (aggregated: 1003 community nodes, 1191 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt: 13806 nodes, 28569 edges, 1003 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
exit code 0
```

Index/freshness evidence: `graphify-out/graph.json` (2026-09-21 10:17:46 +0700) is newer than `src/latent_anything/_diagnostic_validator.py` (2026-09-21 10:05:37 +0700) and `scripts/sprint80_task80_4_smoke.py` (2026-09-21 10:14:19 +0700); `grep -c _diagnostic_validator graphify-out/graph.json` returns 434 and `validate_diagnostic_report` resolves in the updated graph.
