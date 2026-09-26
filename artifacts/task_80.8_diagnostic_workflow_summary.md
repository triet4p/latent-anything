# Task 80.8 — Implement the Diagnostic Workflow State Machine

## Status

**Complete.** Sprint 80 tasks 80.1–80.8 are marked `[x]`; tasks 80.9–80.29 remain pending. Historical Sprint 79 evidence was not changed.

## Summary of Work

Added `src/latent_anything/_diagnostic_workflow.py`, the smallest architecture-neutral coordinator for `capture -> detect -> localize -> explain -> intervene -> compare -> report`. It reuses the existing coordinator-adjacent seams without duplicating a pipeline/config path: declaration stays on the reviewed 80.6 `DiagnosticRequest`/`DiagnosticResult` surface, manifest validity and canonical digests reuse `validate_manifest`/`manifest_digest`, and canonical JSON identity reuses the shared `canonical_json` contract.

Explicit state transitions and contracts:

- Exact order: `WORKFLOW_STAGES = (capture, detect, localize, explain, intervene, compare, report)`, matching the frozen `diagnostics` order. `DiagnosticWorkflow` requires executor and version coverage of exactly these seven stages; anything else fails closed at construction.
- Typed inputs: `StageInvocation(stage, request, manifest, prior, workflow_identity, request_digest, manifest_digest, config_digest)` with `prior` verified as the exact ordered prefix the stage requires. Executor type: `StageExecutor(invocation) -> StageOutput`.
- Typed outputs: `StageOutput(stage, outcome, payload, artifact_refs)` where `outcome` is `completed | not_applicable | unsupported`. Payloads and refs are frozen; payloads must be canonical JSON; refs must be non-empty unique strings. A wrong-stage return or non-`StageOutput` return becomes an honest `failed` run at that stage, not a silent skip.
- Failure states: executor exceptions (including `StageContractError`) are captured as `FailureInfo(stage, error_type, message)`; the run result becomes `status="failed"` with completed stages preserved, artifact refs concatenated only from completed stages, and `stage_results.failure` carrying the failure. A `failed` checkpoint's `next_stage` is the failed stage.
- Resumable artifact boundaries: `WorkflowCheckpoint(request_id, manifest_id, request_digest, manifest_digest, config_digest, workflow_identity, completed_stages, outputs, output_digests, artifact_refs, next_stage, status, failure)` where `completed_stages` must be an exact ordered prefix, per-stage digests bind `(workflow, stage, outcome, payload, refs)`, artifact refs must concatenate stage outputs in order, and `running`/`failed`/`completed` states constrain `next_stage`/`failure` explicitly. `run(..., stop_after=...)` stops at a boundary with `status="running"`; `resume(checkpoint, ...)` verifies request/manifest/config/upstream-output identity and continues without rerunning completed stages. `restart=True` explicitly reruns from the first stage; without it, `completed`/`failed` checkpoints reject.
- Lifecycle mapping: `completed` runs yield `DiagnosticResult(status="completed", completed_stages=all 7, artifact_refs, report_id from the report payload when completed, stage_results={stages, digests, identities, failure: None})`; `running` runs yield `status="running"` with the next-stage message; `failed` runs yield `status="failed"` with the failure message. `report_id` is set only from a `completed` report payload; `unsupported`/`not_applicable` outcomes are recorded per stage and never promoted.
- Coordinator stays method-agnostic: no detector, explainer, or intervention algorithm and no model-family branch exists in this module. `__call__` keeps the coordinator usable as a supplied callable.

## Validation Evidence

Focused validation only (no formatter, linter, type checker, packaging, docs build, or project-wide suite was run):

```text
uv run pytest tests/test_diagnostic_workflow.py -q
10 passed in 3.01s

uv run pytest tests/test_diagnostic_workflow.py tests/test_diagnostics_request.py tests/test_sprint80_core_manifests.py tests/test_capture_binding.py -q
36 passed in 3.08s

uv run python scripts/sprint80_task80_8_smoke.py
PASS run completed(7) identity=131691d238e0 report=smoke-report-80-8
PASS resume stop(detect)+resume == uninterrupted run
PASS fail detect boundary kept(capture) error=RuntimeError
REJECT invalid-resume: checkpoint with status 'failed' requires restart=True
exit code 0

uv run python -c "import latent_anything as la; ..."
9 ['CaptureSelection', 'ComparisonRequest', 'ControlSelection', 'DiagnosticRequest', 'DiagnosticRequestError', 'DiagnosticResult', 'DiagnosticSelection', 'InterventionRequest', 'OutputSelection']
False False False
```

The final line confirms the exact 80.6 nine-name top-level surface is intact, `ResultStatus` stays diagnostics-only, and neither `DiagnosticWorkflow` nor `StageOutput` widened the public surface.

## Affected Claims

- All seven stages have explicit typed inputs/outputs, failure states, and resumable artifact boundaries through supplied executors only.
- Valid resume (`stop_after="detect"` then resume) is exactly equivalent to an uninterrupted run (equal result and checkpoint, including round-tripped checkpoint).
- Invalid resume (mismatched request/manifest/config identity, out-of-order or skipped stages, digest mismatch, resuming a failed/completed checkpoint without `restart=True`, `stop_after` preceding the resumed position) fails closed.
- No 80.9+ detector, execution, persistence (80.21), or rendering (80.22) claim is made by this task.

## Negative Results and Limitations

- Stage executors in tests/smoke are lightweight deterministic doubles supplied at the boundary; they prove coordinator contracts (order, identity, resume equivalence, failure honesty, rejection), not diagnostic-method correctness.
- Checkpoints are in-memory mappings with deterministic digests; content-addressed persistence remains 80.21 and is not implemented here.
- `restart=True` reruns from the first stage by explicit contract; there is no partial re-execution or stage-skipping mode — skipped/out-of-order checkpoints reject.
- Manifest threshold semantics are untouched; this task validates manifest identity/shape via the frozen validator only.
- No formatter, linter, type checker, packaging, docs build, or project-wide suites were run per task constraints.

## Files Modified

- `src/latent_anything/_diagnostic_workflow.py` — new private coordinator (`WORKFLOW_STAGES`, `StageOutput`, `StageInvocation`, `StageExecutor`, `FailureInfo`, `WorkflowCheckpoint`, `DiagnosticWorkflow`, `WorkflowError`, `StageContractError`); zero top-level export changes.
- `tests/test_diagnostic_workflow.py` — 10 consumer-observable tests: full seven-stage flow/contracts, stop/resume equivalence, handler independence, failure boundary, unsupported-not-promoted, resume/config/order rejection, wrong-stage/bad-executor failure, contract-shape rejection, deterministic request-bound identity.
- `scripts/sprint80_task80_8_smoke.py` — committed direct smoke exercising run, fail, and resume.
- `docs/sprint-plans/sprint-80.md` — marked task 80.8 complete only.

## Evidence-Review Readiness

Ready: focused workflow tests prove exact order, explicit contracts, stop/resume equivalence, handler independence, honest failure boundaries, and fail-closed resume/config/order rejection with deterministic identity; the committed smoke reproduces run, fail, and resume; the nine-name 80.6 surface is verified unchanged.

## Graph Refresh Record

Final task-level step, run after implementation, focused validation, artifact, status, and decision log:

```text
graphify update .
Re-extracting code files in . (no LLM needed)...
  AST extraction: 100/257 uncached files (38%) [8 workers]
  AST extraction: 200/257 uncached files (77%) [8 workers]
  AST extraction: 257/257 uncached files (100%) [8 workers]
  warning: 248 source file(s) produced zero nodes and are absent from the graph: hooks.json, evals.json, evals.json, act_policy_representation_benchmark.json, anisotropy_benchmark.json (+243 more). A re-run will retry them (empties are no longer cached); if it persists, please report the file(s) (#1666).
[graphify watch] community set changed since labeling (1058 saved labels, 1034 communities now; renamed 188 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) -> 2026-09-21/
Graph has 14080 nodes (above 5000 limit). Building aggregated community view...
graph.html written (aggregated: 1034 community nodes, 1309 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt: 14080 nodes, 29362 edges, 1034 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
exit code 0
```

Index/freshness evidence: `graphify-out/graph.json` (2026-09-21 05:12:29 +0000) is newer than `src/latent_anything/_diagnostic_workflow.py` (2026-09-21 05:10:30 +0000), `tests/test_diagnostic_workflow.py` (2026-09-21 05:08:20 +0000), `scripts/sprint80_task80_8_smoke.py` (2026-09-21 05:10:52 +0000), and the artifact summary (2026-09-21 05:11:20 +0000). Symbol indexing: `graphify query "DiagnosticWorkflow StageOutput WorkflowCheckpoint resume"` resolves `DiagnosticWorkflow`, `StageOutput`, `WorkflowCheckpoint`, `.run()`, `.resume()`, `._execute()`, `._failed()`, `WorkflowError`, `DiagnosticRequest`, `DiagnosticResult`, and the new test module.
