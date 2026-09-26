# Task Summary: 80.21 — Persist the content-addressed diagnostic artifact

**Sprint:** Sprint 80 — Stable Depth
**Task:** 80.21

## Summary of Work

Added `src/latent_anything/_diagnostic_artifact.py`: one private
assembly/persistence seam that turns a completed diagnostic workflow
(request, frozen manifest, `DiagnosticResult`, `WorkflowCheckpoint`) plus
the structured report payload into a self-contained content-addressed
artifact (`persist_diagnostic_artifact`), reloads and revalidates it
(`load_diagnostic_artifact`), and exposes the persisted bytes for
independent checking (`registered_artifact_bytes`). Assembly rejects
incomplete/nonterminal workflows and stage-chain/result/payload
disagreements before any write; registers evidence where the frozen80.3
report schema expects it without fabricating evidence or rendering prose
(capture provenance rebuilt from the real bound capture record with
provisional artifact references remapped, interventions-section rows
registered from real stage trials, `comparison-<id>-record` evidence
registered from real compare-stage records, `explanation-<id>-record`
evidence registered from real hypothesis rows, provisional claim control
references replaced by the executed manifest controls with recorded
outcomes); derives validator inputs from real detect-stage evidence
(family evidence, passed/failed control outcomes, applicability); runs
the80.4 independent validator and persists its `passed` result bound to
the report and input digests (a failing report persists nothing).
Persistence reuses `ArtifactRef` + `FileSystemRunRecorder` atomic
temp-file/rename writes with SHA-256 file names under `artifacts/<digest>`
and `compute_run_identity`-keyed run ids (`identity[:16]`): identical
input yields an identical artifact digest/run id/bytes, re-persists
rewrite nothing, and pre-existing conflicting bytes reject before any
write. The artifact document is canonical JSON with no timestamps,
randomness, absolute paths, or machine secrets (bounded
`{numpy, python, system}` environment, manifest seeds, per-stage records
with both content digests and workflow `StageOutput.digest` values,
report digest, blob inventory, request/manifest/taxonomy/report-schema
hashes, workflow digests, result summary, validator input and result).
Loading uses only root files plus the declared external manifest:
schema, manifest/taxonomy/report-schema hashes, request round-trip,
run-metadata agreement, the full stage chain order/outcomes/digests,
report digest, every blob's existence/hash/path containment (recorder
`read_artifact` rejects escapes), inventory/digest agreement, and a full
re-run of the independent validator — any schema, provenance,
stage-chain, or validator-result disagreement fails closed. Public and
frozen contracts unchanged; coordinator untouched.

## Files Modified

* [src/latent_anything/_diagnostic_artifact.py](src/latent_anything/_diagnostic_artifact.py) - New assembly/persistence/load seam: workflow completeness and agreement checks, evidence registration for80.16–80.20 provisional refs, real validator-evidence derivation, deterministic document assembly, recorder-backed atomic content-addressed writes, full fresh-context revalidation.
* [tests/test_diagnostic_persistence.py](tests/test_diagnostic_persistence.py) -9 behavioral tests over a real seven-stage workflow: persist/load/revalidate, identical-input byte identity plus idempotent re-persist, fresh-context load, corrupt/missing blob rejection, manifest/schema/validator-result disagreements, incomplete-workflow and conflicting-blob rejection with no partial writes, persist-time validator failure writes nothing, and hidden-path/secret hygiene.
* [scripts/sprint80_task80_21_smoke.py](scripts/sprint80_task80_21_smoke.py) - Committed smoke running the real workflow chain twice (80.18/80.20 chain and the80.19 steering chain), persisting/loading/revalidating both, proving byte identity/idempotency/fresh-context load, and exercising the full fail-closed battery inside a self-cleaning temporary root.
* [docs/sprint-plans/sprint-80.md](docs/sprint-plans/sprint-80.md) - Marked 80.21 `[x]`.

## Testing

* **Test File:** [tests/test_diagnostic_persistence.py](tests/test_diagnostic_persistence.py) (plus run-record/validator/report/workflow regressions)
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_diagnostic_persistence.py tests/test_run_record.py tests/test_diagnostic_validator.py tests/test_diagnostic_report.py tests/test_diagnostic_workflow.py -q`
* **Result:** `60 passed in 24.16s`
* **Smoke (80.21):** `uv run python scripts/sprint80_task80_21_smoke.py` → exit 0. Exact output:

```
PASS persist/load: completed workflow artifact revalidates from persisted files
PASS registration:80.18/80.20 rows independently validator-clean
PASS registration:80.19 steering trial registers and validates identically
PASS determinism/idempotency: identical input -> identical digest/bytes, no rewrites
PASS fresh-context load: only persisted files plus declared manifest
PASS fail-closed: corrupted blob rejected (blob 'stage-record-detect' rejected: artifact digest mismatch ...)
PASS fail-closed: missing report blob rejected (blob 'diagnostic-report' (artifacts/e5d705...) is missing)
PASS fail-closed: manifest hash disagreement rejected (manifest hash disagrees with the persisted artifact)
PASS fail-closed: schema disagreement rejected (artifact schema must be 'diagnostic-artifact-v1')
PASS fail-closed: validator-result disagreement rejected (persisted validator result must be a passing ...)
PASS fail-closed: incomplete workflow rejected (diagnostic result must be completed)
PASS fail-closed: conflicting blob rejected (conflicting blob for digest f9fc7aab... already exists)
PASS fail-closed: report failing independent validation rejected (report failed independent validation; nothing was persisted: unknown evi...)
PASS hygiene: no hidden absolute paths, secrets, or cwd-relative state
public surface: 9/9 80.6 names intact, no persistence symbols leaked
PASS frozen manifest: read-only, unchanged by the smoke run
```

* **Regression smokes:** `sprint80_task80_18_smoke.py` → exit 0 (nine PASS lines); `sprint80_task80_19_smoke.py` → exit 0 (ten PASS lines); `sprint80_task80_20_smoke.py` → exit 0 (eight PASS lines).
* **Graph:** `graphify update .` after graph-visible writes → `AST extraction:255/255 uncached files (100%)`, `Rebuilt:15361 nodes, 34447 edges, 1066 communities`, `graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out`. A final confirming `graphify update .` ran after this artifact was written; its output is reported in the task handoff.
* **Scope note:** no formatters, linters, or project-wide suites were run, per task constraints.

## Additional Notes

* Reuse: `ArtifactRef`/`FileSystemRunRecorder`/`compute_run_identity`
  (run-record lane; the run index file keeps its frozen timestamped
  lifecycle fields but timestamps never enter identity — the test proves
  run files differ only by `created_at`/`updated_at` while identity,
  run_id, and every artifact byte are identical), the80.3 report schema
  and80.4 validator as the persistence gate, `manifest_digest` (80.2),
  taxonomy/report-schema digests from the frozen artifacts, and the
  canonical-JSON codec of the run-record lane for every blob.
* Registration rewiring rules are deterministic and documented in the
  module docstring: capture refs remapped to `capture-<capture_id>-record`,
  intervention/comparison/explanation refs bound to real stage-record
  bytes, causal-claim control refs replaced by executed manifest controls
  (the intervention's own controls remain fully recorded inside the
  registered trial-record blobs and row outcomes).
* The run record acts as the deterministic pointer: run_id =
  `identity[:16]` with the artifact digest/size/request id in metadata;
  conflicting content for the same request yields a different artifact
  digest and a different run identity rather than an overwrite.
* Limitations: loading takes an explicit `run_id` (returned by persist)
  rather than scanning; the localize payload remains a contract-shaped
  stand-in upstream; mixing single-strength and steering trials in one
  intervene stage remains out of scope (both families persist through the
  same registration path, proven by the smoke's second chain).
* Downstream: 80.22 report rendering consumes the registered report.

## Phase E deep review corrections (F1/F2/F3)

Status: corrections complete; the80.21 checkbox remains `[x]` because
every finding assigned to this task was closed with passing focused
evidence (no finding remains open).

* **F1 — intervention-row backing (persistence half):** the
  interventions registration in `_diagnostic_artifact.py` now rejects
  caller rows that are duplicated (`duplicate interventions row id`),
  unbacked by a real registered trial record (`is not backed by a
  registered trial record`), or disagreeing with the recorded trial
  (status vs conclusion, target, or intervention kind — `disagrees with
  the recorded trial conclusion` / `... disagrees with the recorded
  trial`) before any write; caller evidence references are preserved and
  augmented with the registered record blob instead of being dropped.
  New tests: `test_unbacked_duplicate_and_mismatched_intervention_rows_reject_before_write`
  (four roots — unbacked, duplicate, supported-row-over-falsified-trial,
  target mismatch — each asserting no `runs/*.json` is created).
* **F2 — preflight vs mid-write lifecycle:** `persist_diagnostic_artifact`
  now creates the run record as `running` after preflight, writes all
  blobs and the document, then transitions atomically to `completed`
  (`recorder.complete`) only if not already completed — an already
  completed identical run is never downgraded or rewritten. Module and
  function docstrings now state the exact preflight (no record, no files)
  vs mid-write (recoverable `running` record, loading refuses it, retry
  reuses the deterministic identity and completes) behavior. New test:
  `test_mid_write_failure_leaves_running_record_and_retry_completes`
  (injected `add_artifact` failure on the third call → record status
  `running`, `load` raises `must be completed`, identical retry reuses
  run_id/identity and completes, further re-persist leaves the run file
  byte-identical).
* **F3 — capture-reference collision:** `_register_evidence` now fails
  closed with `provisional capture reference ... collides with a
  registered evidence name` when a caller capture artifact reference
  equals any stage-record/evidence/report name (checked against the final
  registered-name set plus all `stage-record-<stage>` and
  `diagnostic-report` names) instead of globally retargeting it. New
  test: `test_capture_reference_collision_rejects_before_write`
  (`stage-record-compare`, `comparison-cmp-both-record`,
  `stage-record-intervene` — each rejected before any write).
* **Run-id normalization:** `load_diagnostic_artifact` validates the run
  id locally (`Path(run_id).name == run_id`, non-empty) and raises
  `DiagnosticArtifactError("invalid run id: ...")`, replacing the raw
  `ValueError` escaping from the run-record lane. New test:
  `test_invalid_run_id_is_normalized_to_diagnostic_artifact_error`
  (`../evil`, `a/b`, `""`).

**Correction validation (exact):**
`uv run pytest tests/test_report_renderer.py tests/test_diagnostic_persistence.py tests/test_run_record.py tests/test_diagnostic_validator.py tests/test_diagnostic_report.py tests/test_diagnostic_workflow.py -q` → **78 passed in28.39s** (14 renderer incl.3 new F1 negatives,13 persistence incl.4 new F1/F2/F3/run-id tests, plus run-record/validator/report/workflow regressions).
`uv run python scripts/sprint80_task80_21_smoke.py` → exit0, all sixteen PASS lines unchanged (including the exact `status`/idempotency and fail-closed battery under the new running→completed flow).
`uv run python scripts/sprint80_task80_22_smoke.py` → exit0, all sixteen PASS lines unchanged.
Graph rebuild after correction writes: `graphify update .` → `AST extraction:255/255 uncached files (100%)`, `Rebuilt:15445 nodes, 34753 edges, 1058 communities`, `graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out`; a final confirming update ran after these artifact edits (reported in the handoff).
