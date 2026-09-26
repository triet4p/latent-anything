# Task Summary: 80.22 — Render the concise AI-engineer report

**Sprint:** Sprint 80 — Stable Depth
**Task:** 80.22 (completes Phase E)

## Summary of Work

Added `src/latent_anything/_report_renderer.py`: one
architecture-neutral private renderer that turns a successfully loaded,
independently validated80.21 diagnostic artifact into concise,
byte-for-byte deterministic report bytes driven only by validated
structured fields — no heuristics, no LLM prose, no model workflows.
`render_diagnostic_report(root, run_id, manifest)` re-runs the full80.21
loader (schema, manifest/taxonomy/report-schema hashes, stage chain,
blob re-hashing and path containment, validator result), re-verifies the
registered report digest through the run-record canonical codec, audits
every evidence reference against registered rows and blobs, cross-checks
registered rows against the stage records they were registered from
(intervention status == recorded conclusion, record digests equal the
blob hashes, comparison status matches its recorded classification,
supported causal claims must reference a supported intervention row and
have no failed required controls), and emits the fixed section order:
Identifiers (artifact/run/report/manifest/taxonomy/report-schema/workflow
identities plus `validator: passed`), Symptom (taxonomy family plus
bounded conclusion from the detect-stage record, escaped description,
metrics, status, hypothesis), Location (capture axes and representation
plus supported localization selections, or an explicit
"location: not applicable" line), Evidence (metric estimates with
uncertainty intervals, every control outcome, sorted family evidence,
explanation outcomes, and a complete `evidence <name>: <sha256>` index of
every content-addressed blob), Causal (one line per registered
intervention with conclusion and record link, claim lines with truthful
`claim allowed` flags, an explicit blocked line whenever a required
manifest control failed, or an explicit unsupported line when no
intervention records exist), Comparison (both sides' run/checkpoint
identities, representation-vs-task signed/absolute deltas with declared
tolerances and change flags, classification, record link, or an explicit
not-applicable line), Limitations (declared limitations plus every
negative result — falsified/inconclusive/unsupported interventions and
comparisons, never silently omitted), and Next action (escaped action and
rationale with every required evidence reference verified as registered).
All variable strings are escaped (backslash, newline, CR, tab, C0/DEL
controls) so hostile labels cannot create sections or raw control bytes;
the output contains no absolute paths, secrets, or volatile timestamps
and is deterministic given the persisted bytes. Unknown statuses,
unregistered references, digest/stage/report contradictions, failed
controls presented as success, and non-passing validator results raise
`ReportRenderingError`/propagate `DiagnosticArtifactError` — nothing is
silently omitted or upgraded.
`persist_rendered_report` renders first, stores the bytes
content-addressed on the same run via `FileSystemRunRecorder.add_artifact`
(conflict-checked, idempotent, `text/markdown`), and materializes the
declared `OutputSelection` copy atomically at
`output_location/artifact_name` (simple-name and `include_report` checks
fail closed) without mutating the validated source artifact. Public and
frozen contracts unchanged (no new `latent_anything` exports; the80.3
schema,80.4 validator, and80.21 loader are consumed read-only). Phase E
of Sprint 80 is now fully checked off.

## Files Modified

* [src/latent_anything/_report_renderer.py](src/latent_anything/_report_renderer.py) - New deterministic renderer: full revalidation-first rendering, contradiction/reference audits, escaping, fixed sections, content-addressed persistence with declared OutputSelection copy.
* [tests/test_report_renderer.py](tests/test_report_renderer.py) -11 behavioral tests: exact section order and identifiers, all four outcomes without promotion, failed-control never success, symptom/location/evidence/next-action content, evidence links re-hash, hostile-label structure safety, contradictory comparison refusal, tampered-artifact/manifest refusal, deterministic replay plus idempotent persistence, output-selection/hygiene fail-closed, unchanged public surface.
* [tests/test_diagnostic_persistence.py](tests/test_diagnostic_persistence.py) - Additive fixtures only: shared `_ablation_spec` helper, `report_id` parameter on the fixture report, declared next-action refs (`e-1`, `o-1`), plus two new chain builders (`completed_run_full_outcomes`, `failed_control_run`) and a branch-order fix so the blocked trial's identity control actually fails; all existing tests unchanged and green.
* [scripts/sprint80_task80_22_smoke.py](scripts/sprint80_task80_22_smoke.py) - Committed smoke reusing the real persistence-test chains through a registered importlib load; renders, asserts every section/outcome/link, proves byte-identical replay and content-addressed idempotent persistence, and exercises escaping, failed-control blocking, contradiction/tamper/manifest/output-selection refusals inside self-cleaning temp roots.
* [docs/sprint-plans/sprint-80.md](docs/sprint-plans/sprint-80.md) - Marked 80.22 `[x]` (Phase E complete).

## Testing

* **Test Files:** [tests/test_report_renderer.py](tests/test_report_renderer.py) plus persistence/validator/report/workflow/run-record regressions
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_report_renderer.py tests/test_diagnostic_persistence.py tests/test_run_record.py tests/test_diagnostic_validator.py tests/test_diagnostic_report.py tests/test_diagnostic_workflow.py -q`
* **Result:** `71 passed in 34.83s`
* **Smoke (80.22):** `uv run python scripts/sprint80_task80_22_smoke.py` → exit 0. Exact output:

```
PASS sections: exact fixed order with report title
PASS identifiers: artifact/run/schema/manifest/workflow identities recorded
PASS symptom/location/evidence: bounded conclusion, location, uncertainty, controls
PASS outcomes: supported/falsified/inconclusive/unsupported rendered without promotion
PASS limitations/next action: negatives explicit, action validated
PASS evidence links:14 content-addressed links re-hash
PASS determinism/persistence: byte-identical render, content-addressed, idempotent
PASS escaping: hostile labels cannot alter report structure
PASS failed control: rendered as blocked, never as success
PASS fail-closed: contradictory comparison status refused rendering
PASS fail-closed: tampered artifact blob refused rendering
PASS fail-closed: manifest hash disagreement refused rendering
PASS fail-closed: output selection without include_report refused rendering
PASS hygiene: no absolute paths, secrets, or volatile timestamps
public surface: 9/9 80.6 names intact, no renderer symbols leaked
PASS frozen manifest: read-only, unchanged by the smoke run
```

* **Regression smoke (80.21):** `uv run python scripts/sprint80_task80_21_smoke.py` → exit 0, all sixteen PASS lines unchanged.
* **Graph:** `graphify update .` after graph-visible writes → `AST extraction:255/255 uncached files (100%)`, `Rebuilt:15427 nodes, 34695 edges, 1078 communities`, `graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out`. A final confirming `graphify update .` ran after this artifact was written; its output is reported in the task handoff.
* **Scope note:** no formatters, linters, or project-wide suites were run, per task constraints.

## Additional Notes

* Reuse: the80.21 loader/`registered_artifact_bytes` as the sole input
  path (invalid artifacts cannot reach the renderer), the run-record
  canonical codec for report-digest verification, `OutputSelection` from
  the frozen80.6 request API for the declared output copy, the80.3
  report-schema row/status vocabulary, the80.4 validator result persisted
  at assembly time, and content-addressed `FileSystemRunRecorder`
  artifact storage for the rendered bytes (`<artifact_name>.rendered`,
  distinct from the registered JSON report blob).
* The renderer treats stage records as the audit source of truth: every
  causal/comparison sentence carries the registered record name plus its
  SHA-256, and the smoke re-hashes all fourteen links from disk.
* Fixture reuse: the smoke loads `tests/test_diagnostic_persistence.py`
  through `importlib.util.spec_from_file_location` with `sys.modules`
  pre-registration (the documented dataclass-safe pattern); the fixture
  module defines no dataclasses and no pytest fixtures at import time.
* Renderer-side contradiction checks are defense-in-depth beyond the
  validator: the80.22 test proves a persisted report whose comparison
  status disagrees with its recorded classification is refused only at
  render time.
* Phase F (core proofs,80.23–80.26) and Phase G remain; Sprint 81 owns
  publication.

## Phase E deep review corrections (F1)

Status: F1 closed with passing focused evidence; the80.22 checkbox
remains `[x]` (the review conditioned it on F1 being closed, which it
now is).

* **F1 — record-content cross-check (renderer half):** the causal
  section previously compared only the row's self-declared digest with
  the inventory and the row's status with its own outcome; it now opens
  and parses the cited record blob before rendering any causal prose and
  requires: exactly one registered record reference; a registered
  `intervention-<id>-record` row id; the record's `intervention_id`
  equal to the stripped row id; the record's `conclusion` to be one of
  the four frozen outcomes and equal to the row status; the record's
  `target` and `kind` equal to the row's target/intervention; the
  record's `metric_id` present and (when available) backed by detect
  metric evidence; and the SHA-256 of the cited record bytes equal to
  the row's `record_digest` and the inventory digest. The comparison
  loop was strengthened to also require the record's `comparison_id`
  equal to the row id. Module and function docstrings now state exactly
  what is opened and cross-checked.
* **New negative tests (tests/test_report_renderer.py):**
  `test_supported_row_vs_falsified_record_refuses_rendering` (tampered
  record conclusion falsified under a supported row → refusal naming the
  record conclusion, asserted again through `persist_rendered_report`
  proving refusal before any output materialization), plus the persisted
  `_tamper` helper that rewrites blob/inventory/report/validator/document
  digests and run metadata consistently so only the new cross-checks can
  refuse; `test_wrong_intervention_id_in_record_refuses_rendering`
  (renamed row id → `record names intervention ...`); and
  `test_unrelated_registered_blob_refuses_rendering` (row repointed at
  the registered `stage-record-detect` blob → refused by the identity
  cross-check). The persistence half of F1 (unbacked/duplicate/
  mismatched rows rejected before write) is covered in
  `tests/test_diagnostic_persistence.py` and recorded in the80.21
  summary.

**Correction validation (exact):**
`uv run pytest tests/test_report_renderer.py tests/test_diagnostic_persistence.py tests/test_run_record.py tests/test_diagnostic_validator.py tests/test_diagnostic_report.py tests/test_diagnostic_workflow.py -q` → **78 passed in28.39s**.
`uv run python scripts/sprint80_task80_21_smoke.py` → exit0, all sixteen PASS lines unchanged.
`uv run python scripts/sprint80_task80_22_smoke.py` → exit0, all sixteen PASS lines unchanged.
Graph rebuild after correction writes: `graphify update .` → `Rebuilt:15445 nodes, 34753 edges, 1058 communities`, graph files updated; a final confirming update ran after these artifact edits (reported in the handoff).
