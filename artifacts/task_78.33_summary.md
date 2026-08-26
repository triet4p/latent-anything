# Sprint 78.33 — Confirmed audit remediation

## Scope and result

**PASS.** This atomic task remediates only the
three confirmed 78.32 seams:

1. `latent_anything.cli._inspect_dataset` now calls the established lazy
   `load_lerobot()` and `require_optional("lerobot.datasets", extra="lerobot")`
   boundaries before resolving `LeRobotDatasetMetadata`. The provider metadata
   constructor and command dispatch are unchanged. A fresh blocked-backend
   subprocess now receives exactly:
   `Optional backend 'lerobot' is unavailable. Install with: uv sync --extra lerobot`.
   Nested dependency failures remain unwrapped by the existing optional seam.
2. `MLPProbe.predict` remains deliberately unsupported. Its public docstring
   now states the fitted precondition, the absent model-state serialization,
   and the exact `RuntimeError`/`NotImplementedError` behavior. No prediction
   implementation or result behavior changed.
3. `compare_probes.linear_config` is narrowed from `Any` to
   `LinearProbeConfig | None`, with the same `None` default and identical
   runtime behavior. Explicit and implicit default configurations produce
   equal `ProbeComparison` results under the existing deterministic fixture.

## Compatibility and decision evidence

- Focused CLI/optional/MLP/probe suite: **89 passed, 5 skipped**.
- Import and API behavior: CLI import remains free of eager LeRobot modules;
  canonical and legacy command paths are untouched; missing-backend error is
  covered in a subprocess with an import blocker.
- `MLPProbe.predict` still raises the existing `NotImplementedError` after fit;
  the test now asserts the contract language as well as the negative signal.
- `compare_probes` introspection reports
  `linear_config: LinearProbeConfig | None = None`; `typing.get_type_hints`
  resolves the same union and explicit/default result parity passes.
- Dynamic-boundary decision appended to
  `.agents/memory/decisions.md` on 2026-08-26: retain `Any` only at genuinely
  heterogeneous registry-factory, metadata/provenance, optional-backend, and
  constructor-kwargs seams; narrow concrete known configs; do not invent
  speculative `TypedDict`/`Protocol` abstractions; exclude literal-text scan
  false positives from the typed count.

## Regenerated inventories

- API-freeze snapshot regenerated with
  `uv run python scripts/api_freeze_snapshot.py --write`.
- Snapshot SHA-256:
  `48d64721b73a9d0c9e73da4a41940008c70dfa7841e500bc11bc8dcd22ddf7f6`.
- 78.32 findings ledger regenerated in place:
  `artifacts/task_78.32_findings.json`.
- Ledger counts are now **182** missing public docstring entries and **41**
  `Any` token hits (**40** typed annotations plus one separately tagged
  `signature-text false positive`). Classification counts: metadata/provenance
  justified **26**, optional-backend justified **4**, requires owner decision
  **10**, signature-text false positive **1**.
- Ledger SHA-256:
  `8cbdde79cf275f35afa90f5a7acefedbde97caf00d804aa31b3d2ae2385be65d`.
- Integrity validation: `ledger_integrity 182 41 40`; exact source reconciliation
  is `missing_docstring_scan_exact 182 duplicates 0 missing 0 extra 0`.

## Gates

- Focused tests: **PASS — 95 passed, 5 skipped** (including API snapshot).
- Ruff: **PASS** (`uv run ruff check src tests`); format: **PASS** (255 files);
  strict Pyright: **PASS** (0 errors); API snapshot check: **PASS**;
  `git diff --check`: **PASS** with the repository's known LF/CRLF warnings.
- MkDocs strict/link gate: **PASS** using the locked docs+viz environment and
  a cleaned temporary site directory.
- CI-equivalent `uv sync --locked --extra viz`: **PASS**.
- Full pytest: **PASS — 1563 passed, 36 skipped, 39 warnings**. A preceding
  run had one unrelated flaky SQLite WAL initialization failure
  (`tests/test_disk_cache.py::test_concurrent_process_writers_preserve_sqlite_integrity`);
  its isolated rerun and the authoritative second full run passed.
- No model, network, CUDA, commit, push, tag, or release operation performed.

## Graph and plan

Sprint 78 plan task 78.33 is marked complete after the final gates above passed.
Graphify was updated and clustered after the final artifact/plan changes:
**10,866 nodes / 20,969 edges / 940 communities**. The known zero-node JSON
sidecar warnings remain non-source graph extraction warnings.
