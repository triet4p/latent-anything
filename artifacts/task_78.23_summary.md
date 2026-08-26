# Task Summary: Sprint 78.23 — run-record SRP ownership

**Sprint:** Sprint 78
**Task:** 78.23 — Split run-record codec, persistence, and comparison seams

## Summary of Work

Refactored the run-record implementation behind a stable
`latent_anything.run_record` facade. Pure canonical JSON/deep-freezing,
identity, runtime metadata, and schema migration now live in
`_run_record_codec.py`; the frozen `ArtifactRef` and `RunRecord` value objects
live in `_run_record_schema.py`; filesystem lifecycle, atomic JSON writes,
content-addressed artifacts, portable artifacts, path safety, and recovery live
in `_run_record_persistence.py`; comparison/report assembly lives in
`_run_record_comparison.py`. The facade restores historical public module
identities and re-exports without changing schemas, bytes, errors, or external
recorder/provider contracts.

## Files Modified

* [src/latent_anything/run_record.py](../src/latent_anything/run_record.py) — stable public facade and module-identity compatibility aliases.
* [src/latent_anything/_run_record_codec.py](../src/latent_anything/_run_record_codec.py) — canonical codec, identity, runtime metadata, and migration.
* [src/latent_anything/_run_record_schema.py](../src/latent_anything/_run_record_schema.py) — private implementation of public schema value objects.
* [src/latent_anything/_run_record_persistence.py](../src/latent_anything/_run_record_persistence.py) — filesystem recorder and artifact lifecycle.
* [src/latent_anything/_run_record_comparison.py](../src/latent_anything/_run_record_comparison.py) — comparison result and report assembly.
* [tests/test_run_record.py](../tests/test_run_record.py) — canonical/migration digest, API/module/pickle, tamper/symlink, and fresh-process snapshots.
* [docs/sprint-plans/sprint-78.md](../docs/sprint-plans/sprint-78.md) — marked atomic task 78.23 complete.

## Metrics and SCCs

`run_record.py` changed from 812 LOC / 714 nonblank / 5,247 AST nodes / 41
functions / 6 classes to a 39-LOC / 58-AST facade. The extracted modules are:

* `_run_record_codec.py`: 192 LOC / 1,234 AST nodes / 9 functions.
* `_run_record_schema.py`: 274 LOC / 1,874 AST nodes / 10 functions / 2 classes; largest class 204 LOC.
* `_run_record_persistence.py`: 281 LOC / 1,852 AST nodes / 20 functions / 3 classes; largest class 238 LOC.
* `_run_record_comparison.py`: 66 LOC / 349 AST nodes / 2 functions / 1 class.

The static source graph remains at **7 SCCs**, with no new run-record cycle:
TCAV facade/statistics/model, RSSM facade/evaluation, LeRobot dataset bridge,
SmolVLA facade/runtime/metrics/loader, benchmark facade/environment/execution/
statistics, Gaussian renderer bridge, and SAE facade/metrics/atlas. Graphify
updated to **10,701 nodes / 20,694 edges / 948 communities**.

## Compatibility and Testing

* Schema-v1 canonical JSON digest: `25a8bc21cf19a67ce9a553d469236e06f19aed0b96760f9b97e3d2ed3b3c4964`.
* Legacy migration digest: `fa243e2d7ca35695c6d381940d844312fe8df8476c8d88ef4da5f063406f5083`.
* Focused run-record/recorder/artifact/portable/tracking tests: **74 passed, 2 skipped**.
* Repository Ruff: **passed**.
* Repository format: **250 files already formatted**.
* Strict Pyright: **0 errors, 0 warnings, 0 informations**.
* Full suite: **1543 passed, 36 skipped, 39 warnings** in 394.77s.
* `git diff --check`: passed; only existing LF→CRLF working-tree warnings.

Coverage includes schema bytes and migration, identity exclusion of lifecycle
fields, nested-input freezing, status transitions, atomic writes, duplicate
identity errors, Windows artifact-path migration, content addressing, portable
artifact checksums, hostile path/symlink/tamper rejection, comparison deltas,
public signatures/module identities, pickle-capable public value objects, and
fresh-process reads. No external recorder/provider or public protocol changed.

## Review

**PASS-WITH-WARNINGS.** All required gates are green and no Blocking finding
was identified. The only warnings are existing deprecation/test warnings and
the deliberate fact that `RunRecord` retains its historical non-pickleable
`mappingproxy` behavior; pickle identity is verified for the public value
objects that were pickle-capable before the refactor.
