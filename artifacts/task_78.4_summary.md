# Task Summary: Portable codec SRP refactor (78.4)

**Sprint:** Sprint 78  
**Task:** 78.4  
**Status:** Complete

## Scope and outcome

Extracted the recursive portable-node encoder/decoder and the shared limits,
shape, byte-accounting, and canonical-JSON contract into focused private
modules. `latent_anything.portable` remains the stable public facade. This is a
pure internal refactor: no changelog entry is required and no public API,
accepted value type, Arrow schema, schema-version bytes, exception text, or
resource guard was changed.

## Files

- `src/latent_anything/portable.py` — stable Arrow facade (`__all__` unchanged).
- `src/latent_anything/_portable_contract.py` — typed limits and validation helpers.
- `src/latent_anything/_portable_nodes.py` — typed recursive encode/decode handlers.
- `tests/test_portable.py` — exact wire-fixture, API-surface, hostile-input, tamper, and property coverage.
- `docs/sprint-plans/sprint-78.md` — task status.

## Architecture and metrics

Before: `portable.py` was 467 LOC with 14 functions and 4 classes; its
`_Encoder` and `_Decoder` were approximately 142 and 129 LOC and combined
Arrow framing, recursive traversal, domain reconstruction, and resource
validation. After: the public facade is 128 LOC with 2 functions and no
classes; `_portable_contract.py` is 83 LOC with 4 functions and 2 classes; and
`_portable_nodes.py` is 296 LOC with 8 functions and 2 classes. The facade
depends on the two internal modules; the node handlers depend on the contract
and domain value modules, with no back-edge into the public facade. No generic
serialization protocol or new public abstraction was introduced.

The public surface remains exactly:

```text
PortableLimits, PortableNodeError, decode_portable, encode_portable
```

The internal class definitions set their historical public module identity so
existing exception and dataclass representations remain stable without making
the internal modules public.

## Parity and security evidence

The following existing wire fixtures retain exact byte length and SHA-256:

| Value | Bytes | SHA-256 |
| --- | ---: | --- |
| `{"answer": 42}` | 1102 | `047a851c57c4c787f32383629aff2baf6d2b8306be4d92ac4250d4ab1a649078` |
| big-endian `>i4` array with nested tuple/bytes | 1770 | `576bfbf9a7f588b80c73e8db602942f2d0f4398fde367bb516e7724ed1439e3d` |
| float32 matrix plus scalar list | 1706 | `f334cd21c4aa691e63af928552b5b7b27ccb48c4cc16cd89fbf4852221776ff1` |

Focused tests also prove cycle, object-dtype, input/manifest/rank/row,
maximum-depth, total-allocation, schema-version, manifest-kind, and complete
public-`__all__` protections. The Hypothesis list round-trip property covers
repeated bounded scalar/container recursion.

## Validation

- Focused portable/artifact/cache/run-record tests: **26 passed** (`tests/test_portable.py`, `tests/test_artifact_store.py`, `tests/test_disk_cache.py`, `tests/test_run_record_portable.py`).
- Strict Pyright (`uv run pyright src tests`): **0 errors, 0 warnings, 0 informations**.
- Ruff check: **All checks passed**; format check: **210 files already formatted**.
- Full default pytest: **1509 passed, 36 skipped, 39 warnings in 159.25s**. Warnings are the existing sklearn convergence, registry deprecation, and UMAP user warnings.
- Final `git diff --check`: **pass** (only normal CRLF conversion warnings for dirty tracked files).
- Final graphify topology: **10,237 nodes / 19,818 edges / 906 communities** after the artifact update. Graphify reported the known 50 non-code JSON files with zero AST nodes and a community-label refresh recommendation; no source extraction failure occurred.

## Review and constraints

The latent-anything-review verdict is **PASS**: focused tests, Ruff, format,
strict Pyright, full pytest, diff-check, and graphify all completed
successfully. No model download, network validation, remote CUDA, commit, or
push was performed. No changelog entry is included because behavior and the
public contract are unchanged.
