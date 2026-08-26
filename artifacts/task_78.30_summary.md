# Sprint 78.30 — deterministic API-freeze compatibility snapshot

## Result

Added the checked-in machine-readable snapshot
[`api_freeze_snapshot_0.1.0b1.json`](api_freeze_snapshot_0.1.0b1.json) and its
generator/comparator [`api_freeze_snapshot.py`](../scripts/api_freeze_snapshot.py).
The comparator fails closed and prints each drifted section/name path, so a
public surface change cannot silently pass review. The entrypoint delegates to
two cohesive private helpers: `_api_freeze_inventory.py` owns public/schema
inventory, while `_api_freeze_runtime.py` owns observed aliases, CLI,
serialization, async-pair, exception, and normalization logic.

Regenerate or verify with:

```text
uv run python scripts/api_freeze_snapshot.py --write
uv run python scripts/api_freeze_snapshot.py --check
```

The JSON uses sorted keys, deterministic declaration order only where the
contract requires it, normalized defaults/annotations, and no object reprs,
addresses, or process-dependent identities.

## A–L inventory

| Section | Inventory | Count | SHA-256 |
|---|---|---:|---|
| A | Current top-level exports; canonical stable projection; RFC0001 additions | 205 current / 202 canonical stable | `8226afc5414e3c13074cc9f96ec7a80132ae401cc1e2c89f0c0adc82f53bb880` |
| B | Beta policy plus observed symbol/registry/CLI/config/result compatibility | 3 symbol aliases; 2 registry aliases; 2 CLI aliases; 1 MPPI input alias; 3 result aliases | `1b9d0d300fdd5295e9c8463acc7bef63a41120a4f0f459ec795b5daf630f0922` |
| C | Public submodule reexports and protocol modules | 8 modules | `98c9d79bd169954b22373711bec13e285790e956381fed42788227b57bd9b3b3` |
| D | Built-in registry rows | 32 | `e8fd704a1ee60db288201eb9b02754e0c8474819b1266687671e42b803ed1ae8` |
| E | Plugin groups and API version | 5 / API `1` | `8e3c5f6982cb5f2b258406774b445b3c3e495f7f23df978c8d5641882ac362f5` |
| F | Optional profiles | 12 | `3fd27b66b977bd286621a89821d59034fa69629b0a4c64bdd23bc1f5b917d1ae` |
| G | Config/spec/limits field/default schemas | 28 | `8ce2a8a8a170e6cfe0bc8cf788f719bd0401c1e0e254964bed4b7ed23fa89fc7` |
| H | Public dataclass/result schemas | 81 | `19339f6a76379d16decde5c4d9c563de2271a7f367d168033613cc947df0adfb` |
| I | Live argparse command/options/exit contracts | 5 commands | `d87cb6aa71989f19488e17b4a2e504f6350f6946d6f291314a8bd6fe5da96baf` |
| J | Runtime-derived portable/run-record/artifact/cache versions and fixture evidence | 5 serialization families; retained golden result digest plus observed builder digest | `665d14dc4a99a3e9d88c2a7898614ba5be78aa52b7a07baaf190863a15d19688` |
| K | Dynamically discovered public sync/async pairs and coroutine kinds | 9 | `66889c6764310bb22e5db60d2e79916aa872de79ee1494d7c8d4d52177398458` |
| L | Dynamically discovered custom exception taxonomy | 7 | `5f17dc95ec1ea977bf5e04c01c5409fd41805cc766ae498233803f5b620e2887` |

Snapshot document SHA-256: `ed13a757798567571c8c0ce643f7bd51af217a38afccaf7c8627c55edc3f3726`.

## Owner correction

The original result-envelope golden digest remains `815ea47a...050ea` in the
checked-in snapshot. The runtime builder currently observes
`272c8459...68458a`; it is recorded separately rather than replacing the
reviewed golden. This makes fixture drift explicit while preserving the prior
compatibility expectation. The disk-cache observation is corrected to the
runtime constant `disk-cache-v1` (not `state-aware-v1`). CLI, aliases, async
pairs, and exception rows are runtime observations rather than duplicated
policy tables. `--check` only reads the snapshot and reports diagnostics; it
does not write files.

## Compatibility policy

`AnalysisMethod`/`Method`, `Intervention`/`BMethod`, and
`InterventionPipeline`/`ManipulationPipeline` remain exact identity-compatible
aliases through the beta deadline `0.9.0`. Registry kind constants and
`method_a`/`method_b`, CLI aliases, MPPI `lambda`/`lambda_`, and result/property
aliases are recorded as retained compatibility surfaces. No alias is removed,
no migration guide or API-freeze ADR is created, and package metadata remains
`0.1.0b1`.

The 202-entry canonical-stable count is the baseline projection with the
legacy symbol slots replaced by canonical names. The current runtime surface is
205 because canonical names are additive during the beta window; `BMethod` was
not a top-level export in the prior beta snapshot and is represented by the
methods submodule compatibility section.

## Verification

- Snapshot comparator, helper, and CLI separation tests: **6 passed**
  (including the four API-surface tests).
- CI-equivalent locked `viz` profile: `uv sync --locked --extra viz`.
- Full suite under that profile: **1552 passed, 36 skipped, 39 warnings**;
  this is the prior `1545/36` baseline plus one RFC0001 test and six snapshot
  drift/separation/helper/error-contract tests.
- Scoped `ruff check src tests`: PASS.
- Scoped `ruff format --check src tests`: PASS (`251 files already formatted`).
- Strict Pyright: PASS (`0 errors, 0 warnings, 0 informations`), including
  the generator via `uv run pyright scripts/api_freeze_snapshot.py`.
- `git diff --check`: PASS, with existing LF/CRLF normalization warnings only.
- Final graphify topology: **10,819 nodes / 20,861 edges / 935 communities**;
  includes the split private helpers and checked-in generator/test dependency
  edges. `graphify update . --force --no-cluster` followed by
  `graphify cluster-only . --no-viz --no-label` was used after the final
  artifact/code updates.

## Review verdict

`PASS-WITH-WARNINGS`: no blocking findings; root-wide Ruff scans bundled
skill/theory notebooks outside the project `src` + `tests` gate and is not used
as a task finding. Future public-surface changes must regenerate and review the
snapshot before merge.
