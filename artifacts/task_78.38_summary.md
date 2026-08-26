# Sprint 78.38 — Theory/evidence ledger gap plan

## Verdict

**PASS-WITH-WARNINGS for planning.** The current evidence ledger remains
honest and unchanged. The exact remaining implementation-applicable and
benchmark-only D0/D1 rows are enumerated once in
[`task_78.38_gap_map.json`](task_78.38_gap_map.json) and described in the
[evidence-gap closure plan](../docs/EVIDENCE_GAP_PLAN.md). No evidence level,
metric, threshold, or historical artifact was promoted or rewritten.

## Authoritative coverage

`uv run python scripts/validate_evidence_ledger.py` reports:

```text
Inventory: 107 capabilities
core: 25/63 (39.7%)
overall: 25/65 (38.5%)
```

The non-qualifying inventory is **40 rows: 29 D0 and 11 D1; 38 core and 2
non-core; 19 headline and 21 non-headline; 24 D2 targets and 16 D3 targets**.
The core gate needs 35 more qualifiers to reach 60/63 (95%); the overall gate
needs 34 more to reach 59/65 (90%). The core gate is binding. The map has no
duplicate or missing IDs and its SHA-256 is
`5b20cbd2fbce93f60409ea0950d33ea1d1cd0254e4c24fe55b9269f8492cfb85`.

### Lane distribution

| Lane | Rows | Target | Scope |
|---|---:|---|---|
| L01 | 1 | D2 | core primitives |
| L02 | 6 | D2 | geometry/trajectory |
| L03 | 3 | D2 | probes/linear structure |
| L04 | 5 | D2/D3 | explanation/intervention |
| L05 | 3 | D2 | density/transport |
| L06 | 2 | D2/D3 | SAE/dictionary |
| L08 | 1 | D2 | beta-VAE |
| L12 | 4 | D2/D3 | JEPA/LeWM/collapse |
| L13 | 4 | D2 | quantization |
| L14 | 2 | D3 | GAIA-1/Genie |
| L15 | 2 | D2/D3 | stochastic/RSSM |
| L16 | 4 | D2/D3 | planning |
| L17 | 2 | D3 | 3DGS |
| L19 | 1 | D3 | OpenVLA |

The machine-readable map contains row-level capability, current evidence and
shortfall, target level, exact M14 lane, prerequisites, command/test selector,
seed/sample/metric acceptance contract, artifact path, cleanup/security,
blocker, and dependencies. Its consistency check reported zero missing
required fields, bad IDs, bad dependencies, or malformed lane identifiers.

## Required blockers kept explicit

- **3DGS L17:** remains blocked because no named checkpoint, revision,
  license/access record, or remote CUDA lane is provisioned. The reference
  renderer is not promoted to real D3 evidence.
- **SmolVLA L21:** the causal claim remains D2 pending a corrected pinned
  CUDA/Linux rerun. The historical D3 artifact remains unverified and is not
  counted. The rerun must preserve model/dataset revisions, controls, seed,
  simulator threshold, signed artifact, and cleanup evidence.
- Other blockers include missing named model/checkpoint access or licenses,
  unimplemented rows (VQGAN, normalizing flows, transport, V-JEPA, residual
  VQ, FSQ, GAIA-1, Genie, policy gradient, MuZero, MCTS), CUDA/VRAM, and
  external GitHub Actions access. These are execution blockers, not waiver
  opportunities to lower thresholds.

## Ordered closure policy

Sprint 79 should execute bounded local D2 batches first (L01/L02, L03–L06,
L08, L12–L16), then named real-model D3 lanes, while preserving all negative
results. Every promotion requires the typed source/test/benchmark/config
records and, for D3, an immutable artifact with model/dataset revision,
license/access, seed, environment, resource, network, cleanup, and SHA-256.
Owner waivers must be scoped, reasoned, dated, and explicit about core impact;
no waiver may hide a core gap. Sprint 80 remains stop-before-release until the
validator reaches 95% core and 90% overall and all headline D3 claims pass.

## Gates and graph

- Evidence validator: **PASS integrity / honest gate failure** at 25/63 and
  25/65; no IDs, statuses, classifications, or evidence links changed.
- Deterministic map reconciliation: **PASS**, 40 expected / 40 mapped,
  duplicates 0, missing 0, extra 0.
- Graphify query was run before planning. Graphify refresh is required after
  this artifact/plan update; no source or model/network/CUDA execution is in
  scope for 78.38.
- Final graphify refresh completed after the artifact and plan edits:
  **11,080 nodes / 21,186 edges / 959 communities**. Graphify reported 53
  JSON sidecars with zero extracted nodes; these are extraction warnings and
  do not alter the validator or the row-level coverage map.
