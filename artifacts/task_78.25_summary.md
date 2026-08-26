# Task Summary: Sprint 78.25 — exhaustive source and maintainability audit

**Sprint:** Sprint 78  
**Task:** 78.25 — final read-only SRP, API, dependency, and safety review

## Verdict

# Latent-Anything Review — PASS-WITH-WARNINGS

No Blocking finding remains. One Advisory maintainability candidate is
deferred by the existing concrete-generation ADR; no source refactor was
authorized or performed in this audit.

## Scope and Coverage

Reviewed every current Python source module under `src/latent_anything/` (137
modules) and all test files (114 modules), including the post-78.24 private
modules, public facades, optional integrations, recorder/artifact paths, and
compatibility snapshots. Graphify query/path/explain were run before source
inspection. The graph-first query connected diffusion capture/analysis to the
LeRobot context, run-record persistence to schema/artifact paths, and
LatentSpace to geometry dispatch and immutable Trajectory consumers.

## Metrics and Dependency Topology

| Measure | Current audit result |
| --- | ---: |
| Source modules / source LOC | 137 / 34,712 |
| Source AST nodes | 184,164 |
| Functions / classes | 1,498 / 288 |
| Static import edges | 393 |
| Strongly connected components | 7 |
| Graphify topology | 10,716 nodes / 20,690 edges / 950 communities |
| Public top-level exports | 202 / 202 present |

Compared with the early Sprint78 graph snapshot (10,200 nodes / 19,775 edges /
920 communities), the graph grew by 516 nodes, 915 edges, and 30 communities
as the completed atomic extractions and compatibility tests landed. This is a
repository evolution comparison, not a claim that all growth is source
complexity.

Largest current source surfaces were measured as follows:

| Module | LOC | Largest class | Largest function | Max function complexity | Judgment |
| --- | ---: | ---: | ---: | ---: | --- |
| `integrations/lerobot_benchmark.py` | 896 | 85 | 52 | 11 | Cohesive benchmark facade/result schema after 78.20 |
| `tcav.py` | 884 | 63 | 111 | 8 | Cohesive statistical facade after 78.5–6 |
| `adapters/jepa.py` | 743 | 481 | 65 | 8 | One adapter lifecycle; private training/evaluation seams already extracted |
| `rssm.py` | 730 | 503 | 54 | 8 | One recurrent transition lifecycle; facade/result ownership is coherent |
| `sae_evaluation.py` | 728 | 165 | 114 | 12 | One SAE evaluation/atlas lifecycle after 78.7 |
| `visualization/data.py` | 714 | 48 | 49 | 10 | Typed renderer-input builders only; no plotting/model work |
| `probes.py` | 680 | 254 | 139 | 20 | Linear-probe lifecycle plus controls; split ownership is focused |
| `integrations/lerobot_dataset.py` | 667 | 73 | 118 | 31 | One raw-object dataset bridge with descriptor/read/stream views |
| `experiment_recorder.py` | 658 | 175 | 78 | 26 | Frozen provider-neutral contract plus local lifecycle adapter |
| `geodesic.py` | 602 | 299 | 117 | 19 | One density-geodesic algorithm and immutable result schema |
| `projection.py` | 601 | 257 | 40 | 8 | Orthonormal-subspace construction and projection operations |
| `integrations/diffusers_conditional.py` | 594 | 385 | 150 | 16 | Single concrete diffusion-generation lifecycle |
| `latent_space.py` | 539 | 493 | 76 | 23 | Explicit geometry-dispatch facade, not a strategy dumping ground |
| `mlp_probe.py` | 527 | 226 | 131 | 18 | Public probe facade after 78.22 |
| `integrations/lerobot_diffusion.py` | 574 | 226 | 111 | 15 | Stable policy capture facade after 78.24 |

The seven SCCs are known compatibility/facade groups: benchmark, SmolVLA,
RSSM, SAE, TCAV, Gaussian renderer, and LeRobot dataset bridge. They are
covered by facade import-order/module-identity tests and no new SCC was
introduced by 78.24. The diffusion extraction specifically has no cycle.

## Tooling Gate

- ruff check: **pass** — `All checks passed!`
- ruff format: **pass** — 251 files already formatted.
- pyright (strict): **pass** — full `src tests` run returned 0 errors, 0 warnings,
  0 informations; the current scoped rerun of public/high-risk modules also
  returned 0/0/0.
- pytest: **pass** — the final full-suite run on the unchanged source/test tree
  was 1,545 passed, 36 skipped, 39 warnings. Current focused audit suites were
  275 passed, 9 skipped, 1 warning (probes, MLP, visualization, clustering,
  geodesic, projection, diffusion) and 27 passed (optional imports,
  visualization isolation, async runtime, artifact store, disk cache).
- API/compatibility subset: **87 passed** (exports, registry, portable,
  run-record schema/pickle/digests, and migration).
- `git diff --check`: **pass**; only existing LF→CRLF working-tree warnings.

## Compatibility, Boundary, and Integrity Findings

- The top-level export snapshot is exactly **202 names**, with no missing
  attributes. Dynamic signature inspection found no `torch.Tensor`, `Tensor`,
  or `torch.nn` type in any top-level public export; static inspection found no
  such leak in public definitions in non-private modules.
- Optional backend isolation passes: base, LeRobot, and visualization imports
  do not eagerly load their backends; missing-extra errors remain actionable.
- Raw LeRobot objects stay behind the documented lazy bridge. NumPy conversion
  occurs at captured-latent boundaries; torch remains internal to model-bound
  paths.
- Artifact/run-record tests cover schema-v1 canonical bytes and migration,
  pickle identity, atomic writes, traversal/symlink rejection, tamper and
  checksum failure, and cross-process reads. Portable tests reject object-array
  pickle fallback and schema/manifest tampering.
- Public API/result/schema/signature snapshots cover the 202 exports, registry,
  portable envelopes, recorder, world models, probes, TCAV/SAE, transformer,
  LeRobot benchmark/SmolVLA/diffusion, and visualization views.
- No tautological or weakened assertion was found in the reviewed compatibility
  and 78.24 tests; negative paths exercise real failure conditions.
- Geometry/adapter/Trajectory ADRs are respected: geometry carries dispatch,
  adapters support the three validated modes, and Trajectory operations remain
  immutable. No premature public Protocol or cross-policy capture abstraction
  was introduced.

## Findings

### Advisory

1. `src/latent_anything/integrations/diffusers_conditional.py:353`,
   `DiffusersConditionalPipeline.generate` (150 LOC, complexity 16) combines
   backend execution, scheduler/denoiser capture, intervention callbacks, and
   result assembly. A future extraction seam could isolate a concrete capture
   session and result assembler; required regression coverage is the existing
   fake-scheduler/hook/order/result suite in
   `tests/test_diffusers_conditional.py`. This is deferred, not a blocker:
   the Sprint37 ADR explicitly keeps the first full generative lifecycle
   concrete and forbids a generic generation protocol before a second
   differing implementation.

### Blocking

None.

## Prior Finding Closure

| Prior finding family | Evidence / status |
| --- | --- |
| Portable/TCAV/transition/RSSM/JEPA/tokenized-world-model monoliths | Closed by 78.4–13; facades and focused ownership modules retain parity snapshots |
| Recorder/reward/transformer/SmolVLA/benchmark oversized responsibilities | Closed or bounded by 78.14–20; focused tests and public schema snapshots pass |
| Repository Ruff B009 in SmolVLA test | Closed by 78.21; Ruff repository gate passes |
| MLP training/control/split ownership | Closed by 78.22; exact split digest, deterministic controls, and TCAV/probes compatibility pass |
| Run-record codec/schema/persistence/comparison ownership | Closed by 78.23; canonical schema/migration digests, path safety, pickle, and recorder tests pass |
| LeRobot Diffusion capture vs. analysis/result ownership | Closed by 78.24; public analysis digest/module/signature and hook-failure tests pass |
| Public export/import/schema compatibility | Closed/verified: 202/202 exports and compatibility subset pass |
| Torch/raw-object and optional-extra boundaries | Closed/verified by static scans and 27 isolation/async/artifact tests |

## Deferred Cohesive Large Modules — Do Not Split Now

`latent_space.py`, `geodesic.py`, `projection.py`, `probes.py`,
`visualization/data.py`, `integrations/lerobot_dataset.py`, `adapters/jepa.py`,
`rssm.py`, `reward_value.py`, `experiment_recorder.py`, and
`integrations/lerobot_diffusion.py` are large but each has one domain owner,
typed result/config boundaries, focused tests, and an ADR-backed concrete
facade. Splitting them now would create reverse dependencies or speculative
Protocols without a third differing implementation. `run_record.py` is now a
stable facade, and the benchmark/SmolVLA/TCAV/SAE facades already have focused
private ownership modules.

## Next Atomic Tasks

No necessary remediation task is recommended. If a second differing
generative integration lands, revisit the deferred Diffusers generation seam
as a new atomic task with API/result/order snapshots and offline hook tests.
Otherwise retain the current concrete shapes and address only release/API
freeze documentation items outside this audit.

**Final review verdict: PASS-WITH-WARNINGS.** The only warning is the deferred
Diffusers maintainability candidate above; no SRP, API, dependency, security,
typing, optional-isolation, or test-integrity blocker remains.
