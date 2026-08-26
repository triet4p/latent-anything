# M14 planning-contract task artifact

Status: documentation contract complete; M14 implementation and evidence remain pending.

## Scope

- Added `docs/M14_REAL_SYSTEM_VALIDATION.md` with 24 normative lanes and an
  acceptance contract covering capability/public symbols, source, current
  tests/evidence tier, model/revision/dataset/backend, license/access,
  environment, command, deterministic acceptance, artifact, resources,
  credentials/network, cleanup, status, blocker, and waiver owner.
- Coverage is explicit: 202 top-level exports in eight exact groups; 32 built-in
  registry entries; 5 plugin groups; 12 optional profiles; five CLI commands;
  portable/result/artifact/run-record schemas; negative/security, sync/async,
  plugin-install, and cross-adapter composition lanes.
- Added explicit non-API/backlog or blocker status for SAM, OpenCLIP, timm,
  Torchvision model adapters, Open3D, trimesh, and unnamed 3DGS checkpoints.
- Expanded Sprints 78–80 with SRP audit/refactor, stale-doc conflict cleanup,
  remote-CUDA invariants, external GitHub Actions account blocker, and
  stop-before-release conditions. Updated `docs/PLAN.md` without marking M14
  complete and linked the contract from `docs/INDEX.md`.

## Validation contract

The executor must run docs/plan validation, `uv run mkdocs build --strict`,
then immediately `graphify update .`; record the resulting topology, warnings,
and any generated/ignored output. No source or test changes, model download,
remote connection, commit, or push are part of this task.

## Recorded validation result

- Coverage contract: PASS — 202/202 exports with no missing or duplicate names,
  24/24 lanes, all 32 registry entries, 5 plugin groups, and 12 profiles.
- `uv run python scripts/validate_evidence_ledger.py`: PASS; current ledger is
  25/63 core (39.7%) and 25/65 overall (38.5%). These are recorded as current
  gaps, not hidden by this planning task.
- `uv run mkdocs build --strict`: PASS. The command emitted only the upstream
  Material-for-MkDocs 2.0 warning; no broken-link or strict-build error.
- `graphify update .`: PASS. Final topology after the required updates: 10,188
  nodes, 19,750 edges, 907 communities (1,166 cross-community edges). Graphify reported 50 JSON/artifact source files
  producing zero AST nodes (including `hooks.json`, `evals.json`,
  `act_policy_representation_benchmark.json`, and `anisotropy_benchmark.json`);
  these are non-code/generated evidence inputs and were not edited. It also
  reported 908 saved labels versus 907 current communities and renamed 18
  communities by hub; no LLM relabel was run.
- Graph outputs and `.gh-pages-build` remain expected generated/ignored state;
  no source or test files were modified.

## Known blockers

The real conditional-diffusion, 3DGS, LeRobot policy, I-JEPA, and external
tracking lanes require model/data access, licenses, hardware, or credentials.
The GitHub Actions account requirement is external. The current evidence ledger
also records thresholds and historical failures that must not be rewritten.
