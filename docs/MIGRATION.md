# Migration and compatibility guide

This guide describes the current `0.1.0b1` beta surface and the planned
pre-stable `0.9.0` compatibility epoch. It does **not** announce a `0.9.0`
release: no `v0.9.0` tag or publication is authorized yet, and package metadata
remains `0.1.0b1` until the Sprint 78 gates and release workflow are verified.
The authoritative machine-readable surface is
[`api_freeze_snapshot_0.1.0b1.json`](../artifacts/api_freeze_snapshot_0.1.0b1.json);
the human alias policy is [`API_COMPATIBILITY.md`](API_COMPATIBILITY.md).

## From the published beta

The published `0.1.0-beta.1` core beta remains supported. Existing imports,
configuration files, result properties, CLI spellings, and local serialized
data remain readable during the beta window. New code should use the canonical
spellings below. This is an additive migration; this document does not remove
an alias or rewrite historical release artifacts.

The snapshot records **205 current runtime top-level exports** and a **202-entry
canonical-stable projection**. The three additive entries are
`AnalysisMethod`, `Intervention`, and `InterventionPipeline`; the legacy
`Method`, `BMethod`, and `ManipulationPipeline` identities remain available.
`BMethod` was not a beta top-level export, but remains available through its
public methods submodule and has an additive top-level canonical counterpart.

## Naming migration

| Canonical spelling | Legacy spelling/path | Current state | Planned policy |
|---|---|---|---|
| `AnalysisMethod` | `Method` (`latent_anything`, `latent_anything.methods.protocols`) | Added Unreleased/Sprint 78.29 under metadata `0.1.0b1`; exact identity alias | Deprecation notice is current Unreleased/Sprint 78.31; removal is planned for `0.9.0`, subject to a separate reviewed decision |
| `Intervention` | `BMethod` (`latent_anything.methods.b_protocols`) | Added Unreleased/Sprint 78.29; exact runtime-checkable Protocol identity | Same planned `0.9.0` removal policy; no import-time warning |
| `InterventionPipeline` | `ManipulationPipeline` (`latent_anything`, `latent_anything.manipulation_pipeline`) | Added Unreleased/Sprint 78.29; exact class identity/behavior | Same planned `0.9.0` removal policy; no import-time warning |

The RFC0001 `0.2.0` window was planned but that release was never published.
It must not be used as a historical `since` or deprecation release. Registry
canonical kinds landed in Sprint 31, also under `0.1.0b1` metadata.

## Complete beta alias ledger

The following are the **18 human-ledger alias rows**, expanded from snapshot
section B (`lambda` and `lambda_` count as separate rows). Rows 1–5 have a
current Unreleased deprecation notice (registry aliases warn at config
construction); rows 6–18 remain supported without a removal deadline. The two
schema/path data migrations below are separate and are not included in this
alias-row count.

| # | Canonical | Legacy | Contract |
|---:|---|---|---|
| 1 | `AnalysisMethod` | `Method` | Exact object identity |
| 2 | `Intervention` | `BMethod` | Exact Protocol identity and runtime conformance |
| 3 | `InterventionPipeline` | `ManipulationPipeline` | Exact class identity and behavior |
| 4 | registry `analysis` | `KIND_METHOD_A` / `method_a` | One `DeprecationWarning` at config construction; lookup is quiet |
| 5 | registry `intervention` | `KIND_METHOD_B` / `method_b` | One `DeprecationWarning` at config construction; lookup is quiet |
| 6 | CLI `capture-points` | `list-capture-points` | Same parser, exit code, and JSON output |
| 7 | CLI `replay-run` | `replay-run-config` | Same parser, dispatch, exit code, and failure message |
| 8 | `MPPIConfig.temperature` | input `lambda` | Same validated value |
| 9 | `MPPIConfig.temperature` | input `lambda_` | Same validated value |
| 10 | `CEMPlanResult.actions` | `selected_actions` | Same read-only array property |
| 11 | `MPPIPlanResult.actions` | `selected_actions` | Same read-only array property |
| 12 | `RolloutResult.trajectory` | `states` | Same read-only trajectory property |
| 13 | `DeterministicLatentTransition.step` | `predict` | Same numeric/error behavior; wrapper identity is not promised |
| 14 | `StochasticGaussianLatentTransition.step` | `predict` | Prediction type remains distinct from mean-returning `step` |
| 15 | `GaussianPrediction.scale` | `std` | Same read-only array value |
| 16 | `StochasticOneStepMetrics.negative_log_likelihood` | `nll` | Same scalar property |
| 17 | `StochasticRolloutMetrics.negative_log_likelihood_by_horizon` | `nll_by_horizon` | Same tuple property |
| 18 | `StochasticRolloutMetrics.mean_error_by_horizon` | `errors_by_horizon` | Same tuple property |

Use the exact identity/behavior tests in `tests/test_api_compatibility.py`,
`tests/test_api_freeze_snapshot.py`, `tests/test_registry_migration.py`, and
the transition/CLI suites before changing a call site.

## Data migrations

Data migrations are separate from spelling aliases and have no removal deadline
while compatible readers remain supported:

1. `result-envelope-v0` → `result-envelope-v1` through
   `latent_anything.portable_results.decode_result_envelope`. The migration is
   explicit, local, allowlisted, and emits no warning; unknown versions fail
   closed.
2. Pre-versioned run records and legacy Windows artifact paths → run-record
   `schema-v1` through `latent_anything.run_record.migrate_run_record`. The
   canonical record is validated before use; path traversal, symlinks,
   malformed data, and checksum failures remain errors.

Do not pickle untrusted data or treat a successful migration as evidence that a
model/checkpoint is reproducible. Portable values use `portable-node-v1` and
artifact storage uses `artifact-envelope-v1`; the exact fixture versions and
digests are in snapshot section J.

## Versioned compatibility policy

The planned sequence is:

1. Keep beta metadata at `0.1.0b1` while Sprint 78 inventory, evidence,
   documentation, and release-workflow gates are incomplete.
2. Review the canonical/legacy surface at the planned `0.9.0` compatibility
   epoch. A separate owner-approved migration decision is required before any
   removal; this guide does not authorize removal.
3. Stop before `1.0.0` publication if the evidence validator is below 95% core
   or 90% overall, headline D3 evidence is missing, an external Actions account
   is unavailable, or any required packaging/docs/security gate fails.

The current evidence validator and M14 matrix remain the authority for release
readiness; this guide only makes the migration contract discoverable.
