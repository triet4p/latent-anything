# API compatibility and deprecation ledger

This English ledger describes the checked-in `1.0.0` candidate API and its
compatibility aliases. The candidate snapshot
[`api_freeze_snapshot_1.0.0.json`](../artifacts/api_freeze_snapshot_1.0.0.json)
has not been released. The published `0.9.0` release-time inventory (205
runtime exports, 202 canonical-stable entries, 7 exceptions; digest
`048ac553adabb11c24d3e1f4d86e0c6d469df6590064014c915d6a08c7d53c26`) is
recorded in the historical root `CHANGELOG.md`. The checked-in
[`api_freeze_snapshot_0.9.0.json`](../artifacts/api_freeze_snapshot_0.9.0.json)
is a later pre-candidate source snapshot, not the release-time contract; its
historical beta counterpart remains
[`api_freeze_snapshot_0.1.0b1.json`](../artifacts/api_freeze_snapshot_0.1.0b1.json).
RFC0001 planned a `0.2.0` vocabulary window, but that release was never
published. Canonical symbols and registry kinds were added under historical
`0.1.0b1` metadata and retained in the `0.9.0` release and the current
candidate. No alias is removed by this ledger.

## Alias records

| # | Canonical spelling | Legacy spelling / path | RFC planned window; actual implementation/release; deprecation/removal | Warning and guarantee | Verification |
|---:|---|---|---|---|---|
| 1 | `AnalysisMethod` | `Method` (`latent_anything`, `methods.protocols`) | Planned RFC `0.2.0` (never released); implemented Unreleased / Sprint78.29 (`0.1.0b1` metadata), retained in `0.9.0` and 1.0.0 candidate | Exact object identity; no import warning because import-time warnings would punish every consumer | Retained through 1.x; no earlier than a separately reviewed major-version migration |
| 2 | `Intervention` | `BMethod` (`methods.b_protocols`) | Planned RFC `0.2.0` (never released); implemented Unreleased / Sprint78.29 (`0.1.0b1` metadata), retained in `0.9.0` and 1.0.0 candidate | Exact Protocol identity; no import warning for the same reason | Retained through 1.x; no earlier than a separately reviewed major-version migration |
| 3 | `InterventionPipeline` | `ManipulationPipeline` (`latent_anything`, `manipulation_pipeline`) | Planned RFC `0.2.0` (never released); implemented Unreleased / Sprint78.29 (`0.1.0b1` metadata), retained in `0.9.0` and 1.0.0 candidate | Exact class identity and behavior; no import warning | Retained through 1.x; no earlier than a separately reviewed major-version migration |
| 4 | `analysis` | `KIND_METHOD_A` / `method_a` registry kind | Planned RFC `0.2.0` (never released); implemented Unreleased / Sprint31 (`0.1.0b1` metadata), retained in `0.9.0` and 1.0.0 candidate | One `DeprecationWarning` at config construction; registry lookup remains quiet | Retained through 1.x; no earlier than a separately reviewed major-version migration |
| 5 | `intervention` | `KIND_METHOD_B` / `method_b` registry kind | Planned RFC `0.2.0` (never released); implemented Unreleased / Sprint31 (`0.1.0b1` metadata), retained in `0.9.0` and 1.0.0 candidate | One `DeprecationWarning` at config construction; registry lookup remains quiet | Retained through 1.x; no earlier than a separately reviewed major-version migration |
| 6 | `capture-points` | CLI `list-capture-points` | Pre-Sprint78 implementation; retained in `0.9.0` and 1.0.0 candidate; not deprecated, no removal planned | Same argparse parser, exit code, and JSON output; warning is impractical for a command alias | `test_cli.py`; snapshot B/I |
| 7 | `replay-run` | CLI `replay-run-config` | Pre-Sprint78 implementation; retained in `0.9.0` and 1.0.0 candidate; not deprecated, no removal planned | Same argparse parser, dispatch, exit code, and failure message; warning is impractical for a command alias | `test_cli.py`; snapshot B/I |
| 8 | `temperature` | `MPPIConfig(lambda=...)` | Pre-Sprint78 implementation; retained in `0.9.0` and 1.0.0 candidate; not deprecated, no removal planned | Pydantic validation produces the same temperature; no warning because field aliases are validated before a construction warning seam | `test_mppi.py`; snapshot B |
| 9 | `temperature` | `MPPIConfig(lambda_=...)` | Pre-Sprint78 implementation; retained in `0.9.0` and 1.0.0 candidate; not deprecated, no removal planned | Same behavior as record 8 | `test_mppi.py`; snapshot B |
| 10 | `actions` | `CEMPlanResult.selected_actions` | Pre-Sprint78 implementation; retained in `0.9.0` and 1.0.0 candidate; not deprecated, no removal planned | Read-only property returns the exact actions array; property access has no warning seam | `test_cem.py`; snapshot B |
| 11 | `actions` | `MPPIPlanResult.selected_actions` | Pre-Sprint78 implementation; retained in `0.9.0` and 1.0.0 candidate; not deprecated, no removal planned | Same exact-array and immutable-result guarantee as record 10 | `test_mppi.py`; snapshot B |
| 12 | `trajectory` | `RolloutResult.states` | Pre-Sprint78 implementation; retained in `0.9.0` and 1.0.0 candidate; not deprecated, no removal planned | Read-only property returns the same trajectory; no warning seam | `test_mppi_rollout.py`; snapshot B |
| 13 | `step` | `DeterministicLatentTransition.predict` | Pre-Sprint78 implementation; retained in `0.9.0` and 1.0.0 candidate; not deprecated, no removal planned | Predict delegates to step with identical numeric/error behavior; wrapper identity is not promised | `test_transition.py`; snapshot B |
| 14 | `step` | `StochasticGaussianLatentTransition.predict` | Pre-Sprint78 implementation; retained in `0.9.0` and 1.0.0 candidate; not deprecated, no removal planned | Predict returns the Gaussian prediction while step returns its mean, preserving the documented type boundary; no warning | `test_transition.py`; snapshot B |
| 15 | `scale` | `GaussianPrediction.std` | Pre-Sprint78 implementation; retained in `0.9.0` and 1.0.0 candidate; not deprecated, no removal planned | Exact read-only array alias; no warning seam | `test_transition.py`; snapshot B |
| 16 | `negative_log_likelihood` | `StochasticOneStepMetrics.nll` | Pre-Sprint78 implementation; retained in `0.9.0` and 1.0.0 candidate; not deprecated, no removal planned | Exact scalar property alias; no warning seam | `test_transition.py`; snapshot B |
| 17 | `negative_log_likelihood_by_horizon` | `StochasticRolloutMetrics.nll_by_horizon` | Pre-Sprint78 implementation; retained in `0.9.0` and 1.0.0 candidate; not deprecated, no removal planned | Exact scalar property alias; no warning seam | `test_transition.py`; snapshot B |
| 18 | `mean_error_by_horizon` | `StochasticRolloutMetrics.errors_by_horizon` | Pre-Sprint78 implementation; retained in `0.9.0` and 1.0.0 candidate; not deprecated, no removal planned | Exact tuple property alias; no warning seam | `test_transition.py`; snapshot B |

## Schema and path migrations

These are data migrations, not public spelling aliases. They have no removal
deadline while supported schema readers remain compatible and emit no warning:

| Legacy data | Canonical data | Boundary and guarantee | Verification |
|---|---|---|---|
| `result-envelope-v0` | `result-envelope-v1` | `decode_result_envelope` reconstructs the missing identity only for the explicit allowlisted migration | `test_portable_results.py`; snapshot J |
| Pre-versioned run records and Windows artifact paths | run-record `schema-v1` | `migrate_run_record` canonicalizes the record and relative artifact paths before validation | `test_run_record.py`; snapshot J |

## Policy and review rule

The 18 aliases above remain in the published `0.9.0` baseline and in the
unpublished `1.0.0` candidate source. The RFC0001 `0.2.0` window was planned but
never released, so it is not an historical `since` or deprecation version.
Within `1.x`, deprecated aliases continue to work with their documented
warnings; no removal is scheduled. Removing one is a breaking change and
requires a separately reviewed major-version migration. New aliases must be
added here, to snapshot B, and to an identity/behavior test in the same change.
The canonical public surface and retained beta aliases remain separately
countable.
