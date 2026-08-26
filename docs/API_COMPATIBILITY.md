# API compatibility and deprecation ledger

This English ledger is the current user-facing source for beta aliases. It is
consistent with [RFC0001](rfcs/0001-semantic-api-vocabulary.md), the checked-in
[API-freeze snapshot](../artifacts/api_freeze_snapshot_0.1.0b1.json), and the
`0.9.0` compatibility-epoch decision. RFC0001 planned a `0.2.0` vocabulary
window, but that release was never published. The canonical symbols were added
Unreleased in Sprint 78.29, while the canonical registry kinds landed in Sprint
31; both remain under package metadata `0.1.0b1`. No alias is removed by this
ledger.

## Alias records

| # | Canonical spelling | Legacy spelling / path | RFC planned window; actual implementation/release; deprecation/removal | Warning and guarantee | Verification |
|---:|---|---|---|---|---|
| 1 | `AnalysisMethod` | `Method` (`latent_anything`, `methods.protocols`) | Planned RFC `0.2.0` (never released); implemented Unreleased / Sprint78.29 (`0.1.0b1` metadata); deprecation notice current Unreleased / Sprint78.31 (warning impractical), removal planned `0.9.0` | Exact object identity; no import warning because import-time warnings would punish every consumer | `test_api_surface.py`; snapshot B |
| 2 | `Intervention` | `BMethod` (`methods.b_protocols`) | Planned RFC `0.2.0` (never released); implemented Unreleased / Sprint78.29 (`0.1.0b1` metadata); deprecation notice current Unreleased / Sprint78.31 (warning impractical), removal planned `0.9.0` | Exact Protocol identity; no import warning for the same reason | `test_api_surface.py`; snapshot B |
| 3 | `InterventionPipeline` | `ManipulationPipeline` (`latent_anything`, `manipulation_pipeline`) | Planned RFC `0.2.0` (never released); implemented Unreleased / Sprint78.29 (`0.1.0b1` metadata); deprecation notice current Unreleased / Sprint78.31 (warning impractical), removal planned `0.9.0` | Exact class identity and behavior; no import warning | `test_api_surface.py`; snapshot B |
| 4 | `analysis` | `KIND_METHOD_A` / `method_a` registry kind | Planned RFC `0.2.0` (never released); implemented Unreleased / Sprint31 (`0.1.0b1` metadata); deprecated current Unreleased / Sprint31 with construction warning, removal planned `0.9.0` | `method_a` normalizes once at config construction and emits one `DeprecationWarning`; registry lookup itself stays quiet | `test_registry_migration.py`; snapshot B |
| 5 | `intervention` | `KIND_METHOD_B` / `method_b` registry kind | Planned RFC `0.2.0` (never released); implemented Unreleased / Sprint31 (`0.1.0b1` metadata); deprecated current Unreleased / Sprint31 with construction warning, removal planned `0.9.0` | Same one-warning construction-boundary rule as record 4 | `test_registry_migration.py`; snapshot B |
| 6 | `capture-points` | CLI `list-capture-points` | Pre-Sprint78 implementation; current metadata `0.1.0b1`; not deprecated, no removal planned | Same argparse parser, exit code, and JSON output; warning is impractical for a command alias | `test_cli.py`; snapshot B/I |
| 7 | `replay-run` | CLI `replay-run-config` | Pre-Sprint78 implementation; current metadata `0.1.0b1`; not deprecated, no removal planned | Same argparse parser, dispatch, exit code, and failure message; warning is impractical for a command alias | `test_cli.py`; snapshot B/I |
| 8 | `temperature` | `MPPIConfig(lambda=...)` | Pre-Sprint78 implementation; current metadata `0.1.0b1`; not deprecated, no removal planned | Pydantic validation produces the same temperature; no warning because field aliases are validated before a construction warning seam | `test_mppi.py`; snapshot B |
| 9 | `temperature` | `MPPIConfig(lambda_=...)` | Pre-Sprint78 implementation; current metadata `0.1.0b1`; not deprecated, no removal planned | Same behavior as record 8 | `test_mppi.py`; snapshot B |
| 10 | `actions` | `CEMPlanResult.selected_actions` | Pre-Sprint78 implementation; current metadata `0.1.0b1`; not deprecated, no removal planned | Read-only property returns the exact actions array; property access has no warning seam | `test_cem.py`; snapshot B |
| 11 | `actions` | `MPPIPlanResult.selected_actions` | Pre-Sprint78 implementation; current metadata `0.1.0b1`; not deprecated, no removal planned | Same exact-array and immutable-result guarantee as record 10 | `test_mppi.py`; snapshot B |
| 12 | `trajectory` | `RolloutResult.states` | Pre-Sprint78 implementation; current metadata `0.1.0b1`; not deprecated, no removal planned | Read-only property returns the same trajectory; no warning seam | `test_mppi_rollout.py`; snapshot B |
| 13 | `step` | `DeterministicLatentTransition.predict` | Pre-Sprint78 implementation; current metadata `0.1.0b1`; not deprecated, no removal planned | Predict delegates to step with identical numeric/error behavior; wrapper identity is not promised | `test_transition.py`; snapshot B |
| 14 | `step` | `StochasticGaussianLatentTransition.predict` | Pre-Sprint78 implementation; current metadata `0.1.0b1`; not deprecated, no removal planned | Predict returns the Gaussian prediction while step returns its mean, preserving the documented type boundary; no warning | `test_transition.py`; snapshot B |
| 15 | `scale` | `GaussianPrediction.std` | Pre-Sprint78 implementation; current metadata `0.1.0b1`; not deprecated, no removal planned | Exact read-only array alias; no warning seam | `test_transition.py`; snapshot B |
| 16 | `negative_log_likelihood` | `StochasticOneStepMetrics.nll` | Pre-Sprint78 implementation; current metadata `0.1.0b1`; not deprecated, no removal planned | Exact scalar property alias; no warning seam | `test_transition.py`; snapshot B |
| 17 | `negative_log_likelihood_by_horizon` | `StochasticRolloutMetrics.nll_by_horizon` | Pre-Sprint78 implementation; current metadata `0.1.0b1`; not deprecated, no removal planned | Exact tuple property alias; no warning seam | `test_transition.py`; snapshot B |
| 18 | `mean_error_by_horizon` | `StochasticRolloutMetrics.errors_by_horizon` | Pre-Sprint78 implementation; current metadata `0.1.0b1`; not deprecated, no removal planned | Exact tuple property alias; no warning seam | `test_transition.py`; snapshot B |

## Schema and path migrations

These are data migrations, not public spelling aliases. They have no removal
deadline while supported schema readers remain compatible and emit no warning:

| Legacy data | Canonical data | Boundary and guarantee | Verification |
|---|---|---|---|
| `result-envelope-v0` | `result-envelope-v1` | `decode_result_envelope` reconstructs the missing identity only for the explicit allowlisted migration | `test_portable_results.py`; snapshot J |
| Pre-versioned run records and Windows artifact paths | run-record `schema-v1` | `migrate_run_record` canonicalizes the record and relative artifact paths before validation | `test_run_record.py`; snapshot J |

## Policy and review rule

The aliases above remain available through the beta window. The RFC0001
`0.2.0` window was planned but never released, so its date is not an historical
`since` or deprecation release. At the planned `0.9.0` compatibility epoch,
removal requires a separate reviewed migration decision and repository-owned
configuration report; this document does not authorize removal. New aliases
must be added here, to snapshot B, and to an identity/behavior test in the same
change. The canonical public surface and the current beta compatibility surface
must remain separately countable.
