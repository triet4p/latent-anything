# Migration and compatibility guide

The package version `1.0.0` was published on 2026-09-26. Protected annotated
tag `v1.0.0` points to release commit
`a449ca33b4b83ce109c29db7cf471919820b1d56`; this guide is a post-release
documentation reconciliation. `0.9.0` remains the prior pre-stable release.

The annotated `v0.9.0` tag points to merged commit
`75341e4292fcdf1703186c58b626666b4a923c19`, and Release workflow run
`35516933525` passed gate/build and publish. The GitHub Release carries the
wheel, source distribution, `SHA256SUMS`, `PROVENANCE.json`, and release notes;
all five asset hashes and a clean local-wheel import smoke were verified.
PyPI publication was explicitly deferred for `0.9.0`. The released `1.0.0`
API snapshot is [`api_freeze_snapshot_1.0.0.json`](../artifacts/api_freeze_snapshot_1.0.0.json).
The `0.9.0`-labeled source snapshot is a later pre-candidate inventory, not
the published tag's release-time snapshot. The exact `v0.9.0` release-time
inventory is recorded in the historical `CHANGELOG.md` section; the historical
beta surface is preserved unchanged in
[`api_freeze_snapshot_0.1.0b1.json`](../artifacts/api_freeze_snapshot_0.1.0b1.json).
The human alias policy is [`API_COMPATIBILITY.md`](API_COMPATIBILITY.md).
The versioned [1.0.0 documentation and input-revision record](VERSIONED_DOCS_1.0.0.md) binds signed-off claims to exact release, benchmark, model, and dataset revisions. Its metadata-only [revision archive](../artifacts/benchmark_revision_archive_1.0.0.json) contains no model weights or dataset bytes.

## Installing the 0.9.0 release

Download an asset from the
[GitHub Release](https://github.com/triet4p/latent-anything/releases/tag/v0.9.0),
verify it against `SHA256SUMS`, and install the local file:

```bash
uv pip install ./latent_anything-0.9.0-py3-none-any.whl
# or:
uv pip install ./latent_anything-0.9.0.tar.gz
```

Do not resolve this version from PyPI: publication is deferred and intentionally
non-gating for this release.

## From the published beta

The published `0.1.0-beta.1` core beta remains supported. Existing imports,
configuration files, result properties, CLI spellings, and local serialized
data remain readable during the beta window. New code should use the canonical
spellings below. This is an additive migration; this document does not remove
an alias or rewrite historical release artifacts.

The published `1.0.0` release snapshot records **214 runtime top-level
exports**, **211 canonical-stable entries**, **28 config schemas**, **89
public dataclass/result schemas**, and **8 public exceptions** (SHA-256
`46ef4edd2bcb5d6235a02633caba10408b6e2000d5614928bef0794b20b005f4`). This
describes the released API contract at tag `v1.0.0`.

The later pre-candidate source snapshot named
[`api_freeze_snapshot_0.9.0.json`](../artifacts/api_freeze_snapshot_0.9.0.json)
records 214 runtime exports, 211 canonical-stable entries, and 8 public
exceptions (digest
`3fd8c73e6fd9fa9bae107b4be6727cc3e9c3c907c9b9d757fdae4c267d8691ec`). Its
`0.9.0` package label does not make it the release-time snapshot. At the
published `v0.9.0` release, the inventory was **205 runtime top-level
exports**, **202 canonical-stable entries**, and **7 public exceptions**
(digest `048ac553adabb11c24d3e1f4d86e0c6d469df6590064014c915d6a08c7d53c26`),
as preserved in the dated `0.9.0` section of the repository-root
`CHANGELOG.md`. Later source additions do not retroactively change the release
assets.

Relative to the published beta, the three additive canonical entries are
`AnalysisMethod`, `Intervention`, and `InterventionPipeline`; the legacy
`Method`, `BMethod`, and `ManipulationPipeline` identities remain available.
`BMethod` was not a beta top-level export, but remains available through its
public methods submodule and has an additive top-level canonical counterpart.

## Naming migration

| Canonical spelling | Legacy spelling/path | Current state | Planned policy |
|---|---|---|---|
| `AnalysisMethod` | `Method` (`latent_anything`, `latent_anything.methods.protocols`) | Added in repository Sprint 78.29 under metadata `0.1.0b1`; exact identity alias shipped in `0.9.0` and retained in the published `1.0.0` release | Retain through 1.x; any removal requires a separately reviewed major-version migration |
| `Intervention` | `BMethod` (`latent_anything.methods.b_protocols`) | Added in repository Sprint 78.29 under metadata `0.1.0b1`; exact runtime-checkable Protocol identity shipped in `0.9.0` and retained in the published `1.0.0` release | Retain through 1.x; any removal requires a separately reviewed major-version migration; no import-time warning |
| `InterventionPipeline` | `ManipulationPipeline` (`latent_anything`, `latent_anything.manipulation_pipeline`) | Added in repository Sprint 78.29 under metadata `0.1.0b1`; exact class identity/behavior shipped in `0.9.0` and retained in the published `1.0.0` release | Retain through 1.x; any removal requires a separately reviewed major-version migration |

The RFC0001 `0.2.0` window was planned but that release was never published.
It must not be used as a historical `since` or deprecation release. Registry
canonical kinds landed in Sprint 31, also under `0.1.0b1` metadata.

## Complete beta alias ledger

The following are the **18 human-ledger alias rows**, expanded from snapshot
section B (`lambda` and `lambda_` count as separate rows). Rows 1–3 are
identity aliases without import-time warnings; deprecated registry aliases
(rows 4–5) warn at config construction. Rows 6–18 remain supported, and no
alias removal is scheduled in `1.x`. The two schema/path data migrations below
are separate and are not included in this alias-row count.

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

Writers emit the current documented schema. Readers migrate only the named
legacy shapes above; they validate the canonical result and do not silently
rewrite the source artifact. The explicit current writer/reader matrix and
major-version boundary are in the [support policy](SUPPORT_POLICY.md).

## Versioned compatibility policy

The current support, Semantic Versioning, deprecation, upstream, and
artifact-reader commitments are in the [support policy](SUPPORT_POLICY.md).
They are in force for the published `1.x` line beginning with `1.0.0`.

For this migration guide:

1. `1.0.0` is the latest published distribution; protected tag `v1.0.0` points
   to release commit `a449ca33b4b83ce109c29db7cf471919820b1d56`.
2. The published release's canonical-stable public surface is the checked-in
   [`1.0.0` API snapshot](../artifacts/api_freeze_snapshot_1.0.0.json).
   Incompatible public API or behavior changes require a major-version release
   and a migration path.
3. All 18 beta aliases are retained in the published `1.0.0` release and
   throughout `1.x`; no alias removal is scheduled. Any removal requires a
   separately reviewed major-version migration.
4. The only current cross-version data migrations are
   `result-envelope-v0` to `result-envelope-v1`, and pre-versioned run records
   with legacy Windows artifact paths to run-record `schema-v1`. Unknown
   versions fail closed; there is no implicit pickle or general conversion.
5. Optional model/provider integrations are bounded by the dependency ranges
   and evidence boundaries in `pyproject.toml` and their integration guides.
   An optional extra or pinned model identifier does not establish model
   quality, general VLA, or GPU/CUDA support.
6. The audited `1.0.0` publication gates passed. Stop before any future
   stable publication if a supported diagnostic, packaging, documentation,
   compatibility, security, or workflow gate fails. Sprint 80 signs off only
   the bounded ordinary-DL core; SmolVLA remains **BLOCKED** and non-gating.
   Former 95% core / 90% overall theory-row thresholds are portfolio-health
   metrics, not release gates.

The [Sprint 80 evidence report](SPRINT_80_DEPTH_EVIDENCE.md) and
[M14 stop-before-release contract](M14_REAL_SYSTEM_VALIDATION.md) remain
authoritative for the release's bounded evidence scope. The published package
does not claim arbitrary-model coverage, general VLA support, or GPU/CUDA readiness.
