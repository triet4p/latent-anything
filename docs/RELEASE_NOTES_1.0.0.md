# Draft 1.0.0 Release Notes

**Status: unpublished candidate.** This page is a draft, not a release announcement. The repository's working-tree metadata targets `1.0.0`, but no `1.0.0` tag, release wheel, source distribution, GitHub Release, or PyPI publication was produced for this candidate. `0.9.0` remains the latest published package.

## Supported diagnostic scope

The candidate's accepted diagnostic claim is deliberately narrow and is based on the final Sprint 80 evidence review, which returned **PASS** with no actionable findings at `8392b04`:

- Prospective encoder v3 model-weight lesion on the frozen four-dimensional linear-autoencoder / digits brightness-bin case.
- Transformer target-evidence-v2 case for protocol-bounded section-header separability and localization at `transformer.h.0`.
- A separately gated transformer stability supplement; it does not retroactively change the immutable v2 artifact's `threshold: null` value.

The evidence, review references, and precise boundaries are recorded in the [Sprint 80 depth-evidence report](SPRINT_80_DEPTH_EVIDENCE.md). This is not a claim of arbitrary-model or dataset coverage, broad VLA support, model quality, deployment readiness, or GPU/CUDA readiness.

## Compatibility and migration

- The candidate API snapshot records 214 runtime exports, 211 canonical-stable entries, 28 config schemas, 89 public dataclass/result schemas, and 8 public exceptions. Its digest is `46ef4edd2bcb5d6235a02633caba10408b6e2000d5614928bef0794b20b005f4`; see [`api_freeze_snapshot_1.0.0.json`](../artifacts/api_freeze_snapshot_1.0.0.json). This describes candidate source, not a released API contract.
- The [compatibility ledger](API_COMPATIBILITY.md) records all 18 aliases retained in the candidate. Deprecated aliases remain available through the `1.x` line; no alias removal is scheduled. Any removal requires a separately reviewed major-version migration.
- The [migration guide](MIGRATION.md) documents the explicit result-envelope and run-record migrations. Unknown serialized schema versions fail closed; the candidate does not promise arbitrary cross-version conversion.
- The [plugin author guide](PLUGIN_AUTHOR_GUIDE.md) and [template](PLUGIN_TEMPLATE.md) describe the version-1 plugin contract. Their `1.x` compatibility promise applies only after a 1.0.0 release passes its gates.

The [support policy](SUPPORT_POLICY.md) is the single user-facing reference
for the conditional 1.x SemVer, deprecation/removal, security reporting,
upstream compatibility, and versioned-artifact migration commitments.

## Explicitly blocked and excluded claims

SmolVLA remains **BLOCKED** as a secondary, non-gating lane. The pinned run captured `(64, 768)` where the frozen v1 proof asserted `(64, 1024)`; the estimator also cannot support 768 dimensions from 64 samples. Execution stopped before diagnosis, no validator-clean artifact exists, and the prospective 768-to-32 v2 manifest remains unexecuted. This does not establish SmolVLA, general VLA, GPU, or CUDA support.

The theory-row percentages and former breadth thresholds are portfolio-coverage facts, not 1.0.0 release gates or diagnostic-quality measures; see the [theory evidence-gap plan](EVIDENCE_GAP_PLAN.md).

## Publication status

The stable-tag ruleset is active and verified. The owner-managed App token proof and PyPI Trusted Publisher first-upload bootstrap remain pending. Sprint 81 Task 4 completed the clean wheel/sdist build and clean base/optional-extra installation checks. Other release-quality and workflow gates remain separate from the Sprint 80 depth PASS. Do not tag or publish this candidate until the audited release process verifies every prerequisite and gate.
