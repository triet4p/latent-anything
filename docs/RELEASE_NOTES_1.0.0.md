# 1.0.0 Release Notes

**Status: published.** The protected annotated tag `v1.0.0` was published on 2026-09-26 from release commit `a449ca33b4b83ce109c29db7cf471919820b1d56`. Audited workflow run [36251907256](https://github.com/triet4p/latent-anything/actions/runs/36251907256) published the GitHub Release and first PyPI distributions through OIDC. PyPI project JSON returns HTTP 200 for version `1.0.0`, with wheel and sdist digests matching the GitHub Release assets. This documentation reconciliation is later than the tagged release commit and does not change its source or assets.

## Supported diagnostic scope

The released diagnostic claim is deliberately narrow and is based on the final Sprint 80 evidence review, which returned **PASS** with no actionable findings at `8392b04`:

- Prospective encoder v3 model-weight lesion on the frozen four-dimensional linear-autoencoder / digits brightness-bin case.
- Transformer target-evidence-v2 case for protocol-bounded section-header separability and localization at `transformer.h.0`.
- A separately gated transformer stability supplement; it does not retroactively change the immutable v2 artifact's `threshold: null` value.

The evidence, review references, and precise boundaries are recorded in the [Sprint 80 depth-evidence report](SPRINT_80_DEPTH_EVIDENCE.md). This is not a claim of arbitrary-model or dataset coverage, broad VLA support, model quality, deployment readiness, or GPU/CUDA readiness.

## Compatibility and migration

- The published `1.0.0` API snapshot at release commit `a449ca3` records 214 runtime exports, 211 canonical-stable entries, 28 config schemas, 89 public dataclass/result schemas, and 8 public exceptions. Its digest is `46ef4edd2bcb5d6235a02633caba10408b6e2000d5614928bef0794b20b005f4`; see [`api_freeze_snapshot_1.0.0.json`](../artifacts/api_freeze_snapshot_1.0.0.json).
- The [compatibility ledger](API_COMPATIBILITY.md) records all 18 aliases retained in the published release. Deprecated aliases remain available through the `1.x` line; no alias removal is scheduled. Any removal requires a separately reviewed major-version migration.
- The [migration guide](MIGRATION.md) documents the explicit result-envelope and run-record migrations. Unknown serialized schema versions fail closed; the released API does not promise arbitrary cross-version conversion.
- The [plugin author guide](PLUGIN_AUTHOR_GUIDE.md) and [template](PLUGIN_TEMPLATE.md) describe the version-1 plugin contract and its 1.x compatibility commitment.

The [support policy](SUPPORT_POLICY.md) is the user-facing reference for the
1.x SemVer, deprecation/removal, security reporting, upstream compatibility,
and versioned-artifact migration commitments.

## Explicitly blocked and excluded claims

SmolVLA remains **BLOCKED** as a secondary, non-gating lane. The pinned run captured `(64, 768)` where the frozen v1 proof asserted `(64, 1024)`; the estimator also cannot support 768 dimensions from 64 samples. Execution stopped before diagnosis, no validator-clean artifact exists, and the prospective 768-to-32 v2 manifest remains unexecuted. This does not establish SmolVLA, general VLA, GPU, or CUDA support.

The theory-row percentages and former breadth thresholds are portfolio-coverage facts, not 1.0.0 release gates or diagnostic-quality measures; see the [theory evidence-gap plan](EVIDENCE_GAP_PLAN.md).

## Publication status

Exact-SHA CI run [36250365395](https://github.com/triet4p/latent-anything/actions/runs/36250365395) passed on release commit `a449ca33b4b83ce109c29db7cf471919820b1d56` for the Python 3.12/3.13/3.14 matrix. Audited release run [36251907256](https://github.com/triet4p/latent-anything/actions/runs/36251907256) completed successfully across all five jobs, including artifact attestations, protected tag creation, GitHub Release publication, and the PyPI OIDC upload after owner approval of environment `pypi` (`22803163260`).

The annotated tag object `3bdb7ff2c5e09def32e39b9b4628750729104b22` points to the release commit. The GitHub Release has five assets, including wheel SHA-256 `3f7081d4cb8994c6a719d85d51a3c0ccff76171673c5a1dc33737d7770b408ea` and sdist SHA-256 `36b6b3950fd791cf9ffa5eeac79050c5052d5b9902d1c98a9a2b7a6d38e8e3a9`. `PROVENANCE.json` binds the tag, commit, and workflow run; GitHub attestations verified for both distributions.

The active PyPI project JSON endpoint returns HTTP 200 for version `1.0.0`; both distribution filenames and digests match the GitHub Release assets. The successful first OIDC upload in run `36251907256` and matching live project metadata are the publisher activation evidence. The release is published; no additional release action is part of this documentation update.
