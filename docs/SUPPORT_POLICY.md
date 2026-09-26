# Support, versioning, and migration policy

**Status:** This is the proposed user-facing policy for the unpublished `1.0.0` candidate. The stable `1.x` commitments below take effect only after the audited `1.0.0` release gates pass and a `1.0.0` distribution is actually published. Until then, `0.9.0` remains the latest published package; candidate metadata and documentation are not a released support promise.

## Supported scope

After `1.0.0` is released, compatibility commitments apply to the canonical-stable public API recorded in the [candidate API snapshot](../artifacts/api_freeze_snapshot_1.0.0.json), the documented plugin API, and the explicitly supported serialized formats below. The snapshot is the inventory; the [API reference](API_REFERENCE.md) and [compatibility ledger](API_COMPATIBILITY.md) explain its public boundaries and aliases. An export not marked canonical-stable, an internal module, or a provider SDK object is not made public API by appearing in source code.

The package declares Python `>=3.12,<3.15`; the CI workflow exercises Python 3.12, 3.13, and 3.14 on Ubuntu. Required dependency ranges, optional extras, and resolver conflicts are defined by [`pyproject.toml`](../pyproject.toml). These bounds do not mean that every dependency version or optional-extra combination has been tested: support is limited to the individual extras and combinations documented by the integration guides and CI/evidence. Resolver success alone does not establish compatibility. The repository's [`uv.lock`](../uv.lock) pins the project validation environment and is not a promise that downstream applications resolve identical transitive versions.

The 1.0 diagnostic claim is limited to the accepted ordinary-DL encoder and transformer cases and the separately gated transformer stability supplement in the [Sprint 80 evidence report](SPRINT_80_DEPTH_EVIDENCE.md). It does not promise arbitrary-model or arbitrary-dataset diagnosis. SmolVLA remains **BLOCKED** and non-gating; broad VLA support and GPU/CUDA readiness are excluded. Optional extras, a pinned model identifier, or an implementation seam do not independently establish model-quality or hardware support.

The project makes no fixed calendar end-of-life promise for a major line. Once a stable line is released, compatibility applies across that major line; bug-fix and security releases target the latest patch of the current major line. Backports to `0.9.0`, older patch releases, or earlier major lines are not promised. Users should upgrade to the latest published patch. Any later end-of-support decision must be stated in a published release notice; this candidate does not set an end date.

## Semantic versioning

The package version follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html) for stable releases:

| Version change | User-visible change allowed |
| --- | --- |
| `MAJOR` (`X.0.0`) | Incompatible public API, behavior, plugin-contract, or required serialized-reader changes. Publish a migration path with the release. |
| `MINOR` (`1.Y.0`) | Backward-compatible public additions and capabilities. Existing supported APIs, plugin contracts, and readable artifact versions remain usable. |
| `PATCH` (`1.Y.Z`) | Backward-compatible bug fixes and security fixes. Do not remove public APIs or supported readers in a patch. |

The `0.9.0` pre-1.0 release does not receive a retroactive `1.x` compatibility guarantee. The working-tree `1.0.0` candidate is not itself a stable release, and no version tag, package upload, or publication is authorized until every release gate and external prerequisite passes. The [release-gate record](release-gates.json), [Sprint 80 evidence](SPRINT_80_DEPTH_EVIDENCE.md), and [M14 stop-before-release contract](M14_REAL_SYSTEM_VALIDATION.md) remain authoritative for that decision.

## Deprecation and removal

A stable public API may be deprecated only when its replacement and migration are documented. Record the old and canonical spellings, replacement, warning behavior, and removal boundary in the [API compatibility ledger](API_COMPATIBILITY.md) and migration guide in the same change. Emit a deprecation warning at a meaningful use or construction boundary where one exists; do not warn merely because an import is evaluated. If no safe warning seam exists, document that limitation in the ledger.

No compatibility alias or deprecated API is scheduled for removal during `1.x`. The 18 aliases listed in the ledger remain available throughout the 1.x line; any proposed removal requires a separately reviewed major-version change and a published migration path. A future major version must not treat the never-published RFC0001 `0.2.0` window as a historical deprecation period.

## Security reporting and fixes

GitHub's private vulnerability-reporting setting is **enabled** for this repository. An authenticated repository-admin API check on 2026-09-26 returned `{"enabled": true}` for `GET /repos/triet4p/latent-anything/private-vulnerability-reporting` (enablement via `PUT` returned HTTP 204). Whether a non-maintainer reporter can use **Security → Report a vulnerability** end to end has not been verified, so this repository still makes no claim of usable confidential intake.

Do not put vulnerability details, exploit code, credentials, or private data in a public issue, discussion, or pull request. Do not send sensitive reports through this repository until non-maintainer access to **Security → Report a vulnerability** is verified; after verification, use that feature. There is no substitute email address or private form documented here. A `SECURITY.md` policy document does not itself accept private submissions.

When the private route is verified end to end, include the affected released version or commit, environment, impact, and a minimal safe reproduction. Do not include real credentials or sensitive user data. Maintainers will assess the report, coordinate a fix and advisory disclosure through GitHub's private advisory workflow, and publish a security fix for the latest supported patch when warranted. No response-time or remediation-date SLA is promised. See [`SECURITY.md`](../SECURITY.md) for the current channel status.

## Upstream and optional-integration compatibility

The supported upstream ranges are the bounded ranges declared in
[`pyproject.toml`](../pyproject.toml) and described by each integration guide.
Those constraints do not promise that every upstream version in a range has
been exercised, and they do not include a moving upstream default branch. For
example, LeRobot targets `>=0.6.0,<0.7.0`; its `main` branch and the resolver-
conflicting legacy profiles are outside that contract.

Before widening or changing an upstream range, maintainers must keep the range
bounded, run the lower-bound import/compatibility smoke required by
[`OPTIONAL_INTEGRATIONS.md`](OPTIONAL_INTEGRATIONS.md), and run focused tests
for the consumed upstream seams. Record the exact tested versions plus
immutable model, dataset, or backend revisions and their evidence in the
integration guide and release record. Do not describe an untested version as
validated. Where an integration has a runtime compatibility checker, out-of-
range versions must return its actionable unsupported-version diagnostic.
Update `uv.lock` for reproducible repository validation; it is not a
downstream dependency pin or compatibility guarantee.

An upstream revision pin establishes reproducibility for that named evidence run only. It does not establish general model quality, arbitrary-checkpoint support, broad VLA support, or GPU/CUDA readiness. The blocked and excluded lanes listed above remain outside the 1.0 claim unless separately re-evidenced and reviewed.

## Versioned artifact readers and writers

Writers emit the current explicit format version; readers accept only the current version and the legacy forms for which a named migration is implemented. A migration is local and explicit, validates the canonical output before use, and does not silently rewrite the source artifact. Unknown versions, malformed payloads, failed integrity checks, or unsupported types fail closed. There is no pickle fallback or general cross-version conversion promise.

| Format | Current writer | Supported legacy read/migration |
| --- | --- | --- |
| `portable-node-v1` | `encode_portable` writes v1. | No older portable-node conversion is promised. |
| `result-envelope-v1` | `encode_result_envelope` writes v1. | `decode_result_envelope` explicitly migrates the supported `result-envelope-v0` shape to v1, then applies the same allowlist and identity checks. Unknown versions fail closed. |
| `artifact-envelope-v1` | `ArtifactStore` writes v1 with canonical metadata, identity, size, and SHA-256 checksum. | No older artifact-envelope conversion is promised. |
| Run-record `schema-v1` | The run-record writer emits the canonical v1 record. | `migrate_run_record` accepts the documented pre-versioned record and legacy Windows artifact-path forms, canonicalizes them, and validates before use. Traversal, symlink, malformed-record, and checksum failures remain errors. |
| `disk-cache-v1` | `SQLiteDiskCache` stores framework artifacts through the portable envelope path and validated cache keys. | The disk cache is derived state, not an archival interchange format; no cache-version migration is promised. Recompute entries that cannot be read. |

The exact format fixtures and digests are recorded in section J of the [API snapshot](../artifacts/api_freeze_snapshot_1.0.0.json). Existing readers for the supported `result-envelope-v0` and pre-versioned run-record forms remain available throughout `1.x`; removing a reader is a breaking change and requires a major-version migration decision. For operation and security boundaries, see [Portable artifacts and disk cache](PORTABLE_ARTIFACTS.md) and [Migration and compatibility](MIGRATION.md).

This policy does not make successful deserialization proof that a model, checkpoint, dataset, or execution can be reproduced. Preserve the associated provenance and immutable revisions separately.
