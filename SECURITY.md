# Security policy

See the [support and versioning policy](docs/SUPPORT_POLICY.md) for release support and the security-fix scope.

## Private reporting availability

GitHub's private vulnerability-reporting setting is **enabled** for this repository. An authenticated repository-admin API check on 2026-09-26 returned `{"enabled": true}` for `GET /repos/triet4p/latent-anything/private-vulnerability-reporting` (enablement via `PUT` returned HTTP 204). End-to-end usability by a non-maintainer reporter (**Security → Report a vulnerability**) remains unverified, so the project still cannot promise confidential intake or coordinated disclosure until that path is tested. This file does not itself accept private submissions.

Do not submit vulnerability details, exploit code, credentials, or private data through public Issues, Discussions, or pull requests. There is no substitute email address or private form documented here because none has been verified as a repository-controlled security channel. See the [support and versioning policy](docs/SUPPORT_POLICY.md) for the matching security-reporting status.

Until the non-maintainer reporting path is tested, the absence of a verified end-to-end channel remains a release-readiness finding; do not interpret this policy as evidence that `1.0.0` is released or authorized for publication.

## Report contents and handling after enablement

Once the private route is operational, include the affected published version or commit, relevant environment, security impact, and a minimal safe reproduction. Do not include real credentials or sensitive user data. Maintainers will assess reports and coordinate any fix and advisory disclosure through GitHub's private advisory workflow. No acknowledgement time, remediation deadline, or backport is promised. Security fixes target the latest patch of the current supported major line; see the [support policy](docs/SUPPORT_POLICY.md#supported-scope).
