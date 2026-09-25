"""Behavioral checks for the depth-first release-readiness preflight."""

from __future__ import annotations

from scripts.check_release_readiness import release_readiness_blockers


def _ready_external_prerequisites() -> str:
    return """{
      "schema": "latent-anything-release-prerequisites-v1",
      "external_prerequisites": {
        "release-tag-ruleset": {
          "status": "ready",
          "evidence": "Verified rule permits audited release workflow tag writes only."
        },
        "pypi-trusted-publisher": {
          "status": "ready",
          "environment": "pypi",
          "evidence": "Verified owner-controlled PyPI Trusted Publisher activation."
        }
      }
    }"""


def _ready_plan() -> str:
    return """## Atomic Tasks

- [x] Confirm the Sprint 80 diagnostic-depth gate has no unresolved release blocker and every supported 1.0 claim is signed off in its evidence report.
- [x] Enforce the depth-first stop-before-release contract: no tag or publish while any supported capture, detection, localization, statistical-control, explanation, causal-validation, reporting, packaging, documentation, or workflow blocker remains.
- [x] Finalize version metadata, changelog, release notes, API reference, migration guide, plugin SDK, model/LeRobot guides, and theory coverage report.
- [x] Build wheel/sdist from a clean checkout and verify install/import/examples in clean base and optional-extra environments.
- [ ] Publish signed/checksummed package artifacts and the stable GitHub/PyPI release through the audited workflow.
- [ ] Tag versioned documentation and archive benchmark/model/dataset revision manifests without redistributing restricted weights/data.
- [ ] Verify post-publication install, links, plugin discovery, and one lightweight end-to-end example.
- [x] State semantic-versioning, deprecation, security-reporting, upstream-compatibility, and artifact-migration policies.
- [ ] Mark completed sprints/milestones, publish the final artifact, and open the evidence-led post-1.0 backlog.

## Notes / Blockers
"""


def _ready_depth_report() -> str:
    return """**Sprint 80 diagnostic-depth gate:** **PASS — no unresolved blocker for the bounded supported ordinary-DL core.** This is not `1.0.0` publication approval.

## Stable-depth gate interpretation and excluded claims

| Gate for the bounded supported core | Evidence-based disposition |
|---|---|
| Capture and provenance | PASS for the accepted cases. |
| Detection and controls | PASS for the accepted cases. |
| Localization | PASS for the accepted cases. |
| Explanation | PASS for the accepted cases. |
| Causal evidence and comparison | PASS for the accepted cases. |
| Persistence and reproducibility | PASS for the accepted cases. |
| Usability | PASS for the accepted cases. |
| Quality/compatibility | Scoped PASS evidence only. |

SmolVLA remains explicitly blocked and non-gating.

The accepted bounded result records a deterministic rendered report.
"""


def test_open_package_gate_blocks_tag_and_publication_preflight() -> None:
    plan = _ready_plan().replace(
        "- [x] Build wheel/sdist from a clean checkout",
        "- [ ] Build wheel/sdist from a clean checkout",
        1,
    )

    blockers = release_readiness_blockers(plan, _ready_depth_report(), _ready_external_prerequisites())

    assert blockers == [
        "Sprint 81 task 4 is not complete: Build wheel/sdist from a clean checkout and verify install/import/examples in clean base and optional-extra environments."
    ]


def test_supported_depth_failure_blocks_even_when_release_tasks_are_complete() -> None:
    report = _ready_depth_report().replace(
        "| Capture and provenance | PASS",
        "| Capture and provenance | BLOCKED",
        1,
    )

    blockers = release_readiness_blockers(_ready_plan(), report, _ready_external_prerequisites())

    assert blockers == ["Supported bounded-core depth gate is not PASS: Capture and provenance"]


def test_bounded_core_passes_preflight_with_smolvla_excluded_and_publication_pending() -> None:
    blockers = release_readiness_blockers(_ready_plan(), _ready_depth_report(), _ready_external_prerequisites())

    assert blockers == []


def test_pending_pypi_publisher_blocks_tag_preflight() -> None:
    release_gates = _ready_external_prerequisites().replace(
        '"pypi-trusted-publisher": {\n          "status": "ready"',
        '"pypi-trusted-publisher": {\n          "status": "pending"',
        1,
    )

    blockers = release_readiness_blockers(_ready_plan(), _ready_depth_report(), release_gates)

    assert blockers == [
        "External release prerequisite is not ready: pypi-trusted-publisher (status: pending)"
    ]


def test_unverified_tag_ruleset_blocks_tag_preflight() -> None:
    release_gates = _ready_external_prerequisites().replace(
        '"release-tag-ruleset": {\n          "status": "ready"',
        '"release-tag-ruleset": {\n          "status": "pending"',
        1,
    )

    blockers = release_readiness_blockers(_ready_plan(), _ready_depth_report(), release_gates)

    assert blockers == [
        "External release prerequisite is not ready: release-tag-ruleset (status: pending)"
    ]


def test_missing_supported_reporting_evidence_blocks_release_preflight() -> None:
    report = _ready_depth_report().replace("deterministic rendered report", "persisted output", 1)

    blockers = release_readiness_blockers(_ready_plan(), report, _ready_external_prerequisites())

    assert blockers == ["Supported bounded-core reporting evidence is missing"]
