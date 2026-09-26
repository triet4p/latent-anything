"""Behavioral checks for the depth-first release-readiness preflight."""

from __future__ import annotations

from scripts.check_release_readiness import release_readiness_blockers


def _ready_external_prerequisites() -> str:
    return """{
      "schema": "latent-anything-release-prerequisites-v1",
      "external_prerequisites": {
        "release-tag-ruleset": {
          "status": "ready",
          "evidence": "Authenticated active stable-tag ruleset."
        },
        "release-app-token-proof": {
          "status": "ready",
          "evidence": "Nonpublishing default-branch token scope was verified."
        },
        "pypi-trusted-publisher": {
          "status": "pending",
          "configuration_status": "verified",
          "configuration": {
            "project_name": "latent-anything",
            "repository_owner": "triet4p",
            "repository_name": "latent-anything",
            "workflow_filename": "release.yml",
            "environment": "pypi"
          },
          "project_json_http_status": 404,
          "configuration_evidence": "The owner-provided publisher configuration matches this project."
        }
      }
    }"""


def _ready_plan() -> str:
    return (
        "## Atomic Tasks\n\n"
        "- [x] Confirm the Sprint 80 diagnostic-depth gate has no unresolved "
        "release blocker and every supported 1.0 claim is signed off in its "
        "evidence report.\n"
        "- [x] Enforce the depth-first stop-before-release contract: no tag or "
        "publish while any supported capture, detection, localization, "
        "statistical-control, explanation, causal-validation, reporting, "
        "packaging, documentation, or workflow blocker remains.\n"
        "- [x] Finalize version metadata, changelog, release notes, API "
        "reference, migration guide, plugin SDK, model/LeRobot guides, and "
        "theory coverage report.\n"
        "- [x] Build wheel/sdist from a clean checkout and verify "
        "install/import/examples in clean base and optional-extra environments.\n"
        "- [ ] Publish signed/checksummed package artifacts and the stable "
        "GitHub/PyPI release through the audited workflow.\n"
        "- [ ] Tag versioned documentation and archive benchmark/model/dataset "
        "revision manifests without redistributing restricted weights/data.\n"
        "- [ ] Verify post-publication install, links, plugin discovery, and "
        "one lightweight end-to-end example.\n"
        "- [x] State semantic-versioning, deprecation, security-reporting, "
        "upstream-compatibility, and artifact-migration policies.\n"
        "- [ ] Mark completed sprints/milestones, publish the final artifact, "
        "and open the evidence-led post-1.0 backlog.\n\n"
        "## Notes / Blockers\n"
    )


def _ready_depth_report() -> str:
    return (
        "**Sprint 80 diagnostic-depth gate:** **PASS — no unresolved blocker "
        "for the bounded supported ordinary-DL core.** This is not "
        "`1.0.0` publication approval.\n\n"
        "## Stable-depth gate interpretation and excluded claims\n\n"
        "| Gate for the bounded supported core | Evidence-based disposition |\n"
        "|---|---|\n"
        "| Capture and provenance | PASS for the accepted cases. |\n"
        "| Detection and controls | PASS for the accepted cases. |\n"
        "| Localization | PASS for the accepted cases. |\n"
        "| Explanation | PASS for the accepted cases. |\n"
        "| Causal evidence and comparison | PASS for the accepted cases. |\n"
        "| Persistence and reproducibility | PASS for the accepted cases. |\n"
        "| Usability | PASS for the accepted cases. |\n"
        "| Quality/compatibility | Scoped PASS evidence only. |\n\n"
        "SmolVLA remains explicitly blocked and non-gating.\n\n"
        "The accepted bounded result records a deterministic rendered report.\n"
    )


def test_pending_release_app_token_proof_blocks_preflight() -> None:
    release_gates = _ready_external_prerequisites().replace(
        '"release-app-token-proof": {\n          "status": "ready"',
        '"release-app-token-proof": {\n          "status": "pending"',
        1,
    )

    blockers = release_readiness_blockers(_ready_plan(), _ready_depth_report(), release_gates)

    assert blockers == ["External release prerequisite is not ready: release-app-token-proof (status: pending)"]


def test_missing_release_app_token_proof_blocks_preflight() -> None:
    release_gates = _ready_external_prerequisites().replace(
        '        "release-app-token-proof": {\n'
        '          "status": "ready",\n'
        '          "evidence": "Nonpublishing default-branch token scope was verified."\n'
        "        },\n",
        "",
        1,
    )

    blockers = release_readiness_blockers(_ready_plan(), _ready_depth_report(), release_gates)

    assert blockers == ["Required external prerequisite is missing: release-app-token-proof"]


def test_open_package_gate_blocks_tag_and_publication_preflight() -> None:
    plan = _ready_plan().replace(
        "- [x] Build wheel/sdist from a clean checkout",
        "- [ ] Build wheel/sdist from a clean checkout",
        1,
    )

    blockers = release_readiness_blockers(plan, _ready_depth_report(), _ready_external_prerequisites())

    assert blockers == [
        "Sprint 81 task 4 is not complete: "
        "Build wheel/sdist from a clean checkout and verify "
        "install/import/examples in clean base and optional-extra environments."
    ]


def test_reordered_plan_matches_required_tasks_by_title_not_position() -> None:
    state_line = (
        "- [x] State semantic-versioning, deprecation, security-reporting, "
        "upstream-compatibility, and artifact-migration policies.\n"
    )
    publish_line = (
        "- [ ] Publish signed/checksummed package artifacts and the stable "
        "GitHub/PyPI release through the audited workflow.\n"
    )
    plan = _ready_plan().replace(state_line, "", 1).replace(publish_line, state_line + publish_line, 1)

    blockers = release_readiness_blockers(plan, _ready_depth_report(), _ready_external_prerequisites())

    assert blockers == []


def test_supported_depth_failure_blocks_even_when_release_tasks_are_complete() -> None:
    report = _ready_depth_report().replace(
        "| Capture and provenance | PASS",
        "| Capture and provenance | BLOCKED",
        1,
    )

    blockers = release_readiness_blockers(_ready_plan(), report, _ready_external_prerequisites())

    assert blockers == ["Supported bounded-core depth gate is not PASS: Capture and provenance"]


def test_verified_pending_publisher_allows_first_upload_with_expected_project_404() -> None:
    blockers = release_readiness_blockers(_ready_plan(), _ready_depth_report(), _ready_external_prerequisites())

    assert blockers == []


def test_pending_publisher_without_verified_configuration_blocks_preflight() -> None:
    release_gates = _ready_external_prerequisites().replace(
        '"configuration_status": "verified"',
        '"configuration_status": "unverified"',
        1,
    )

    blockers = release_readiness_blockers(_ready_plan(), _ready_depth_report(), release_gates)

    assert blockers == ["PyPI Trusted Publisher configuration is not verified"]


def test_pending_publisher_with_mismatched_repository_blocks_preflight() -> None:
    release_gates = _ready_external_prerequisites().replace(
        '"repository_name": "latent-anything"',
        '"repository_name": "other-project"',
        1,
    )

    blockers = release_readiness_blockers(_ready_plan(), _ready_depth_report(), release_gates)

    assert blockers == ["PyPI Trusted Publisher configuration mismatch: repository_name must be 'latent-anything'"]


def test_pending_publisher_without_configuration_blocks_preflight() -> None:
    release_gates = _ready_external_prerequisites().replace(
        '"configuration": {',
        '"publisher_configuration": {',
        1,
    )

    blockers = release_readiness_blockers(_ready_plan(), _ready_depth_report(), release_gates)

    assert blockers == ["PyPI Trusted Publisher configuration is missing"]


def test_activated_publisher_requires_post_upload_project_and_activation_evidence() -> None:
    release_gates = _ready_external_prerequisites().replace(
        '"status": "pending"',
        '"status": "active"',
        1,
    )

    blockers = release_readiness_blockers(_ready_plan(), _ready_depth_report(), release_gates)

    assert blockers == [
        "Active PyPI Trusted Publisher requires project JSON HTTP 200 (observed: 404)",
        "Active PyPI Trusted Publisher has no post-upload activation evidence",
    ]


def test_activated_publisher_passes_with_post_upload_project_evidence() -> None:
    release_gates = (
        _ready_external_prerequisites()
        .replace(
            '"status": "pending"',
            '"status": "active"',
            1,
        )
        .replace(
            '"project_json_http_status": 404,',
            '"project_json_http_status": 200,\n'
            '          "activation_evidence": '
            '"PyPI project publisher activation was verified after the first upload.",',
            1,
        )
    )

    blockers = release_readiness_blockers(_ready_plan(), _ready_depth_report(), release_gates)

    assert blockers == []


def test_pending_publisher_with_non_404_project_response_blocks_preflight() -> None:
    release_gates = _ready_external_prerequisites().replace(
        '"project_json_http_status": 404',
        '"project_json_http_status": 503',
        1,
    )

    blockers = release_readiness_blockers(_ready_plan(), _ready_depth_report(), release_gates)

    assert blockers == ["Pending PyPI Trusted Publisher requires the pre-upload project JSON HTTP 404 (observed: 503)"]


def test_unverified_tag_ruleset_blocks_tag_preflight() -> None:
    release_gates = _ready_external_prerequisites().replace(
        '"release-tag-ruleset": {\n          "status": "ready"',
        '"release-tag-ruleset": {\n          "status": "pending"',
        1,
    )

    blockers = release_readiness_blockers(_ready_plan(), _ready_depth_report(), release_gates)

    assert blockers == ["External release prerequisite is not ready: release-tag-ruleset (status: pending)"]


def test_missing_supported_reporting_evidence_blocks_release_preflight() -> None:
    report = _ready_depth_report().replace("deterministic rendered report", "persisted output", 1)

    blockers = release_readiness_blockers(_ready_plan(), report, _ready_external_prerequisites())

    assert blockers == ["Supported bounded-core reporting evidence is missing"]
