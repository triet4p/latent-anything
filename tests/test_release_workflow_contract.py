"""Behavioral contract tests for the fail-closed stable-release workflow."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github" / "workflows" / "release.yml"
_RELEASE_GATES = _ROOT / "docs" / "release-gates.json"


def _workflow() -> dict[str, Any]:
    # Workflow nodes are dynamic YAML data, narrowed at this trusted config boundary.
    parsed = yaml.load(_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    if not isinstance(parsed, dict):
        raise AssertionError("release workflow must be a YAML mapping")
    return parsed


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [step for step in job["steps"] if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def _needs(job: dict[str, Any]) -> set[str]:
    needs = job["needs"]
    return {needs} if isinstance(needs, str) else set(needs)


def test_release_tag_requires_dispatch_gate_and_signed_artifact_attestation() -> None:
    workflow = _workflow()
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["on"]["workflow_dispatch"]["inputs"]["tag"]["required"] == "true"

    jobs = workflow["jobs"]
    gate = jobs["gate-and-build"]
    assert gate["permissions"] == {"contents": "read"}

    attestation = jobs["attest-artifacts"]
    assert _needs(attestation) == {"gate-and-build"}
    assert attestation["permissions"] == {
        "attestations": "write",
        "contents": "read",
        "id-token": "write",
    }
    attest_step = _step(attestation, "Attest release distributions and integrity metadata")
    assert attest_step["uses"] == "actions/attest@v4"
    assert set(attest_step["with"]["subject-path"].splitlines()) == {
        "dist/*.whl",
        "dist/*.tar.gz",
        "dist/SHA256SUMS",
        "dist/PROVENANCE.json",
    }

    tag = jobs["create-release-tag"]
    assert _needs(tag) == {"gate-and-build", "attest-artifacts"}
    assert tag["permissions"] == {"contents": "read"}
    assert (
        'git tag --annotate --message "Release $RELEASE_TAG"'
        in _step(tag, "Create and push annotated release tag")["run"]
    )


def test_release_tag_fails_closed_without_app_credentials_before_checkout() -> None:
    tag = _workflow()["jobs"]["create-release-tag"]
    steps = tag["steps"]
    guard = _step(tag, "Require release GitHub App credentials")
    token = _step(tag, "Create repository-scoped release tag token")
    checkout = _step(tag, "Checkout approved release commit")
    identity = _step(tag, "Configure audited release tag identity")
    push = _step(tag, "Create and push annotated release tag")
    names = [step.get("name") for step in steps]

    guard_index, token_index, checkout_index, identity_index, push_index = (
        names.index(step["name"]) for step in (guard, token, checkout, identity, push)
    )
    assert guard_index < token_index < checkout_index < identity_index < push_index
    assert guard["env"] == {
        "RELEASE_APP_ID": "${{ secrets.RELEASE_APP_ID }}",
        "RELEASE_APP_PRIVATE_KEY": "${{ secrets.RELEASE_APP_PRIVATE_KEY }}",
    }

    marker = "python - <<'PY'\n"
    assert marker in guard["run"]
    guard_script = guard["run"].split(marker, 1)[1].split("\nPY", 1)[0]
    for app_id, private_key, missing in (
        ("", "", {"RELEASE_APP_ID", "RELEASE_APP_PRIVATE_KEY"}),
        ("", "example-private-key", {"RELEASE_APP_ID"}),
        ("12345", "", {"RELEASE_APP_PRIVATE_KEY"}),
    ):
        result = subprocess.run(
            [sys.executable, "-c", guard_script],
            capture_output=True,
            check=False,
            env={
                **os.environ,
                "RELEASE_APP_ID": app_id,
                "RELEASE_APP_PRIVATE_KEY": private_key,
            },
            text=True,
        )
        assert result.returncode == 1
        message = "Add the missing repository secret(s): "
        reported = set(result.stderr.split("::error::" + message, 1)[1].strip().split(", "))
        assert missing <= reported
        for value in (app_id, private_key):
            if value:
                assert value not in result.stdout + result.stderr
    token_action = token["with"]
    assert token["uses"] == "actions/create-github-app-token@v3"
    assert token_action == {
        "app-id": "${{ secrets.RELEASE_APP_ID }}",
        "private-key": "${{ secrets.RELEASE_APP_PRIVATE_KEY }}",
        "owner": "${{ github.repository_owner }}",
        "repositories": "${{ github.event.repository.name }}",
        "permission-contents": "write",
    }
    assert checkout["with"]["token"] == "${{ steps.release-app-token.outputs.token }}"
    assert checkout["with"]["persist-credentials"] == "true"
    assert 'git push origin "$tag_ref"' in push["run"]

    secret_values = {
        "RELEASE_APP_ID": "example-app-id",
        "RELEASE_APP_PRIVATE_KEY": "example-private-key",
    }
    result = subprocess.run(
        [sys.executable, "-c", guard_script],
        capture_output=True,
        check=False,
        env={**os.environ, **secret_values},
        text=True,
    )
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert all(value not in output for value in secret_values.values())


def test_release_tag_identity_creates_annotated_tag_without_runner_config(tmp_path: Path) -> None:
    tag = _workflow()["jobs"]["create-release-tag"]
    configure = _step(tag, "Configure audited release tag identity")
    tag_step = _step(tag, "Create and push annotated release tag")
    commands = [shlex.split(line) for line in configure["run"].splitlines() if line.strip()]
    assert commands == [
        ["git", "config", "user.name", "github-actions[bot]"],
        ["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"],
    ]
    assert 'git tag --annotate --message "Release $RELEASE_TAG"' in tag_step["run"]

    repository = tmp_path / "repo"
    home = tmp_path / "home"
    repository.mkdir()
    home.mkdir()
    git_env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    git_env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "GIT_CONFIG_GLOBAL": str(home / "missing.gitconfig"),
            "GIT_CONFIG_NOSYSTEM": "1",
        }
    )

    def run_git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            env=git_env,
            capture_output=True,
            check=False,
            text=True,
        )

    assert run_git("init", "--quiet").returncode == 0
    seed = run_git(
        "-c",
        "user.name=Seed",
        "-c",
        "user.email=seed@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "seed",
    )
    assert seed.returncode == 0, seed.stderr

    tag_arguments = ("tag", "--annotate", "--message", "Release v1.0.0", "v1.0.0", "HEAD")
    without_identity = run_git(*tag_arguments)
    assert without_identity.returncode != 0

    for command in commands:
        configured = subprocess.run(
            command,
            cwd=repository,
            env=git_env,
            capture_output=True,
            check=False,
            text=True,
        )
        assert configured.returncode == 0, configured.stderr

    created = run_git(*tag_arguments)
    assert created.returncode == 0, created.stderr
    tag_object = run_git("cat-file", "-p", "refs/tags/v1.0.0")
    assert tag_object.returncode == 0, tag_object.stderr
    assert "tagger github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>" in tag_object.stdout


def test_pypi_staging_copies_only_distributions_matching_verified_hashes(tmp_path: Path) -> None:
    publish = _workflow()["jobs"]["publish-pypi"]
    attestations = _step(publish, "Verify GitHub build attestations")
    stage = _step(publish, "Stage only verified PyPI distributions")
    publish_step = _step(
        publish,
        "Publish distributions and attestations with PyPI Trusted Publishing",
    )
    steps = publish["steps"]
    names = [step.get("name") for step in steps]
    assert names.index(attestations["name"]) < names.index(stage["name"]) < names.index(publish_step["name"])

    marker = "python - <<'PY'\n"
    assert marker in stage["run"]
    stage_script = stage["run"].split(marker, 1)[1].split("\nPY", 1)[0]
    source = tmp_path / "dist"
    source.mkdir()
    expected = {
        "latent_anything-1.0.0-py3-none-any.whl": b"verified wheel bytes",
        "latent_anything-1.0.0.tar.gz": b"verified sdist bytes",
    }
    records = []
    for name, contents in expected.items():
        (source / name).write_bytes(contents)
        records.append({"name": name, "sha256": hashlib.sha256(contents).hexdigest()})
    (source / "SHA256SUMS").write_bytes(b"verified checksums\n")
    (source / "release-body.md").write_bytes(b"release notes\n")
    (source / "PROVENANCE.json").write_text(json.dumps({"files": records}), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-c", stage_script],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    staged = tmp_path / "dist-pypi"
    assert {path.name for path in staged.iterdir()} == set(expected)
    assert {name: (staged / name).read_bytes() for name in expected} == expected

    tampered = tmp_path / "tampered"
    tampered_source = tampered / "dist"
    tampered_source.mkdir(parents=True)
    tampered_records = []
    for name, contents in expected.items():
        (tampered_source / name).write_bytes(contents)
        tampered_records.append({"name": name, "sha256": hashlib.sha256(contents).hexdigest()})
    (tampered_source / "SHA256SUMS").write_bytes(b"verified checksums\n")
    (tampered_source / "release-body.md").write_bytes(b"release notes\n")
    (tampered_source / "PROVENANCE.json").write_text(
        json.dumps({"files": tampered_records}),
        encoding="utf-8",
    )
    (tampered_source / "latent_anything-1.0.0.tar.gz").write_bytes(b"changed sdist bytes")
    rejected = subprocess.run(
        [sys.executable, "-c", stage_script],
        cwd=tampered,
        capture_output=True,
        check=False,
        text=True,
    )
    assert rejected.returncode != 0
    assert "verified distribution changed before staging" in rejected.stderr
    assert {path.name for path in (tampered / "dist-pypi").iterdir()} == {"latent_anything-1.0.0-py3-none-any.whl"}


def test_release_tag_app_matches_integration_bypass_contract() -> None:
    workflow = _workflow()
    tag = workflow["jobs"]["create-release-tag"]
    token = _step(tag, "Create repository-scoped release tag token")
    gate_data = json.loads(_RELEASE_GATES.read_text(encoding="utf-8"))
    contract = gate_data["stable_tag_workflow"]

    assert contract["workflow_app"]["installation_repositories"] == ["latent-anything"]
    assert contract["workflow_app"]["repository_permissions"] == {"contents": "write"}
    assert contract["workflow_app"]["token_permissions"] == {"contents": "write"}
    assert contract["workflow_app"]["required_repository_secrets"] == [
        "RELEASE_APP_ID",
        "RELEASE_APP_PRIVATE_KEY",
    ]
    assert token["with"]["app-id"] == "${{ secrets.RELEASE_APP_ID }}"
    assert contract["tag_ruleset"] == {
        "target": "tag",
        "ref_include": ["refs/tags/v*"],
        "rules": ["creation", "update", "deletion", "non_fast_forward"],
        "enforcement": "active",
        "bypass_actor": {
            "actor_type": "Integration",
            "actor_id_from_secret": "RELEASE_APP_ID",
            "bypass_mode": "always",
        },
    }


def test_build_records_checksums_and_provenance_for_only_the_release_candidate() -> None:
    gate = _workflow()["jobs"]["gate-and-build"]
    build = _step(gate, "Build wheel and sdist")
    assert build["run"].splitlines() == ["rm -rf dist", "uv build --wheel --sdist --out-dir dist"]
    validate = _step(gate, "Validate package contents and record provenance")
    assert '"schema": "latent-anything-release-provenance-v1"' in validate["run"]
    assert '(dist / "SHA256SUMS").write_text(' in validate["run"]
    assert '(dist / "PROVENANCE.json").write_text(' in validate["run"]


def test_github_release_assets_are_uploaded_and_rehashed_after_the_tag() -> None:
    publish = _workflow()["jobs"]["publish"]
    assert _needs(publish) == {"gate-and-build", "create-release-tag"}
    assert publish["permissions"] == {"contents": "write"}
    create_release = _step(publish, "Create or update GitHub Release with verified assets")
    assert create_release["uses"] == "softprops/action-gh-release@v2"
    assert create_release["with"]["fail_on_unmatched_files"] == "true"
    verify_assets = _step(publish, "Verify uploaded GitHub Release asset hashes")
    assert "gh release download" in verify_assets["run"]
    assert "local_digest != remote_digest" in verify_assets["run"]
    assert "SHA256SUMS" in create_release["with"]["files"]
    assert "PROVENANCE.json" in create_release["with"]["files"]


def test_pypi_publish_is_oidc_only_and_follows_all_verified_release_jobs() -> None:
    publish = _workflow()["jobs"]["publish-pypi"]
    assert _needs(publish) == {
        "gate-and-build",
        "attest-artifacts",
        "create-release-tag",
        "publish",
    }
    assert publish["environment"] == {
        "name": "pypi",
        "url": "https://pypi.org/project/latent-anything/",
    }
    assert publish["permissions"] == {
        "attestations": "read",
        "contents": "read",
        "id-token": "write",
    }
    steps = publish["steps"]
    step_names = [step.get("name") for step in steps]
    assert step_names.index("Verify package checksums, provenance, and exact distribution set") < step_names.index(
        "Verify GitHub build attestations"
    )
    assert step_names.index("Verify GitHub build attestations") < step_names.index(
        "Stage only verified PyPI distributions"
    )
    assert step_names.index("Stage only verified PyPI distributions") < step_names.index(
        "Publish distributions and attestations with PyPI Trusted Publishing"
    )

    verify_attestations = _step(publish, "Verify GitHub build attestations")
    assert "gh attestation verify" in verify_attestations["run"]
    publish_step = _step(publish, "Publish distributions and attestations with PyPI Trusted Publishing")
    assert publish_step["uses"] == "pypa/gh-action-pypi-publish@v1.14.2"
    assert publish_step["with"] == {"packages-dir": "dist-pypi", "attestations": "true"}
    assert not {"username", "password"} & set(publish_step["with"])
    assert all("secrets." not in str(step) for step in steps)
