"""Tests for release-note extraction used by the GitHub Release workflow."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "extract_release_notes.py"
_SPEC = importlib.util.spec_from_file_location("extract_release_notes", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

extract_release_notes = _MODULE.extract_release_notes
is_prerelease_version = _MODULE.is_prerelease_version
normalize_tag = _MODULE.normalize_tag


def test_normalize_tag_removes_optional_v_prefix() -> None:
    """A leading v on numeric release tags is ignored for changelog lookup."""
    assert normalize_tag("v0.1.0-beta.1") == "0.1.0-beta.1"
    assert normalize_tag("0.1.0-beta.1") == "0.1.0-beta.1"
    assert normalize_tag("refs/tags/v0.1.0-beta.1") == "0.1.0-beta.1"


def test_normalize_tag_keeps_theory_prefix_distinct() -> None:
    """Theory deploy tags are not normalized into package release versions."""
    assert normalize_tag("theory-v0.1.0") == "theory-v0.1.0"


def test_extract_release_notes_returns_complete_section() -> None:
    """The matching changelog section becomes release body text."""
    changelog = """# Changelog

## [Unreleased]

### Added

- Next change.

## [0.1.0-beta.1] - 2026-07-10

### Added

- Headline capability summary.

### Known limitations

- Probing remains future work.

## [0.0.1] - 2026-06-01

- Older change.
"""

    notes = extract_release_notes(changelog, "v0.1.0-beta.1")

    assert notes.version == "0.1.0-beta.1"
    assert notes.date == "2026-07-10"
    assert notes.title == "Latent Anything 0.1.0-beta.1 - Core latent-space framework beta"
    assert notes.prerelease is True
    assert "## [0.1.0-beta.1] - 2026-07-10" in notes.body
    assert "Headline capability summary." in notes.body
    assert "Older change." not in notes.body


def test_extract_release_notes_fails_without_matching_changelog_section() -> None:
    """Release tags without changelog sections fail before release creation."""
    changelog = """# Changelog

## [Unreleased]

- Next change.
"""

    with pytest.raises(ValueError, match="No CHANGELOG.md section"):
        extract_release_notes(changelog, "v0.1.0-beta.1")


def test_extract_release_notes_fails_when_section_has_no_body_content() -> None:
    """A release section must contain more than its version heading."""
    changelog = """# Changelog

## [Unreleased]

## [0.1.0-beta.1] - 2026-07-10

## [0.0.1] - 2026-06-01

- Older change.
"""

    with pytest.raises(ValueError, match="no release body content"):
        extract_release_notes(changelog, "v0.1.0-beta.1")


def test_prerelease_detection_handles_beta_rc_and_stable_versions() -> None:
    """Beta and rc tags become prereleases; stable semver does not."""
    assert is_prerelease_version("0.1.0-beta.1") is True
    assert is_prerelease_version("0.1.0-rc.1") is True
    assert is_prerelease_version("0.1.0b1") is True
    assert is_prerelease_version("0.1.0") is False
