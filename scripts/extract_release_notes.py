"""Extract GitHub Release metadata from CHANGELOG.md."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

_SECTION_RE = re.compile(r"^## \[(?P<version>[^\]]+)\] - (?P<date>.+)$")
_PEP440_PRERELEASE_RE = re.compile(r"(?:a|b|rc)\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class ReleaseNotes:
    """Release metadata extracted from a changelog section."""

    version: str
    date: str
    title: str
    body: str
    prerelease: bool


def normalize_tag(tag: str) -> str:
    """Normalize a pushed tag to the changelog version key."""
    normalized = tag.removeprefix("refs/tags/")
    if normalized.startswith("v") and len(normalized) > 1 and normalized[1].isdigit():
        return normalized[1:]
    return normalized


def is_prerelease_version(version: str) -> bool:
    """Return whether *version* should create a GitHub prerelease."""
    lower = version.lower()
    has_semver_marker = any(marker in lower for marker in ("-alpha", "-beta", "-rc"))
    has_pep440_marker = _PEP440_PRERELEASE_RE.search(lower) is not None
    return has_semver_marker or has_pep440_marker


def build_release_title(version: str) -> str:
    """Build the explicit GitHub Release title."""
    suffix = "Core latent-space framework beta" if is_prerelease_version(version) else "Core latent-space framework"
    return f"Latent Anything {version} - {suffix}"


def extract_release_notes(changelog_text: str, tag: str) -> ReleaseNotes:
    """Extract the matching changelog section for *tag*.

    The tag may include an optional leading ``v``. The matching
    changelog heading must use ``## [<version>] - <date>``.
    """
    version = normalize_tag(tag)
    lines = changelog_text.splitlines()
    start_index: int | None = None
    date = ""

    for index, line in enumerate(lines):
        match = _SECTION_RE.match(line)
        if match is None:
            continue
        if match.group("version") == version:
            start_index = index
            date = match.group("date")
            break

    if start_index is None:
        msg = f"No CHANGELOG.md section found for release tag {tag!r} (normalized to {version!r})."
        raise ValueError(msg)

    end_index = len(lines)
    for index in range(start_index + 1, len(lines)):
        if lines[index].startswith("## "):
            end_index = index
            break

    body_lines = lines[start_index:end_index]
    content_lines = [
        line for line in lines[start_index + 1 : end_index] if line.strip() != "" and not line.startswith("<!--")
    ]
    if not content_lines:
        msg = f"CHANGELOG.md section for {version!r} has no release body content."
        raise ValueError(msg)

    body = "\n".join(body_lines).strip() + "\n"

    return ReleaseNotes(
        version=version,
        date=date,
        title=build_release_title(version),
        body=body,
        prerelease=is_prerelease_version(version),
    )


def write_github_output(output_path: Path, values: dict[str, str]) -> None:
    """Append key/value pairs to a GitHub Actions output file."""
    with output_path.open("a", encoding="utf-8") as output_file:
        for key, value in values.items():
            output_file.write(f"{key}={value}\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Release tag, for example v0.1.0-beta.1")
    parser.add_argument("--changelog", default="CHANGELOG.md", help="Path to CHANGELOG.md")
    parser.add_argument("--body-file", required=True, help="Path to write the extracted release body")
    parser.add_argument("--github-output", help="Optional GitHub Actions output file")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    changelog_path = Path(args.changelog)
    body_path = Path(args.body_file)

    notes = extract_release_notes(changelog_path.read_text(encoding="utf-8"), args.tag)
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_path.write_text(notes.body, encoding="utf-8")

    print(f"version={notes.version}")
    print(f"title={notes.title}")
    print(f"prerelease={str(notes.prerelease).lower()}")
    print(f"body_path={body_path}")

    if args.github_output is not None:
        write_github_output(
            Path(args.github_output),
            {
                "version": notes.version,
                "title": notes.title,
                "prerelease": str(notes.prerelease).lower(),
                "body_path": str(body_path),
            },
        )


if __name__ == "__main__":
    main()
