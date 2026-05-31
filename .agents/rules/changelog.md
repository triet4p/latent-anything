# Changelog Rules

Follow the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format with [Semantic Versioning](https://semver.org/).

## File location

`CHANGELOG.md` at the repository root.

## Structure

```markdown
# Changelog

## [Unreleased]

### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security

## [x.y.z] - YYYY-MM-DD
...
```

- Keep `[Unreleased]` at the top at all times. New entries always go here first.
- When a version is released, rename `[Unreleased]` to `[x.y.z] - YYYY-MM-DD` and open a fresh `[Unreleased]` above it.
- Versions are listed newest-first.

## When to update

Update `CHANGELOG.md` in the same commit as the code change — not as a separate follow-up. Every user-visible change must have an entry before the commit is made.

## Sections and what goes in them

| Section | Use for |
|---|---|
| `Added` | New features or capabilities |
| `Changed` | Changes to existing behavior (including performance improvements) |
| `Deprecated` | Features that will be removed in a future version |
| `Removed` | Features removed in this version |
| `Fixed` | Bug fixes |
| `Security` | Vulnerability patches |

Omit any section that has no entries for a given version.

## Writing entries

- Write for the **user**, not the developer. Describe what changed from the user's perspective, not how the code changed internally.
- Use plain past tense: "Added `Trajectory.concat()`" not "feat: add trajectory concat".
- Be specific: name the class, method, config key, or behavior that changed.
- One entry per logical change. Do not bundle unrelated changes into one bullet.
- Do not copy-paste git commit messages — changelog entries are higher-level than commits.

## What NOT to include

- Internal refactors with no user-visible effect.
- Test-only changes.
- Documentation-only changes (unless they fix a documented API that was wrong).
- Dependency bumps (unless they fix a security issue or change user-visible behavior).
