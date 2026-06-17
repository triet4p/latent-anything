# Git Rules

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/) for all commit messages.

**Format:** `<type>(<scope>): <short description>`

- `<scope>` is optional but recommended — use the layer or module name (e.g., `core`, `adapter`, `layer-a`, `runtime`).
- Description is lowercase, imperative mood, no trailing period.
- Add a body (blank line after subject) when the *why* is not obvious from the diff.

Common types:

| Type | When to use |
|---|---|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation changes only |
| `chore` | Maintenance tasks (deps, config, tooling) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or updating tests |
| `style` | Formatting, whitespace — no logic change |

## Commits

- **One logical change per commit.** Do not mix unrelated changes (e.g., a bug fix and a refactor) in the same commit.
- **Solo project:** direct commits to `main` are acceptable. Skip the branch/PR flow.
- Stage specific files by name — never use `git add .` or `git add -A`, which can accidentally include unintended files.

## Branch naming

Only needed when collaborating or experimenting with risky changes.

**Format:** `<type>/<short-description>`

Examples: `feat/trajectory-immutable`, `fix/umap-cache-key`, `docs/architecture-update`, `chore/add-ruff-config`

- Use the same type prefixes as commit messages.
- Lowercase, hyphens only — no underscores or slashes beyond the type prefix.

## What not to commit

Never commit:
- `.venv/`, `__pycache__/`, `*.pyc`, `*.pyo`
- `.env` or any file containing secrets or API keys
- Build artifacts, compiled outputs, or large binary files
- IDE/editor config (`.vscode/`, `.idea/`) unless agreed upon by the team

Ensure a `.gitignore` covers these before the first commit in any new sub-project.

## Theory deployment

The deploy workflow (`.github/workflows/deploy-latent-anything-theory.yml`) triggers **only** on:

1. **Push tag** matching `theory-v*` — e.g., `git tag theory-v0.2.0 && git push origin theory-v0.2.0`.
2. **Manual dispatch** via GitHub UI — for emergency or pre-release testing.

Do **not** deploy theory content by pushing to `main` alone. Tag-based deployment is intentional — it gives a final review checkpoint before content goes live on GitHub Pages.

**Tag naming convention for theory:** `theory-v<major>.<minor>.<patch>` (e.g., `theory-v0.1.0`, `theory-v0.2.0`). Tags follow [Semantic Versioning](https://semver.org/) independently of the main framework package version.