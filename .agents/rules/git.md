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
- **Never commit to `main` directly.** All changes go through a branch and are merged via pull request.
- Stage specific files by name — never use `git add .` or `git add -A`, which can accidentally include unintended files.

## Branch naming

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