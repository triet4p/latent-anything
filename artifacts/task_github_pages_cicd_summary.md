# Task Summary: GitHub Pages CI/CD for lumen-theory

**Sprint:** Ad-hoc
**Task:** Set up automated deployment of `lumen-theory` to GitHub Pages via GitHub Actions

## Summary of Work

Created a CI/CD pipeline that automatically builds and deploys the `lumen-theory` sub-project as a static MkDocs site to `https://triet4p.github.io/lumen/` whenever changes are pushed to `main` under the `lumen-theory/` directory.

- `lumen-theory/mkdocs.yml` — MkDocs Material configuration with Vietnamese locale, MathJax support for LaTeX, `mkdocs-jupyter` for notebook rendering, and explicit `nav` covering all 6 theory notes and 2 notebooks.
- `.github/workflows/deploy-lumen-theory.yml` — GitHub Actions workflow using `astral-sh/setup-uv` + `uvx` (no `pyproject.toml` changes needed) to build the site, then `peaceiris/actions-gh-pages@v4` to push to the `gh-pages` branch.
- `.gitignore` — Added `lumen-theory/site/` to prevent local build output from being committed.

The workflow uses `uvx --with mkdocs-material --with mkdocs-jupyter` so no additional packages need to be added to `pyproject.toml`. It triggers on every push to `main` that touches `lumen-theory/**` or the workflow file itself, and can also be run manually via `workflow_dispatch`.

## Files Modified

- [lumen-theory/mkdocs.yml](../lumen-theory/mkdocs.yml) — New MkDocs site configuration
- [.github/workflows/deploy-lumen-theory.yml](../.github/workflows/deploy-lumen-theory.yml) — New GitHub Actions deployment workflow
- [.gitignore](../.gitignore) — Added `lumen-theory/site/` exclusion

## Testing

**Automated:** The workflow runs `mkdocs build` on every push; a failed build blocks deployment.

**Local verification:**
```bash
cd lumen-theory
uvx --with mkdocs-material --with mkdocs-jupyter mkdocs build
uvx --with mkdocs-material --with mkdocs-jupyter mkdocs serve
# → http://127.0.0.1:8000/lumen/
```

## Post-deploy Setup (one-time, manual)

After the first push, enable GitHub Pages in the repository settings:
1. Go to `https://github.com/triet4p/lumen/settings/pages`
2. Set **Source** → "Deploy from a branch"
3. Select branch `gh-pages`, folder `/(root)`
4. Click **Save**

The site will then be live at `https://triet4p.github.io/lumen/`.

## Additional Notes

- `force_orphan: true` keeps the `gh-pages` branch as a single-commit branch — the deployment history stays clean.
- `execute: false` in `mkdocs-jupyter` renders saved notebook outputs without re-running cells, so there is no need to install ML dependencies (torch, sklearn, etc.) in CI.
- Image references using relative paths (e.g., `../assets/euclidean-blind.png`) are resolved correctly by MkDocs because `docs_dir: .` preserves the existing directory structure.
- `exclude_docs: .venv/**` prevents the virtual environment from being copied into the `site/` output.
