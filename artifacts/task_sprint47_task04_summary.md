# Task Summary: Sprint 47 Task 04

**Sprint:** Sprint 47
**Task:** Keep visualization optional; prove base-package import isolation

Added the `viz` optional extra (`plotly`, `kaleido>=1.0.0`, `ipywidgets`,
`anywidget`) and kept plotly/kaleido/ipywidgets out of every module-level
import in `latent_anything.visualization` (lazy `require_optional` inside
frontend functions). `latent_anything/__init__.py` does not import the
subpackage.

**Testing:** 6 isolation tests in `tests/test_visualization_isolation.py` prove `import latent_anything` and the visualization package never pull the frontends, and that a missing plotly raises the actionable `uv sync --extra viz` error.
