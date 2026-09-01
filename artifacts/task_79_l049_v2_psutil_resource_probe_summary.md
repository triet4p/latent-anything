# Task Summary: Sprint 79 L04.9 v2 resource-probe dependency isolation

**Sprint:** Sprint 79
**Task:** Fresh-clone RSS dependency and deterministic resource-probe tests

## Summary of Work

Added `psutil` as a direct runtime dependency for honest Windows/Linux RSS
tracking. The resource-only probe now accepts injected resource and psutil
modules, allowing tests to cover both measured RSS and unavailable RSS without
ambient host dependencies. Added explicit coverage that missing resource and
psutil measurements publish `rss_unavailable` with unavailable CPU provenance
and are rejected by the `require_measured=True` eligibility gate, while the
positive measured case passes that same gate.

## Files Modified

* `pyproject.toml` - direct compatible `psutil` runtime dependency.
* `uv.lock` - refreshed project dependency metadata.
* `scripts/m14_l049_v2_resource_probe.py` - deterministic measurement seams.
* `tests/test_m14_l049_v2.py` - injected RSS success and unavailable-resource
  coverage.

## Testing

The repository-standard test profile is the locked visualization environment:
`uv sync --locked --extra viz`, followed by `uv run pytest ...`. The docs gate
uses its separate locked profile: `uv sync --locked --extra docs`, followed by
`uv run mkdocs build --strict`.

* Focused resource/probe suite (locked `viz` profile):
  `uv run --extra viz pytest tests/test_m14_l049_v2.py -k resource_probe -q` —
  `6 passed, 204 deselected`.
* Full L04 matrix (locked `viz` profile; `test_m14_l04_*.py`, v2, and the
  validation contract): `570 passed, 1 skipped`.
* Full project suite: an earlier base-environment snapshot reported `2162
  passed, 46 skipped`; because it was not run under the repository-standard
  `viz` profile, it is an incomplete environment result, not a definitive
  project gate.
* Strict Pyright: `uv run pyright` — `0 errors, 0 warnings, 0 informations`.
* Ruff check/format and `git diff --check`: passed for the changed files.
* `uv lock --check`; `uv sync --locked --extra viz`; and
  `uv sync --locked --extra docs`: passed.
* MkDocs strict (locked `docs` profile): `uv run mkdocs build --strict` —
  passed; only the upstream Material-for-MkDocs 2.0 advisory was emitted.
  An earlier base-environment attempt was blocked by the missing
  `mkdocs-material` theme; that was an incomplete environment result, not a
  documentation failure or definitive gate. No documentation files were
  changed.

* `graphify update .`: passed; graph rebuilt with 13,127 nodes, 27,093 edges,
  and 994 communities.

## Additional Notes

No evidence artifacts or promotion state were changed; D3 remains `0`. No
commit or push was performed.
