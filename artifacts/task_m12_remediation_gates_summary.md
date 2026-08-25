# Task Summary: M12 remediation R5 — validation and graph refresh

**Date:** 2026-08-25
**Scope:** bounded closure of the Sprint 70 VQ-VAE and Sprint 72 tokenized
world-model evidence blocker, plus governance/documentation consistency.

## Commands and results

| Command | Result |
| --- | --- |
| `uv run pytest -q` | **PASS** — 1371 passed, 32 skipped, 39 warnings in 221.55s |
| `uv run ruff check src tests scripts` | **PASS** — all checks passed |
| `uv run ruff format --check src tests scripts` | **PASS** — 229 files already formatted |
| `uv run pyright` | **PASS** — 0 errors, 0 warnings, 0 informations (version-update warning only) |
| `uv run python scripts/validate_evidence_ledger.py` | **PASS** — inventory 107; core 23/63; overall 23/65 |
| `uv run pytest tests/test_scripts/test_validate_evidence_ledger.py -q` | **PASS** — 10 passed in 1.12s |
| `uv run --extra docs mkdocs build --strict` | **PASS** — documentation built in 141.16s; upstream MkDocs 2.0 warning only |
| `uv run ruff check .` | **BLOCKED by pre-existing scope** — 1920 findings, predominantly `.agents` skill files and theory notebooks; changed source/test/script scope is clean |
| `uv run ruff format --check .` | **BLOCKED by pre-existing scope** — 106 files would be reformatted, predominantly `.agents` and theory notebooks; changed source/test/script scope is clean |

All nine M12 reproduction scripts were rerun successfully on CPU:

* deterministic, stochastic, and RSSM transition benchmarks — D2, stable;
* reward/value, CEM, and MPPI planning benchmarks — acceptance fields true;
* VQ-VAE digits evidence — strict non-collapse acceptance true;
* JEPA world-model evidence — latent collapse fraction 0 and stable rollout;
* tokenized world-model evidence — D2, tokenizer perplexity 5.50977280035793,
  dead-code rate 0, active train/held-out codes 8/8, and explicit greedy failure
  at horizon 1.

`graphify update .` completed successfully on the final tree: the final owner
refresh rebuilt the graph to 8942 nodes, 17433 edges, and 781 communities.
The tool reported 42 non-code JSON files with zero AST nodes and a community
label-refresh recommendation; these do not affect source validation.

## Closure decision

The bounded M12 remediation is complete and `docs/PLAN.md` marks Milestone 12
and Sprint 72 remediation as done. The broad Ruff checks remain explicitly
reported as repository-scope limitations rather than being claimed as passes
or “fixed” by changing unrelated `.agents`/theory files.
