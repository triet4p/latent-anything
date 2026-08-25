# Sprint 35 Carryover Remediation 05 — Portable Fidelity Evidence

**Status:** Complete (2026-08-26)

## Scope

The final fidelity artifact contained an absolute Windows checkpoint path,
which made an otherwise revision-pinned local-only evidence record machine
specific. The artifact now records the canonical repository-relative label
`.cache/hf-sd-vae-ft-mse-31f26fdeee1355a5c34592e401dd41e45d25a493` while
retaining the model ID, revision, safetensors size, and SHA-256 provenance.

The C35.3 and C35.4 summary headers were also corrected: C35.3 is explicitly
the completed fidelity-only historical boundary, and C35.4 is the completed
final interpolation/Milestone 8 closure.

## Validation

- `uv sync --locked` — passed; the lockfile remained unchanged.
- `uv run pytest tests/test_diffusers_vae_fidelity.py -q -m 'not large_download'` — 2 passed, 1 deselected.
- `git diff --check` — passed with ordinary Windows LF-to-CRLF warnings.
- `graphify update .` — passed; 10,167 nodes, 19,729 edges, 899 communities.

No metrics, model revision, checkpoint hash, or declared evidence scope was
changed to conceal history.
