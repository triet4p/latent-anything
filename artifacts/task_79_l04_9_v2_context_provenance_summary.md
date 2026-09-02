# Task Summary: Sprint 79 L04.9 v2 context provenance

## Summary

Replaced the raw policy-bound CLI validation context with an opaque frozen,
slot-based, weak-referenceable value whose identity and immutable provenance
snapshot are retained in a PID-bound registry. Validation now requires exact
runtime type, live registry identity, weak-reference identity, unchanged
fields, canonical syntax, and the issuing process ID. Stage CLI digests are
read only from the registry snapshot. Copy/deepcopy, direct construction,
field mutation, stale process IDs, and pickle attempts fail closed. The former
public-looking bind helper is no longer exported; only the canonical promotion
path issues contexts after independently reloading the exact
`RealPromotionPolicy` and its pinned D1/D2/source commitments.

## Files

- `scripts/_m14_l049_v2_validation_context.py` — registry-backed context and
  identity validation.
- `scripts/_m14_l049_v2_promotion.py` — exact policy/source checks and sole
  canonical issuance path.
- `tests/test_m14_l049_v2_validation_context.py` — focused provenance,
  lifecycle, tamper, PID, concurrency, and serialization coverage.
- `docs/M14_REAL_SYSTEM_VALIDATION.md` — realistic in-process threat model.
- `CHANGELOG.md` — user-visible security/validation behavior.

## Verification

- `uv run pytest tests/test_m14_l049_v2_validation_context.py -q` — 7 passed.
- `uv run pytest tests/test_m14_l049_v2.py -q` — 212 passed.
- Official D3 production validation with immutable output snapshot — 2 passed.
- `uv run ruff check` and `uv run ruff format --check` on changed Python —
  passed.
- Strict Pyright on the changed context/promotion modules — 0 errors.
- `graphify update .` — completed; graph rebuilt with 13,184 nodes and 27,211
  edges (the expected graphify outputs are dirty).

The accepted D3 artifact was not rewritten; its exact bytes and SHA-256 remain
unchanged.
