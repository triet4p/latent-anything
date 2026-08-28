# Task Summary: Sprint 79 L04.7a — WikiText-2 acquisition manifest

**Sprint:** Sprint 79
**Task:** L04.7a

## Summary of Work

Implemented an offline-testable and explicitly network-gated acquisition
script for the frozen M14 L04 tuned-lens corpus. The script makes two explicit
`datasets.load_dataset` calls, one for each permitted split, so an upstream
`test` split can never enter the manifest. It pins
`Salesforce/wikitext`, `wikitext-2-raw-v1`, revision
`f776294184f13b8ff2337b3841cf9269a6216d1e`, the CC BY-SA 3.0/GFDL license,
official train/validation sizes, seed 79, max 128 tokens per row, and the
8192/2048 selections. Blank rows are dropped; non-blank text is UTF-8 hashed
and selected by `(sha256(text), original split-local index)`. Seed 79 is stored
as downstream tuned-lens training provenance and does not randomize selection.
The sanitized
manifest stores no raw text, only permitted provenance, counts, selected
indices/text hashes, and independent canonical content/split digests.

The standalone PEP 723 script uses the already locked `datasets==4.8.5`
dependency in an isolated script environment, rather than adding it to the
runtime package extras. Acquisition was not invoked and no network, model,
CUDA, or SSH operation was performed.

## Files Modified

- [`scripts/m14_l04_wikitext_manifest.py`](../scripts/m14_l04_wikitext_manifest.py) — pinned acquisition, selection, digest, validation, and atomic manifest writer.
- [`tests/test_m14_l04_wikitext_manifest.py`](../tests/test_m14_l04_wikitext_manifest.py) — synthetic offline selection, UTF-8, tamper, leakage, digest, and LF-byte coverage.
- [`docs/sprint-plans/sprint-79.md`](../docs/sprint-plans/sprint-79.md) — marked L04.7a complete.
- [`CHANGELOG.md`](../CHANGELOG.md) — recorded the new user-visible acquisition/manifest tooling.
- [`.agents/memory/decisions.md`](../.agents/memory/decisions.md) — appended the owner decision preserving plan immutability and one-time corpus provisioning.

## Testing

- `uv run pytest tests/test_m14_l04_wikitext_manifest.py -q` — 20 passed.
- `uv run ruff check scripts/m14_l04_wikitext_manifest.py tests/test_m14_l04_wikitext_manifest.py` — passed.
- `uv run ruff format --check scripts/m14_l04_wikitext_manifest.py tests/test_m14_l04_wikitext_manifest.py` — passed.
- `uv run pyright scripts/m14_l04_wikitext_manifest.py` — passed.

## Additional Notes

The frozen plan `artifacts/m14/l04-explanations.plan.json` was not modified;
its canonical digest remains
`f3c315e356af0ee54d4196cc365ee22bd997b069d18a3e72c6b479f94e0b3e1a`.
Pending corpus digests are intentionally not fabricated. A future owner-
approved acquisition may write `artifacts/m14/l04-wikitext-2-manifest.json`,
which must pass the validator before tuned-lens execution. No authored fixture
substitution or text retention is permitted.
