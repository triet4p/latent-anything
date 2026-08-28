# Task Summary: Sprint 79 L04.7a — WikiText-2 acquisition manifest

**Sprint:** Sprint 79
**Task:** L04.7a

## Summary of Work

Implemented and completed the owner-approved one-time acquisition for the
frozen M14 L04 tuned-lens corpus. The offline-testable, explicitly
network-gated script makes two explicit `datasets.load_dataset` calls, one for
each permitted split, so an upstream `test` split can never enter the manifest.
It pins
`Salesforce/wikitext`, `wikitext-2-raw-v1`, revision
`f776294184f13b8ff2337b3841cf9269a6216d1e`, the CC BY-SA 3.0/GFDL license,
official train/validation sizes, seed 79, max 128 tokens per row, and the
8192/2048 selections. Blank rows are dropped; non-blank text is UTF-8 hashed
and selected by `(sha256(text), original split-local index)`. Seed 79 is stored
as downstream tuned-lens training provenance and does not randomize selection.
The sanitized manifest stores no raw text, only permitted provenance, counts,
selected indices/text hashes, and independent canonical content/split digests.

The single direct authenticated PowerShell `ssh.exe` acquisition used source
SHA `e6b1bf71de46d6b6879ce6c57fef9e939f1d2fcc`, completed with one command and
one result, passed remote cleanup, and deleted raw stdout/stderr after
verified sanitization. The retained manifest is 989,676 bytes with SHA-256
`0908f843efd72ce93c628e34cdb27f56e37e764d196ef91be7eff6d7757b78f3`, content
digest `bd235bad5a7643c860bca04a98ba545214f25702cd7625dd4ff591f0ea32cf7b`,
and split digest
`bb2dab8721bb8e244bf38f9add6af9e5c2fc70291ce4de6cf2263f7e0f970703`.
Train is 36,718 official / 23,767 non-blank / 8,192 selected; validation is
3,760 / 2,461 / 2,048. The sanitized audit self-digest recomputes to
`1593dc429e97776c2ab4334213b50b1450ddeb28f015432d478ab467d6cd1326`.

The standalone PEP 723 script uses the already locked `datasets==4.8.5`
dependency in an isolated script environment, rather than adding it to the
runtime package extras. This closure pass did not invoke acquisition or any
network, model, CUDA, or SSH operation; the owner-approved acquisition is
represented only by the retained sanitized manifest and audit records.

## Files Modified

- [`scripts/m14_l04_wikitext_manifest.py`](../scripts/m14_l04_wikitext_manifest.py) — pinned acquisition, selection, digest, validation, and atomic manifest writer.
- [`tests/test_m14_l04_wikitext_manifest.py`](../tests/test_m14_l04_wikitext_manifest.py) — synthetic offline selection, UTF-8, tamper, leakage, digest, and LF-byte coverage.
- [`m14/l04-wikitext-2-manifest.json`](m14/l04-wikitext-2-manifest.json) — retained sanitized manifest; raw corpus text is absent.
- [`m14/l04-wikitext-2-manifest.attempt1.audit.json`](m14/l04-wikitext-2-manifest.attempt1.audit.json) — one-time transport/acquisition/cleanup audit.
- [`m14/l04-wikitext-2-manifest.attempt1.exit.txt`](m14/l04-wikitext-2-manifest.attempt1.exit.txt) — sanitized exit and validator status.
- [`docs/sprint-plans/sprint-79.md`](../docs/sprint-plans/sprint-79.md) — marked L04.7a complete.
- [`CHANGELOG.md`](../CHANGELOG.md) — recorded the new user-visible acquisition/manifest tooling.
- [`.agents/memory/decisions.md`](../.agents/memory/decisions.md) — appended the owner decision preserving plan immutability and one-time corpus provisioning.

## Testing

- `uv run pytest tests/test_m14_l04_wikitext_manifest.py -q` — 20 passed.
- `uv run ruff check scripts/m14_l04_wikitext_manifest.py tests/test_m14_l04_wikitext_manifest.py` — passed.
- `uv run ruff format --check scripts/m14_l04_wikitext_manifest.py tests/test_m14_l04_wikitext_manifest.py` — passed.
- `uv run pyright scripts/m14_l04_wikitext_manifest.py` — passed.
- `uv run python -c "...validate_manifest(...)..."` on the retained manifest — passed.
- Audit self-digest recomputation, manifest SHA/content/split linkage, sanitized-field scan, and exact count/source/tool checks — passed.

## Additional Notes

The frozen plan `artifacts/m14/l04-explanations.plan.json` was not modified;
its canonical digest remains
`f3c315e356af0ee54d4196cc365ee22bd997b069d18a3e72c6b479f94e0b3e1a`.
The corpus-provisioning blocker is resolved for the acquisition lane, but
`TunedLogitLens` remains D0: a translator implementation and a real CUDA
execution are still required before any ledger promotion. No authored-fixture
substitution, corpus-text retention, rerun, or evidence-level change occurred.
