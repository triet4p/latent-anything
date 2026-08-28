# Task Summary: Sprint 79 L04.3 — Fail-closed explanation dispatch infrastructure

**Sprint:** Sprint 79  
**Task:** L04.3  
**Status:** complete for infrastructure only; no real-model evidence was produced.

## Summary

The L04 CLI now preserves the offline `--check` contract and exposes a
fail-closed `--run-real --use-case` dispatcher. Each invocation handles exactly
one of the seven frozen use cases and atomically writes use-case-specific
partial-artifact, run-record, and failure-envelope JSON files under the caller
selected directory. It never writes the finalizer-owned
`l04-explanations.json`.

Production dispatch does not load a model or claim computation. Integrated
Gradients, TCAV, direct lens, disentanglement, true interchange patching, and
additive steering return explicit `not_implemented_pending_L04.x` statuses;
tuned lens returns `blocked_missing_corpus`. A private dependency-injection
seam is available for later offline mechanics tests only; injected handlers are
marked `dependency-injected-offline`, evidence-ineligible, D0, and cannot be
accepted or promoted.

## Files

- [`scripts/m14_l04_explanations.py`](../scripts/m14_l04_explanations.py) — CLI,
  one-use-case dispatch, and handler seam.
- [`scripts/_m14_l04_artifact.py`](../scripts/_m14_l04_artifact.py) — seven
  execution/five ledger-record artifact construction.
- [`scripts/_m14_l04_envelope.py`](../scripts/_m14_l04_envelope.py) — run and
  failure envelopes with compatibility validator exports.
- [`scripts/_m14_l04_digest.py`](../scripts/_m14_l04_digest.py) — strict code
  SHA and deterministic task-source digests.
- [`scripts/_m14_l04_io.py`](../scripts/_m14_l04_io.py) — safe atomic JSON writes.
- [`scripts/_m14_l04_validate.py`](../scripts/_m14_l04_validate.py) — fail-closed
  schema, digest, linkage, mapping, and evidence-eligibility validation. Full
  plan/fixture validation is a pre-write barrier; recorded source maps remain
  auditable after later L04 source changes.
- [`scripts/_m14_l04_boundary.py`](../scripts/_m14_l04_boundary.py) — lazy
  `TransformerLMIntegration` identity seam; no model is constructed.
- [`tests/test_m14_l04_runner.py`](../tests/test_m14_l04_runner.py) — offline
  status, isolation, tamper, injection, atomic-write, CLI, and factory tests.

## Verification

- `uv run pytest -q` — 1665 passed, 36 skipped, 39 warnings; focused L04 tests
  report 33 passed.
- `uv run ruff format --check scripts tests` — PASS.
- `uv run ruff check scripts tests` — PASS.
- `uv run pyright` — PASS.
- `uv run python scripts/validate_evidence_ledger.py` — unchanged honest
  `33/63` core and `33/65` overall.
- `uv run mkdocs build --strict` — PASS.
- `graphify update .` — PASS.
- `git diff --check` — PASS.

No commit, push, remote/model/network execution, or L04.4+ computation was
performed. Existing evidence levels remain unchanged.
