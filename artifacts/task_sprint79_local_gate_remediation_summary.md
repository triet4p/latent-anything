# Sprint79-LocalGateRemediation

## Batch

This artifact records the three consecutive Sprint 79 remediation tasks as one
explicitly tagged batch: `Sprint79-LocalGateRemediation`.

## Observable correction

Successful production Integrated Gradients dispatch now derives and retains a
canonical `execution_result_digest` for the real handler result, rejects a
provided mismatched digest, and propagates the verified digest through the
artifact provenance, run record, and failure envelope (including the retained
nested run record). The validators require a valid digest and exact triad
linkage for a successful real Integrated Gradients execution. Focused
regression coverage observes the same 64-hex digest in all triad members while
all three envelope validators return empty error lists. Support-only evidence
remains unpromoted and fail-closed.

The Ruff formatter was applied only to
`scripts/_m14_l04_validate_tcav.py`; the formatting check is clean.

The live evidence ledger remains below release thresholds at 36/63 core
(57.142857%) and 36/65 overall (55.384615%), with 24 D0 rows and 5 D1 rows;
24 additional core qualifiers and 23 additional overall qualifiers are still
required to reach 95% core / 90% overall. No release gate or blocker was
weakened.

## Exact files changed

- `scripts/_m14_l04_digest.py`
- `scripts/_m14_l04_integrated_gradients.py`
- `scripts/_m14_l04_artifact.py`
- `scripts/_m14_l04_envelope.py`
- `scripts/_m14_l04_validate.py`
- `scripts/_m14_l04_validate_tcav.py` (formatter-only)
- `tests/test_m14_l04_integrated_gradients_handler.py`
- `docs/EVIDENCE_GAP_PLAN.md`
- `docs/PLAN.md`
- `docs/sprint-plans/sprint-79.md`
- `CHANGELOG.md`
- `artifacts/task_sprint79_local_gate_remediation_summary.md`

No later pending Sprint 79 item was marked complete.

## Focused verification

- Reproduction command: `uv run pytest tests/test_m14_l04_integrated_gradients_handler.py::test_production_dispatch_success_keeps_real_result_and_validates_triads -q` — **failed before the fix** at the missing artifact-provenance digest assertion; **passed after the fix (1 passed)**.
- `uv run pytest tests/test_m14_l04_integrated_gradients_handler.py -q` — **22 passed**.
- `uv run pytest tests/test_m14_l04_runner.py -q` — **18 passed**.
- `uv run pytest tests/test_m14_l04_direct_lens_handler.py tests/test_m14_l04_tcav_handler.py tests/test_m14_l04_tuned_lens.py tests/test_m14_l04_disentanglement.py tests/test_m14_l04_activation_patching.py -q` — **145 passed, 1 skipped**.
- `uv run ruff format scripts/_m14_l04_validate_tcav.py` — **1 file reformatted**.
- `uv run ruff format --check scripts/_m14_l04_validate_tcav.py` — **1 file already formatted**.
- `uv run python -m py_compile scripts/_m14_l04_digest.py scripts/_m14_l04_integrated_gradients.py scripts/_m14_l04_artifact.py scripts/_m14_l04_envelope.py scripts/_m14_l04_validate.py scripts/_m14_l04_validate_tcav.py` — **passed (no output)**.
- `uv run python scripts/validate_evidence_ledger.py --json` — **exit status 0; `errors: []`; coverage `core: [36, 63, 0.5714285714285714]`, `overall: [36, 65, 0.5538461538461539]`**.

## Follow-up documentation consistency correction

The evidence-table rows now match the live ledger and the L03/L04 records:
`THY-T03-LINEAR-STRUCTURE-TRONG-LATENT` is D2,
`THY-T05-LINEAR-PROBING` is D2,
`THY-T05-NONLINEAR-PROBING` is D2, and
`THY-T05-ACTIVATION-PATCHING` is D3. Their stale “missing benchmark” and
“D1 hook tests” rationales were replaced with the corresponding validated
L03 held-out GPT-2 benchmarks and exact-SHA real-CUDA interchange evidence.

Additional focused verification:

- `uv run python scripts/validate_evidence_ledger.py --json` — **exit status 0,
  `errors: []`; coverage remains core [36, 63, 0.5714285714285714] and overall
  [36, 65, 0.5538461538461539]**.
- `uv run python -c "from pathlib import Path; q=chr(96); e=Path('docs/EVIDENCE_GAP_PLAN.md').read_text(encoding='utf-8'); p=Path('docs/PLAN.md').read_text(encoding='utf-8'); assert f'{q}THY-T03-LINEAR-STRUCTURE-TRONG-LATENT{q} | D2 |' in e; assert f'{q}THY-T05-LINEAR-PROBING{q} | D2 |' in e; assert f'{q}THY-T05-NONLINEAR-PROBING{q} | D2 |' in e; assert f'{q}THY-T05-ACTIVATION-PATCHING{q} | D3 |' in e; assert 'Sprint 79 L04 implementation and local contract/remediation gates are in place' in p; print('DOC_STATUS_ROWS_OK')"` — **`DOC_STATUS_ROWS_OK`**.

## Sprint plan status

The three consecutive batch items at `docs/sprint-plans/sprint-79.md` are
marked `[x]` only after their focused checks passed. Milestone 14 and the
stable release remain incomplete in `docs/PLAN.md`; real-system evidence,
release-candidate gates, and `1.0.0` publication remain pending.
