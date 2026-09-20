# Sprint 79 baseline closure batch (tasks 595, 598, 600)

English batch summary for the consecutive 79.595 / 79.598 / 79.600 reconciliation batch.
`0.9.0` remains unreleased: no version bump, tag, publication, or alias removal was performed.

## Scope

- Task 595: freeze the exhaustive theory-gap inventory as the honest `0.9.0` evidence baseline.
- Task 598: reclassify explanation failures as blockers for corresponding Sprint 80 diagnostic claims.
- Task 600: freeze measured performance evidence/limitations and move unavailable overhead work to secondary integration.

## Preserved negative evidence (unchanged bytes, D-levels, counts)

- Ledger remains exactly **41/63 core and 41/64 scoped overall** (validator: `uv run python scripts/validate_evidence_ledger.py --json`).
- 19 D0 rows and 5 D1 rows among the active applicable inventory; qualifying rows remain D2/D3 only.
- Retained failures/threshold misses: TCAV Wilson lower `0.5291118177871466` vs `> 0.55` and corrected empirical p `0.24` vs `<= 0.05`; SAE minimum matched cosine `0.7387987235614891` vs `> 0.85`; steering randomized-direction failures for seeds 29 (`0.32630348205566406`) and 67 (`0.10096931457519531`) vs `<= 0.1`; manifold held-out ranking AUC `0.4560546875` vs `0.55` with latent-vs-raw delta `-0.4124755859375` vs `-0.05`.
- L17 remains blocked (no named 3DGS checkpoint); L19/OpenVLA remains a historical D0 hardware-excluded feasibility record authorizing no capability/performance/quality claim; SmolVLA causal claim remains D2 pending a corrected pinned CUDA rerun; 16 GiB ceiling preserved.

## Decisions applied

- Former 95% core / 90% overall breadth percentages are portfolio-health facts, not `0.9` or depth gates; none was converted into a pass or waiver.
- TCAV, SAE, steering, manifold/geometry, and other empirical failures are blockers only for their corresponding Sprint 80 diagnostic claims, not for the `0.9` pre-stable baseline.
- Unavailable real-policy/LeRobot overhead and heavyweight-model timing are secondary integration work with explicit non-claims, not `0.9` blockers.
- Sprint 80 is the depth gate; Sprint 81 owns stable publication. Old Sprint 80 stable-publication references were renumbered to Sprint 81.

## Changed paths

- `docs/EVIDENCE_LEDGER.md`
- `docs/EVIDENCE_GAP_PLAN.md`
- `docs/M14_REAL_SYSTEM_VALIDATION.md`
- `docs/MIGRATION.md`
- `docs/API_REFERENCE.md`
- `docs/PERFORMANCE.md`
- `docs/LEROBOT_INTEGRATION.md`
- `docs/INDEX.md`
- `docs/sprint-plans/sprint-78.md`
- `docs/sprint-plans/sprint-79.md`
- `README.md`
- `CHANGELOG.md`
- `artifacts/task_79_baseline_closure_595_598_600_summary.md` (this file)

## Intentionally unchanged

- `docs/PLAN.md`, `docs/sprint-plans/sprint-80.md`, `docs/sprint-plans/sprint-81.md`, `.agents/memory/decisions.md`: already encode the depth-first contract.
- `docs/evidence-ledger.json` and all M14/artifacts bytes: no promotion, deletion, or relabeling.
- Task 79.603 (`0.9.0` publication) was not performed.

## Verification

- Structural link check of edited docs performed by inspection; no project-wide validation run (main agent owns final validation).
