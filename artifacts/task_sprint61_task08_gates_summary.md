# Task Summary: Sprint 61 Task 8 — D3 promotion, ADR/changelog/artifact/gates

**Sprint:** Sprint 61
**Task:** Promote ACT/Diffusion/SmolVLA claims to D3 only where evidence passes; update ADR/changelog/artifact/gates.

## Summary of Work

Ran the CUDA statistical lane on the remote server (trietlm@di-server, RTX 4060 Ti, LeRobot 0.6.1) through the remote-cuda-test workflow from the pushed commit `f2654f2`. The acceptance gate passed on the pinned public SmolVLA pair against `libero_spatial` (seeds 1–3, strengths 1/5/10, 30 episodes): baseline bit-exact, no-hook success equality, measurable intervention action changes, all episodes within budget. The committed artifact demonstrates all three causal outcomes: baseline/no-hook 100% success; random leaves behavior unchanged at every strength; the targeted intervention leaves behavior unchanged at strength 1 (offline on-target 0.86, success delta 0.00 — recorded overstatement disagreement) and harms success from 1.0 to 0.0 at strengths 5 and 10 (recorded reversal disagreements; all six episodes max out at 280 steps). ACT and Diffusion have no intervention surface, so their claims remain observational. `THY-T05-CAUSAL-INTERVENTION-VS-OBSERVATIONAL-STUDY` was promoted to D3 in `docs/evidence-ledger.json` (source/test/benchmark/config/artifact roles), and the ADR, changelog, `docs/EVIDENCE_LEDGER.md`, `docs/PLAN.md`, sprint plan, lessons, CI lane, and the three reproducible artifacts (JSON, config, PNG) were updated.

## Files Modified

* `docs/evidence-ledger.json`, `docs/EVIDENCE_LEDGER.md` — D3 promotion for the causal-intervention capability.
* `artifacts/smolvla_simulation_benchmark.json`, `artifacts/smolvla_simulation_benchmark_config.json`, `artifacts/smolvla_simulation_benchmark.png` — real-model reproducible artifacts (provenance: remote CUDA lane at `f2654f2`, `di-server`).
* `.agents/memory/decisions.md`, `.agents/memory/lessons-learned.md`, `CHANGELOG.md`, `docs/PLAN.md`, `docs/sprint-plans/sprint-61.md`, `docs/LEROBOT_INTEGRATION.md` — gates and records.

## Testing

* **Status:** Passed — statistical lane `test_smolvla_simulation_statistical_benchmark` PASSED on CUDA (11/11), offline suite passed locally, ruff/pyright/ledger clean.
* **Execution Command:** `LATENT_ANYTHING_RUN_NETWORK=1 uv run --extra lerobot-smolvla --extra 3d --extra viz pytest tests/test_lerobot_benchmark.py -v` (remote)

## Additional Notes

The remote flow surfaced three environment/protocol issues that were fixed in-sprint and logged as lessons: LIBERO advances its initial-state index per reset (fresh env per cell required), the env factory keys suites by the LIBERO suite name, and `hf-libero` prompts on stdin at import when `~/.libero/config.yaml` is missing or stale (non-interactive bootstrap added).
