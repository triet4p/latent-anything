# Sprint 77 Phase A Task 05 — closure gates and owner recommendation

Status: historical Phase-A closure snapshot (2026-08-25). The owner-approved
Phase-B Rust/PyO3 deferral is recorded in
`artifacts/task_sprint77_phase_b_task01_rust_deferral_summary.md`; later
Phase-B closure validation and cumulative audit passed as recorded in the
Phase-B task artifacts.

## Final evidence

- Workload/harness artifact: `artifacts/sprint77_phase_a_benchmark.json`
  (13 fixed offline cases, seed 771, Windows 11 / Python 3.13.3 / NumPy 2.4.6
  / Torch 2.10.0 / PyArrow 24.0.0, two warmups, eight repetitions).
- Profile artifact: `artifacts/sprint77_phase_a_profile.json`.
- Before/after comparison:
  `artifacts/sprint77_phase_a_comparison.json`; Euclidean DTW median improved
  38,297.3 → 27,478.2 µs (-28.25%) with unchanged digest.
- Proposed budgets and hard/advisory gate policy: `docs/PERFORMANCE.md`.
- No real LeRobot policy, remote provider, network, GPU, or Rust evidence is
  claimed. The LeRobot case is the existing offline captured-latent boundary.

## Exact commands and results

```text
uv run pytest -q tests/test_dtw.py tests/test_sprint77_phase_a_benchmark.py tests/test_sprint77_phase_a_profile.py tests/test_sprint77_phase_a_compare.py
13 passed

uv run pytest -q
1491 passed, 32 skipped, 39 warnings in 217.69s (0:03:37)

uv run ruff check src tests scripts/sprint77_phase_a_benchmark.py scripts/sprint77_phase_a_profile.py scripts/sprint77_phase_a_compare.py
PASS: All checks passed
uv run ruff format --check src tests scripts/sprint77_phase_a_benchmark.py scripts/sprint77_phase_a_profile.py scripts/sprint77_phase_a_compare.py
PASS: 208 files already formatted
uv run pyright
PASS: 0 errors, 0 warnings, 0 informations
uv run python scripts/validate_evidence_ledger.py
PASS: inventory 107; core 23/63 (36.5%); overall 23/65 (35.4%)
uv run --extra docs mkdocs build --strict
PASS: Documentation built in 30.42 seconds after removing an invalid root-doc nav reference; upstream Material-for-MkDocs warning is informational
git diff --check
PASS (only normal CRLF conversion warnings from git)
uv run python -c '...run_suite twice...'
PASS: semantic reproducibility, 13 correctness digests
```

The repository-wide `uv run ruff check .` was also run as a diagnostic: it
remains non-zero with 1,920 pre-existing `.agents`, theory, and notebook
findings outside the changed source/test/script scope. The changed-file and
source/test/script gate above is the applicable Phase-A review gate; those
baseline findings were not hidden or edited.

## Recommendation for owner decision

The evidence supports deferring a Rust implementation for pre-stable work: the
only clear Python hotspot (Euclidean DTW point-cost dispatch) was already
reduced by a semantics-preserving NumPy path, while geodesic time is a bounded
finite-difference algorithm and portable/cache/recorder time is dominated by
PyArrow/SQLite/filesystem work. The Phase-B decision records this deferral and
its future reconsideration conditions; it does not permanently prohibit a
cross-language experiment.

## Scope confirmation

Sprint 77 Phase A and the Rust/PyO3 deferral are complete. Phase-B closure
validation remains; Sprint 77 is not marked complete. Sprint 78/Milestone 14
and carryover gates were not started.

## Graphify trace

```text
graphify update .
PASS; 10,011 nodes / 19,512 edges / 898 communities; 46 zero-node JSON/source warnings.
```
