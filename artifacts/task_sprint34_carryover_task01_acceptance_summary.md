# Sprint 34 Carryover Task C34.1 — Acceptance Contract

**Status:** Complete

## Scope

Defined the offline held-out meaningful-integration acceptance contract before
benchmark implementation. Sprint 35 real-checkpoint work is explicitly out of
scope.

## Frozen evaluation design

- Dataset: `sklearn.datasets.load_digits`, normalized to `[0, 1]`, shape
  `(n, 1, 8, 8)`; the runtime package revision and BSD-3-Clause license are
  recorded in the generated artifact.
- Split: deterministic 80/20 index split using seed `42`; only the training
  partition may fit the ConvVAE, PCA, SAE, or steering direction.
- Baselines: all-zero image reconstruction is the primary simple baseline;
  the training-pixel-mean reconstruction is retained as a stronger diagnostic.
- Hard gates: finite held-out metrics; at least 10% held-out MSE improvement
  over the all-zero baseline; latent utilization at least `1e-3`; finite,
  correctly shaped held-out PCA/SAE outputs; finite unit-norm steering direction
  and finite decoded steering effect.
- Runtime: record wall-clock CPU runtime against a 30-second advisory budget;
  runtime does not override a failed quality gate.

## Files

- `docs/PLAN.md` — active carryover status and contract.
- `docs/sprint-plans/sprint-34.md` — atomic carryover tasks and thresholds.

## Validation

- `git diff --check` — passed after the plan and artifact changes.
- `graphify update .` — run immediately after this task; result is recorded
  below.

## Graph refresh

- Command: `graphify update .`
- Result: completed with 10,046 nodes, 19,541 edges, and 884 communities in
  `graphify-out/GRAPH_REPORT.md`; graphify reported 46 zero-node JSON/source
  warnings (known `#1666` behavior) and retried them on the refresh.
