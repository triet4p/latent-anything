# Review remediation: contracts and provenance

## Scope

Fixed transition/rollout compatibility, metadata-aware caching, evaluator precedence, reward source identity, and the premature reward/value Protocols.

## Evidence

- `LatentTransition.mean_rollout` now declares the metadata mapping already required by every concrete implementation and `RolloutPipeline`.
- Rollout cache keys include caller metadata; a regression test proves different metadata cannot reuse the first trajectory.
- Explicit CEM/MPPI evaluators are invoked even when the pipeline already attached an evaluation.
- Reward evaluation requires trajectory `source_space_identity` to match the scorer.
- `_RewardScorer` and `_ValueEstimator` remain concrete-first classes until a third differing implementation exists.

## Verification

Focused rollout, reward, CEM, and MPPI tests pass.
