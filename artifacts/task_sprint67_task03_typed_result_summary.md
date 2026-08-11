# Task Summary: Sprint 67 Task 3 — Typed evaluation result

**Sprint:** Sprint 67
**Task:** Return rewards, returns, values, masks, uncertainty, and provenance in a typed result.

## Summary of Work

Added immutable `RewardValueEvaluationResult` with per-step rewards, returns, values, masks, terminal flags, reward/value uncertainty, Bellman residuals, discount/horizon, source, and provenance serialization.

## Files Modified

* [src/latent_anything/reward_value.py](/F:/ai-ml/latent-anything/src/latent_anything/reward_value.py) - Typed result and serialization.
* [src/latent_anything/pipeline_models.py](/F:/ai-ml/latent-anything/src/latent_anything/pipeline_models.py) - Optional rollout evaluation attachment.

## Testing

* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_reward_value.py -q`

## Additional Notes

Arrays are defensively copied and made read-only at the result boundary.
