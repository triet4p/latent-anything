# Task Summary: Sprint 58 Task 3 — ACT capture-point selection

**Sprint:** Sprint 58
**Task:** Identify one semantically justified encoder/decoder/action-query capture point.

## Summary of Work

Selected `model.decoder` and retained its first query token. LeRobot 0.6.1 source shows this decoder output is transposed and passed directly to `action_head`; query index zero therefore corresponds to the first selected action.

## Files Modified

* `src/latent_anything/integrations/lerobot_act.py` — capture-point constant and first-query extraction.
* `docs/LEROBOT_INTEGRATION.md` — source-backed rationale and shape contract.
* `.agents/memory/decisions.md` — architectural decision and deferred causal scope.

## Testing

* **Test File:** `tests/test_lerobot_act.py`
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_lerobot_act.py -q`

## Additional Notes

The adapter captures only observational representations. No intervention is applied.
