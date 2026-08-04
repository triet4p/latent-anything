# Task Summary: Sprint 58 Task 4 — ACT policy adapter

**Sprint:** Sprint 58
**Task:** Implement an ACT-specific adapter using the shared capture lifecycle and policy metadata.

## Summary of Work

Implemented `ACTPolicyAdapter`, immutable checkpoint/capture metadata, queue-aware episode capture, and explicit conversion of the selected query to a read-only NumPy latent.

## Files Modified

* `src/latent_anything/integrations/lerobot_act.py` — adapter, capture result, episode trace, and latent-space metadata.

## Testing

* **Test File:** `tests/test_lerobot_act.py`
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_lerobot_act.py -q`

## Additional Notes

Queued actions remain valid outputs but have `representation=None` when no new decoder forward occurs.
