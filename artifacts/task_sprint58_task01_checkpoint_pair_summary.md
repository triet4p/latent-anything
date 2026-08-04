# Task Summary: Sprint 58 Task 1 — Pinned ACT checkpoint pair

**Sprint:** Sprint 58
**Task:** Select a public ACT checkpoint/dataset pair and pin both revisions.

## Summary of Work

Pinned `lerobot/act_aloha_sim_insertion_human` at policy revision `33259aa86eb45fdf85350280044a33d9d50e40c3` and its training dataset `lerobot/aloha_sim_insertion_human` at dataset revision `cc571a3c661df81b566dbfde3d5c1e85fcdf7884` in `ACTCheckpointSpec`.

## Files Modified

* `src/latent_anything/integrations/lerobot_act.py` — checkpoint constants and immutable specification.
* `docs/LEROBOT_INTEGRATION.md` — public pair and reproduction documentation.

## Testing

* **Test File:** `tests/test_lerobot_act.py`
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_lerobot_act.py -q`

## Additional Notes

The revisions were resolved from the public Hugging Face repositories. The pair is also covered by an opt-in network smoke test.
