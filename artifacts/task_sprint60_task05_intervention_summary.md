# Task Summary: Sprint 60 Task 5 — Bounded action-expert intervention

**Sprint:** Sprint 60
**Task:** Add one bounded intervention with safe hook lifecycle, strength control, and no-change identity behavior.

## Summary of Work

`SmolVLAIntervention` is one additive, bounded intervention on the action-expert representation (`model.vlm_with_expert.lm_expert.norm` output): `z <- z + strength * direction` at every denoising step. It validates direction shape/finiteness and a bounded `|strength| <= max_strength`; strength zero short-circuits and returns the unchanged output, so baseline and intervened actions are bit-identical. Hooks are registered per `select_action` call inside `_SmolVLAHookSession` (context manager) and always removed, even when the policy forward raises.

## Files Modified

* `src/latent_anything/integrations/lerobot_smolvla.py` — `SmolVLAIntervention`, `_SmolVLAHookSession`, expert hook integration.

## Testing

* **Tests:** `test_smolvla_intervention_strength_zero_is_bit_exact_identity`, `test_smolvla_intervention_is_bounded_and_validated`, `test_smolvla_hook_session_removes_hooks_after_policy_exception`
* **Status:** Passed
* **Real verification:** intervention on the real checkpoint changed the action (relative change ~0.6% at strength 2.0 with a 0.01/dim direction).

## Additional Notes

The hook captures the post-intervention expert tensor, so representation drift (Task 6) measures the representation the policy actually consumed.
