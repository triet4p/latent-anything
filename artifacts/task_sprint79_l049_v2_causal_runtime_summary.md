# Task Summary: Sprint 79 L04.9 v2 causal runtime correction

**Sprint:** Sprint 79
**Task:** L04.9 v2 Stage A causal scoring and raw GPT-2 capture

## Summary of Work

Added a private raw-block capture seam for the concrete
`TransformerLMIntegration` path. Stage A now uses pre-`ln_f` GPT-2 block
outputs, independently resolves clean/corrupt endpoint positions, and scores
each row with signed directional recovery from primitive clean, corrupted, and
patched margins. The validator recomputes those primitives. Historical D0
assessment sidecars and all evidence bytes were left unchanged.
The Stage A artifact schema is versioned to v2 for the new primitive-margin
records.

## Files Modified

* [src/latent_anything/_transformer_runtime.py](/F:/ai-ml/latent-anything/src/latent_anything/_transformer_runtime.py) - private raw block capture, intervention targeting, and shape validation.
* [src/latent_anything/integrations/transformer_lm.py](/F:/ai-ml/latent-anything/src/latent_anything/integrations/transformer_lm.py) - private raw-capture integration seam and native-index documentation.
* [scripts/_m14_l049_v2_real_runtime.py](/F:/ai-ml/latent-anything/scripts/_m14_l049_v2_real_runtime.py) - raw source/recipient direction and paired Stage A/Stage B margins.
* [scripts/_m14_l049_v2_stage_a.py](/F:/ai-ml/latent-anything/scripts/_m14_l049_v2_stage_a.py) - primitive recovery serialization and group aggregation.
* [scripts/_m14_l049_v2_validate_stage_a.py](/F:/ai-ml/latent-anything/scripts/_m14_l049_v2_validate_stage_a.py) - independent primitive recovery validation.
* [scripts/_m14_l049_v2_stage_b.py](/F:/ai-ml/latent-anything/scripts/_m14_l049_v2_stage_b.py) - Stage B candidate-schema binding to the revised Stage A schema.
* [tests/test_transformer_lm.py](/F:/ai-ml/latent-anything/tests/test_transformer_lm.py) - terminal `ln_f`, variable-length, tuple/list, and cleanup regressions.
* [tests/test_m14_l049_v2.py](/F:/ai-ml/latent-anything/tests/test_m14_l049_v2.py) - primitive recovery and rehashed tamper regressions.
* [CHANGELOG.md](/F:/ai-ml/latent-anything/CHANGELOG.md), [docs/sprint-plans/sprint-79.md](/F:/ai-ml/latent-anything/docs/sprint-plans/sprint-79.md), [.agents/memory/lessons-learned.md](/F:/ai-ml/latent-anything/.agents/memory/lessons-learned.md) - truthful implementation record.

## Testing

* **Focused status:** Passed (`165 passed` for transformer, v2, and remote-postprocess tests).
* **Execution:** `uv run pytest -q tests/test_transformer_lm.py tests/test_m14_l049_v2.py tests/test_m14_l04_remote_postprocess.py`
* **Static status:** Ruff and Pyright strict passed for the project.

## Additional Notes

No network, remote CUDA, holdout access, retention, finalization, deletion,
commit, or push was performed. This is an implementation-only correction; it
does not claim a new real-CUDA result or evidence promotion.
