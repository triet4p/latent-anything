# Task Summary: Sprint 60 Task 3 — Capture points with token/modality metadata

**Sprint:** Sprint 60
**Task:** Define capture points for vision/language/state context and action-expert representations with token/modality metadata.

## Summary of Work

Four module seams on the official `select_action` path are captured per executed action query:

| Representation | Hook location | Token metadata |
| --- | --- | --- |
| Vision context | `model.vlm_with_expert.vlm.model.vision_model` (SigLIP) | patch-token count, camera name, prefix offset |
| Language context | `model.vlm_with_expert.vlm.model.text_model.embed_tokens` | token count, prefix offset |
| State context | `model.state_proj` | single token, prefix offset |
| Action expert | `model.vlm_with_expert.lm_expert.norm` | chunk-token count, denoising step |

`SmolVLATokenMetadata` records modality, token count, prefix offset (context kinds) or denoising step (action expert). Real-model verification produced 14 captures per query: 2×1024 vision tokens, 48 language tokens, 1 state token, 10 expert captures at 50 chunk tokens.

## Files Modified

* `src/latent_anything/integrations/lerobot_smolvla.py` — `SmolVLARepresentation`, `SmolVLATokenMetadata`, hook closures, capture metadata.

## Testing

* **Test File:** `tests/test_lerobot_smolvla.py::test_smolvla_capture_records_modalities_with_token_metadata`
* **Status:** Passed

## Additional Notes

Camera names follow `config.image_features` order intersected with the prepared batch, mirroring the upstream `prepare_images` iteration; queue hits execute no model query and produce no captures.
