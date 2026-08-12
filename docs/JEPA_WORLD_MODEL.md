# JEPA/LeWM World Model

Sprint 71 adds `JEPAWorldModelAdapter`, a compact CPU-trainable reference
implementation of the decoder-free JEPA/LeWM training shape:

`context_encoder(observation_t), action_t -> predictor -> target_encoder(observation_t+1)`

The target branch runs under stop-gradient and is updated by EMA. The adapter
implements `ModelAdapter` and the shared mean `LatentTransition` surface, but
intentionally does not implement `DecodableAdapter` and has no `decode` method.

## Reproduce the evidence

```text
uv run python scripts/jepa_world_model_benchmark.py
```

This writes `artifacts/jepa_world_model_evidence.json` and its pinned config.
The benchmark uses `compact-jepa-lewm-v1` and
`synthetic-controlled-latent-dynamics-v1`, with held-out one-step prediction,
collapsed-baseline comparison, latent variance/covariance health, and masked
open-loop horizon drift. It is synthetic CPU D2 evidence; the artifact keeps
the observed anisotropy and rollout drift visible.

## Pipeline and records

Use `jepa_world_model` for analysis configuration and `jepa_transition` for
`RolloutPipelineSpec`. A fitted adapter can be passed directly to
`AnalysisPipeline` and `RolloutPipeline`. `FileSystemRunRecorder` provides
`complete_jepa_evaluation()` to store the typed prediction/rollout report as a
content-addressed JSON artifact.

## Public checkpoint smoke

`tests/test_latent_anything/test_jepa_checkpoint.py` is marked
`network`/`large_download` and pins `facebook/ijepa_vith14_1k` to revision
`be440b1cac639542ae553e71a9c7afd925ab5fac`. Run it only with the explicit
network environment and the `transformers` extra; it is a structural
checkpoint smoke, not a claim that the compact vector adapter is an image
checkpoint loader.
