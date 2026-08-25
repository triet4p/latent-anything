# Tokenized world model

Sprint 72 adds a compact offline reference lane for tokenized dynamics. A
fitted `VQVAE` remains frozen as the observation tokenizer and decoder. The
`TokenizedWorldModel` trains an action-conditioned GRU encoder/decoder with a
cross-entropy next-frame objective and generates each frame token by token.

The public token representation stays categorical:

```python
tokens = model.encode(images)                 # integer code IDs
prediction = model.predict_next(tokens, action)
trajectory = model.rollout(tokens[0], actions, sampling="sample", seed=72)
```

The class conforms to the existing adapter and mean-transition seams. Greedy
`step` and `mean_rollout` are the behavior consumed by `RolloutPipeline`;
seeded sampling, padding, codebook-version checks, likelihood diagnostics,
and teacher-forced/free-running comparison remain concrete tokenized-model
behavior.

The reproducible benchmark is:

```text
uv run python scripts/tokenized_world_model_benchmark.py
```

It writes `artifacts/tokenized_world_model_evidence.json` and its matching
configuration. The artifact is D2 synthetic CPU evidence. It reports
teacher-forced perplexity, code usage, dead-code health, greedy and sampled
drift by horizon, decoder consistency, a task proxy, seeded parity, and
failure-horizon fields. The fitted compact tokenizer passes the non-trivial
usage gate, but greedy free-running error appears at the first held-out
horizon; this sprint does not claim healthy large-scale VQ, a real checkpoint,
or real-world world-model performance.
