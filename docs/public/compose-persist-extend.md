# Compose, persist, and extend a pipeline

Use this guide to choose among the released `AnalysisPipeline`, `InterventionPipeline`, and `RolloutPipeline`, build a small analysis from registry configuration, and persist its result without confusing a serialization round-trip with a reproducible model run. These are three concrete execution stories, not a general-purpose workflow engine.

## Choose by input and lifecycle

| Pipeline | Input and lifecycle | Result | Choose it when… |
| --- | --- | --- | --- |
| Analysis | One data batch → `adapter.encode(data)` → `method.fit(latents)` → `method.transform(latents)` on that same batch. | `PipelineResult(latents, transformed, latent_space)` | You want to encode examples and fit/transform a Layer A analysis method such as PCA. |
| Intervention | A method-specific call: `run_data(data)` or `run_trajectory(trajectory)`. `fit()` delegates to stateful methods. Data execution with `apply_latent()` uses encode → edit → decode; otherwise the method receives the data directly. | Data route: `numpy.ndarray`. Trajectory route: `numpy.ndarray` or `Trajectory`, as defined by the method. | You want to apply an intervention to a compatible data batch or latent trajectory. |
| Rollout | One initial latent point plus an ordered action matrix `(horizon, action_dim)` → predictive-mean transition rollout. | `RolloutResult`; its trajectory has the initial state followed by one predicted state per action. | You want to evaluate an existing latent transition over an action sequence. |

`InterventionPipeline` is the released public name for the exact `ManipulationPipeline` alias. Its shared metadata value remains `pipeline_kind == "manipulation"`. It needs a `FlatBatchDecodableAdapter` for `run_data()`, even when the wrapped method acts directly on data; it does not install arbitrary model-forward hooks. `run_trajectory()` instead delegates to the method's trajectory operation.

The shared `PipelineContract` is deliberately small: it reports pipeline kind and an associated latent space when available. The three stories do **not** share a generic `run()` signature. Their input axes, fitting lifecycle, and result meaning are different, so keep calls and results specific to the selected pipeline.

## Prerequisites and a bounded CPU example

Use Python `>=3.12,<3.15` and the released base package:

```bash
python -m pip install "latent-anything==1.0.0"
```

The example uses only the base package, NumPy arrays, the built-in fixed-random `HiddenStateAdapter`, and PCA. It generates 12 four-feature rows in memory; it does not train a model, download a checkpoint, or require an optional extra. `HiddenStateAdapter` is a small synthetic feature stack, not a wrapper for a user's pretrained model.

```python
from __future__ import annotations

from tempfile import TemporaryDirectory

import numpy as np

from latent_anything import (
    FileSystemRunRecorder,
    ObjectSpec,
    PipelineSpec,
    RuntimeProfiler,
    build_pipeline_from_config,
    decode_result_envelope,
    encode_result_envelope,
)


spec = PipelineSpec(
    adapter=ObjectSpec(
        kind="adapter",
        name="hidden_state",
        params={"input_dim": 4, "hidden_dim": 3, "random_state": 17},
    ),
    method=ObjectSpec(
        kind="analysis",
        name="pca",
        params={"n_components": 2},
    ),
)
pipeline = build_pipeline_from_config(spec)
data = np.arange(48, dtype=np.float64).reshape(12, 4) / 47.0
profiler = RuntimeProfiler()
result = pipeline.run(data, profiler=profiler)

assert result.latents.shape == (12, 3)
assert result.transformed.shape == (12, 2)
print(f"PipelineResult: latents={result.latents.shape}, transformed={result.transformed.shape}")
print(f"Profile stages: {sorted(profiler.snapshot().stage_totals())}")

# The typed envelope stores declared values and explicit identity metadata.
config = spec.model_dump(mode="json")
payload = encode_result_envelope(
    result,
    provenance={"adapter": "hidden_state", "input": "arange-48-v1"},
    behavior_state={"pipeline_config": config, "seed": 17},
)
round_trip = decode_result_envelope(payload).value
assert type(round_trip) is type(result)
assert np.array_equal(round_trip.latents, result.latents)

# A run record is explicit; pipelines do not create one automatically.
with TemporaryDirectory() as root:
    recorder = FileSystemRunRecorder(root)
    run = recorder.start(
        "hidden-state-pca",
        config=config,
        model_revisions={"adapter": "hidden_state-random-state-17"},
        dataset_revisions={"input": "arange-48-v1"},
        seeds=(17,),
        runtime_profile=profiler.snapshot(),
    )
    artifact = recorder.add_portable_artifact(
        run.run_id,
        payload,
        name="analysis-result",
        artifact_type="result-envelope-v1",
    )
    stored = recorder.read_portable_artifact(artifact)
    restored = decode_result_envelope(stored.payload).value
    assert type(restored) is type(result)
    completed = recorder.complete(run.run_id, metrics={"samples": float(data.shape[0])})
    print(f"Run record: {completed.status}; artifacts={len(completed.artifacts)}")
```

Expected output:

```text
PipelineResult: latents=(12, 3), transformed=(12, 2)
Profile stages: ['encode', 'method']
Run record: completed; artifacts=1
```

This checks config-based construction, output shapes, two observed profile stages, and a typed result round-trip stored as a portable run artifact. It is a tiny CPU wiring demonstration, not evidence about a trained model, task performance, or real-world reproducibility.

## Registry configuration and plugin discovery

`ObjectSpec` describes one named registry object: a canonical `kind`, a non-empty registered `name`, and keyword `params` for its factory. Use the released registry kinds `adapter`, `analysis`, `intervention`, and `runtime`; avoid the retained `method_a` / `method_b` compatibility spellings in new configs. `PipelineSpec` plus `build_pipeline_from_config()` composes an adapter and analysis method. Intervention and rollout have their own spec/build functions; they are not nested inside an invented generic pipeline language. A caller-owned `Registry` can be passed to config builders when names should be isolated from the global built-ins.

External plugins use Python entry points. The groups `latent_anything.adapter`, `.analysis`, and `.intervention` map to those registry kinds; `.transition` and `.planner` are capability groups whose current entries live under `runtime`. `list_entry_points()` reads metadata without importing targets. `load_entry_points(registry=...)` is an explicit operation: it imports plugin code, requires the callable to declare `__latent_anything_plugin_api_version__ = "1"`, and reports isolated failures and duplicates. Listing is not loading, but loading is **not sandboxed** and can execute arbitrary third-party import/factory code. Install and load only plugins you trust; inspect the discovery report and retain plugin distribution/version provenance.

This config API builds individual registry objects. It is not a remote execution protocol, dependency-injection framework, or general DAG/workflow definition.

## Batching, cache, profiling, async, and streaming boundaries

These capabilities are separate concrete runtime tools rather than automatic stages that every pipeline applies:

- **Batching:** `BatchExecutor(batch_size)` explicitly splits NumPy arrays along the first/sample axis for encode, decode, transform, or a supplied array operation, then eagerly concatenates outputs in input order. `AnalysisPipeline.run()` itself encodes and fits the complete supplied batch; it does not silently chunk a method fit. Batching therefore does not make a global fit equivalent to independent per-chunk fits, nor does it provide streaming output.
- **Cache:** `AnalysisPipeline(cache=InMemoryCache())` reuses adapter-encoding arrays; it still fits and transforms the method on each run. `RolloutPipeline(cache=...)` caches completed mean-rollout payloads and marks hits on `RolloutResult.cache_hit`. The cache is process-local and returns defensive copies. `InterventionPipeline` has no cache argument. `RolloutPipeline.stream()` intentionally bypasses both in-memory rollout caching and run-record persistence. Use the separate disk-cache API for validated portable cache entries, not as an archival artifact store.
- **Profiling:** Pass a `RuntimeProfiler` to a supported call and inspect `snapshot()` / `stage_totals()`. Events record elapsed time for concrete stages (for example, encode and method, intervention encode/method/decode, or rollout transition/cache/evaluation). This is stage timing, not a memory profiler, throughput guarantee, or model-quality measure.
- **Async:** The released async counterparts wrap the same synchronous work in worker threads (`run_async()`, intervention `run_data_async()` / `run_trajectory_async()`, and `BatchExecutor` async methods). They are useful at an async boundary; they do not promise parallel CPU speedup or provide a separate backend. `stream_async()` is an async generator, not a coroutine-returning `run()` method; an in-flight Python operation must settle before cancellation can be observed.
- **Streaming:** `RolloutPipeline.stream()` / `stream_async()` consume disjoint, ordered, exact `numpy.ndarray` action chunks of shape `(rows, action_dim)`. `max_chunk_rows` bounds each chunk (default `1024`); the source is advanced only after that chunk completes. The first chunk starts from the initial state, but yielded trajectories contain only predicted rows; concatenating them matches the eager rollout without its initial row. Empty chunks are skipped. Streaming predicts means only, does not overlap windows or prefetch, and does not retain or persist a full result. Close an async generator explicitly if you stop consuming it early.

Optional provider integrations are a different boundary from plugins. The base package does not import Diffusers, Transformers, gsplat, or LeRobot; install the specific published optional extra only when you need that integration. An extra provides dependencies, not a guarantee for arbitrary checkpoints, models, hardware, or cloud services. The CPU example above does not use any optional extra.

## Run records, typed envelopes, and what round-tripping proves

`FileSystemRunRecorder` explicitly records a run's config, code/framework versions, model and dataset revisions, seeds, environment, metrics, profile, links, and artifacts. `start()` creates or reuses a record for its declared reproducible-input identity; `complete()` records final metrics and artifacts, while `fail()` and interrupted-run recovery describe other lifecycle outcomes. Artifact bytes are content-addressed; `add_portable_artifact()` wraps a typed payload in the filesystem artifact envelope and attaches its reference. A pipeline does not start or complete a run for you.

There are two related, distinct formats:

- `encode_portable()` / `decode_portable()` use `portable-node-v1` for supported NumPy/domain values without pickle. Arrays and metadata are bounded and validated; object-dtype arrays, cycles, unsupported values, and malformed payloads are rejected.
- `encode_result_envelope()` / `decode_result_envelope()` use `result-envelope-v1` for an explicit allowlist of typed results and config models. The decoder never imports a class named by untrusted artifact data. Provenance and behavior-affecting state contribute to the envelope identity; identity mismatch, unsupported types, unknown versions, or invalid payloads fail closed. The reader explicitly upgrades only the documented `result-envelope-v0` shape before the same validation. `FileSystemRunRecorder.add_portable_artifact()` adds a separate `artifact-envelope-v1` integrity/checksum wrapper around those bytes.

Writers emit current format versions. Run-record readers accept the explicitly supported pre-versioned record/path migration; `disk-cache-v1` is derived cache state and has no archive migration promise. These boundaries and exact migration rules are in the support policy.

A successful decode proves that supported data and declared types passed the reader's structural, version, and integrity checks. It does **not** restore plugin code, model weights, datasets, preprocessing, dependency versions, or execution hardware, and it does not prove that rerunning the model will reproduce an output. Preserve immutable revisions, seeds, config, and trusted code separately; never use successful deserialization as a substitute for experimental reproducibility.

## Specialist references and released contract

- [Pipeline story contracts, rollout streaming, and planner composition](https://github.com/triet4p/latent-anything/blob/main/docs/PIPELINES.md)
- [External plugin groups, config construction, compatibility, and trust boundary](https://github.com/triet4p/latent-anything/blob/main/docs/PLUGIN_AUTHOR_GUIDE.md)
- [Portable values, typed envelopes, artifact storage, and SQLite cache](https://github.com/triet4p/latent-anything/blob/main/docs/PORTABLE_ARTIFACTS.md)
- [Optional integration and import-isolation boundaries](https://github.com/triet4p/latent-anything/blob/main/docs/OPTIONAL_INTEGRATIONS.md)
- [Support, versioning, and artifact-reader migration policy](https://github.com/triet4p/latent-anything/blob/main/docs/SUPPORT_POLICY.md)
- [Frozen `1.0.0` API reference](https://github.com/triet4p/latent-anything/blob/v1.0.0/docs/API_REFERENCE.md) and [API-freeze snapshot](https://github.com/triet4p/latent-anything/blob/v1.0.0/artifacts/api_freeze_snapshot_1.0.0.json)
