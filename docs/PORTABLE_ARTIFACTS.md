# Portable artifacts and disk cache

Sprint 74 adds an offline, versioned representation for NumPy-facing latent
values and typed results. The public API returns ordinary NumPy/domain objects;
PyArrow is an implementation dependency and does not leak through these
boundaries.

## Value nodes

```python
from latent_anything import decode_portable, encode_portable

payload = encode_portable(latent_value)
restored = decode_portable(payload)
```

The `portable-node-v1` Arrow IPC schema stores one binary row per NumPy array
with explicit `array_id`, `dtype`, `shape`, and `payload` columns. A
canonical JSON manifest is held in Arrow schema metadata. Dtypes, shapes, byte
order, immutable `LatentValue`/`Trajectory` construction, nested string-key
mappings, and bounded sequences are restored without pickle.

The codec rejects object-dtype arrays, cycles, unsupported classes,
non-finite metadata floats, malformed payload lengths, and values over the
configured `PortableLimits`. Decode untrusted bytes with conservative limits.

## Typed envelopes

`encode_result_envelope` and `decode_result_envelope` implement
`result-envelope-v1` for an explicit built-in allowlist of typed planning and
analysis results plus configuration models. The decoder never imports a type
named by an artifact. Provenance and behavior-affecting state are required
metadata inputs to the envelope identity; a tampered identity fails closed.
An explicit local migration hook accepts the narrowly supported
`result-envelope-v0` shape and upgrades it to v1 before the same allowlist
and identity checks; unknown versions fail closed.

```python
from latent_anything import decode_result_envelope, encode_result_envelope

payload = encode_result_envelope(
    plan,
    provenance={"plugin": "example", "version": "1.0"},
    behavior_state={"config_identity": "cfg-1", "checkpoint_identity": "ckpt-1"},
)
envelope = decode_result_envelope(payload)
plan_again = envelope.value
```

Do not cache a fitted or state-mutating output unless `behavior_state` captures
every state field required to reproduce behavior.

## Filesystem envelopes and run records

`ArtifactStore` wraps portable bytes in `artifact-envelope-v1` with canonical
metadata, SHA-256 payload checksum, identity, and size. Writes are same-folder
fsynced and atomically replaced. Relative paths, traversal, symlink
components, oversized headers/files, truncated data, schema mismatches, and
checksum mismatches are rejected. Existing symlink, junction, and Windows
reparse-point roots/components are rejected and checked again after directory
creation. These portable `pathlib` checks cannot eliminate a hostile process
that swaps a path between validation and open/replace; use a private directory
with appropriate OS permissions when the filesystem is adversarial.
`FileSystemRunRecorder.add_portable_artifact`
attaches the resulting envelope through the existing content-addressed
`ArtifactRef` contract and preserves plugin/config/checkpoint provenance.

## Disk cache

`SQLiteDiskCache` is a concrete stdlib SQLite backend using WAL and bounded,
deterministic LRU eviction. `make_disk_cache_key` hashes the existing runtime
`CacheKey` with required non-empty plugin, checkpoint, and behavior-state
identities. The low-level `set/get` methods store opaque bytes for cache
internals; framework artifacts must use `set_portable/get_portable`, which
validates the Arrow envelope before caching and on restore. It adds no cache
Protocol and no pickle format.

```python
from latent_anything import CacheKey, SQLiteDiskCache, make_disk_cache_key

key = make_disk_cache_key(
    CacheKey("analysis", "encode", "demo", config_hash, state_hash, data_hash, "0.1"),
    plugin_identity="demo@1.0",
    checkpoint_identity="ckpt-1",
    behavior_state_identity="state-1",
)
cache = SQLiteDiskCache("cache.sqlite")
cache.set_portable(key, payload)
restored_payload = cache.get_portable(key)
```

Cross-process parity is demonstrated by
`python scripts/sprint74_portable_roundtrip.py`; the offline CPU size/latency
comparison is `python scripts/sprint74_artifact_benchmark.py`. These are
reproducibility lanes, not real-model or CUDA evidence.
