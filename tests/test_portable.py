"""Focused Sprint 74 Task 01 tests for the versioned Arrow node layer."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pyarrow as pa  # pyright: ignore[reportMissingTypeStubs]
import pytest

from latent_anything import LatentSpace, LatentValue, Trajectory
from latent_anything.portable import PortableLimits, PortableNodeError, decode_portable, encode_portable


def _replace_column(payload: bytes, name: str, values: list[object]) -> bytes:
    table = pa.ipc.open_file(pa.BufferReader(payload)).read_all()
    index = table.schema.get_field_index(name)
    replacement = pa.array(values, type=table.schema.field(name).type)
    mutated = table.set_column(index, table.schema.field(name), replacement)
    output = BytesIO()
    with pa.ipc.new_file(output, mutated.schema) as writer:
        writer.write_table(mutated)
    return output.getvalue()


def test_numpy_nodes_preserve_dtype_shape_endianness_and_nested_metadata() -> None:
    values = np.array([1, 2, 3], dtype=">i4")
    payload = encode_portable({"array": values, "nested": ("ok", b"bytes")})

    restored = decode_portable(payload)

    assert isinstance(restored, dict)
    restored_array = restored["array"]
    assert isinstance(restored_array, np.ndarray)
    assert restored_array.dtype.str == values.dtype.str
    assert restored_array.shape == values.shape
    np.testing.assert_array_equal(restored_array, values)
    assert restored["nested"] == ("ok", b"bytes")


def test_latent_value_and_trajectory_restore_immutable_domain_objects() -> None:
    space = LatentSpace(dim=2, source_model="demo", metadata={"revision": "r1"})
    value = LatentValue(np.array([1.0, 2.0], dtype=np.float32), space, {"tags": ["x"]})
    trajectory = Trajectory(np.arange(6, dtype=np.float32).reshape(3, 2), metadata={"source": "demo"})

    restored_value = decode_portable(encode_portable(value))
    restored_trajectory = decode_portable(encode_portable(trajectory))

    assert isinstance(restored_value, LatentValue)
    assert restored_value.identity == value.identity
    np.testing.assert_array_equal(restored_value.to_numpy(), value.to_numpy())
    assert isinstance(restored_trajectory, Trajectory)
    np.testing.assert_array_equal(restored_trajectory.to_numpy(), trajectory.to_numpy())
    with pytest.raises(TypeError):
        restored_value.metadata["new"] = "blocked"  # type: ignore[index]


def test_resource_limits_reject_cycles_and_oversized_arrays() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(PortableNodeError, match="cyclic"):
        encode_portable(cyclic)

    with pytest.raises(PortableNodeError, match="per-array"):
        encode_portable(np.zeros(10, dtype=np.float64), limits=PortableLimits(max_array_bytes=8))


def test_object_arrays_are_rejected_without_pickle_fallback() -> None:
    with pytest.raises(PortableNodeError, match="object-dtype"):
        encode_portable(np.array([{"not": "portable"}], dtype=object))


def test_decode_rejects_input_manifest_rows_rank_and_malformed_dtypes_before_restore() -> None:
    payload = encode_portable({"array": np.zeros((1, 1), dtype=np.float32)})
    with pytest.raises(PortableNodeError, match="input"):
        decode_portable(payload, limits=PortableLimits(max_input_bytes=len(payload) - 1))
    with pytest.raises(PortableNodeError, match="manifest"):
        decode_portable(payload, limits=PortableLimits(max_manifest_bytes=1))
    with pytest.raises(PortableNodeError, match="rank"):
        decode_portable(payload, limits=PortableLimits(max_shape_rank=1))
    with pytest.raises(PortableNodeError, match="dtype"):
        decode_portable(_replace_column(payload, "dtype", ["not-a-dtype"]))
    with pytest.raises(PortableNodeError, match="object-dtype"):
        decode_portable(_replace_column(payload, "dtype", ["O"]))

    two_arrays = encode_portable({"a": np.zeros(1), "b": np.ones(1)})
    with pytest.raises(PortableNodeError, match="row"):
        decode_portable(two_arrays, limits=PortableLimits(max_array_rows=1))


def test_schema_version_and_arrow_container_are_present() -> None:
    payload = encode_portable({"answer": 42})

    assert payload.startswith(b"ARROW1")
    assert decode_portable(payload) == {"answer": 42}
