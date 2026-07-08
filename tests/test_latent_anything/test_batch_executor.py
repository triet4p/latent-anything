"""Tests for BatchExecutor (Sprint 22 — Runtime #1)."""

from __future__ import annotations

import numpy as np
import pytest

from latent_anything.adapters import RandomProjection
from latent_anything.methods import PCA
from latent_anything.runtime import BatchExecutor


@pytest.fixture
def synthetic_data() -> np.ndarray:
    """Return deterministic synthetic flat-batch data."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(10, 6)).astype(np.float64)


def identity_plus_marker(chunk: np.ndarray) -> np.ndarray:
    """Return a deterministic transform that makes order mistakes visible."""
    row_ids = chunk[:, :1]
    return np.concatenate([row_ids, chunk * 2.0 + 1.0], axis=1)


class TestBatchExecutorConstruction:
    """BatchExecutor construction invariants."""

    def test_construct_with_positive_batch_size(self) -> None:
        """Positive batch_size is accepted."""
        executor = BatchExecutor(batch_size=4)
        assert executor.batch_size == 4

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_rejects_non_positive_batch_size(self, batch_size: int) -> None:
        """Zero and negative batch sizes are rejected."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            BatchExecutor(batch_size=batch_size)

    @pytest.mark.parametrize("batch_size", [1.5, "4", True])
    def test_rejects_non_integer_batch_size(self, batch_size: object) -> None:
        """Non-integer and bool batch sizes are rejected."""
        with pytest.raises(TypeError, match="batch_size must be a positive integer"):
            BatchExecutor(batch_size=batch_size)  # pyright: ignore[reportArgumentType]


class TestBatchExecutorChunking:
    """Deterministic first-axis chunking."""

    def test_exact_divisibility_chunks(self) -> None:
        """Exact divisibility yields equal-sized chunks."""
        data = np.arange(24, dtype=np.float64).reshape(6, 4)
        chunks = list(BatchExecutor(batch_size=2).iter_chunks(data))
        assert [chunk.shape for chunk in chunks] == [(2, 4), (2, 4), (2, 4)]
        np.testing.assert_array_equal(np.concatenate(chunks, axis=0), data)

    def test_remainder_batch_chunks(self) -> None:
        """Non-divisible data keeps a final remainder chunk."""
        data = np.arange(28, dtype=np.float64).reshape(7, 4)
        chunks = list(BatchExecutor(batch_size=3).iter_chunks(data))
        assert [chunk.shape for chunk in chunks] == [(3, 4), (3, 4), (1, 4)]
        np.testing.assert_array_equal(np.concatenate(chunks, axis=0), data)

    def test_batch_size_one_chunks(self) -> None:
        """batch_size=1 yields one sample per chunk in original order."""
        data = np.arange(15, dtype=np.float64).reshape(5, 3)
        chunks = list(BatchExecutor(batch_size=1).iter_chunks(data))
        assert [chunk.shape for chunk in chunks] == [(1, 3)] * 5
        np.testing.assert_array_equal(np.concatenate(chunks, axis=0), data)

    def test_batch_size_larger_than_data_chunks_once(self) -> None:
        """batch_size larger than the data yields one chunk."""
        data = np.arange(12, dtype=np.float64).reshape(3, 4)
        chunks = list(BatchExecutor(batch_size=99).iter_chunks(data))
        assert len(chunks) == 1
        np.testing.assert_array_equal(chunks[0], data)

    def test_rejects_scalar_data(self) -> None:
        """Scalar arrays have no first-axis batch dimension."""
        with pytest.raises(ValueError, match="at least one dimension"):
            list(BatchExecutor(batch_size=2).iter_chunks(np.array(1.0)))


class TestBatchExecutorMapArray:
    """Generic numpy map behavior."""

    def test_map_array_preserves_order_and_shape_exactly(self) -> None:
        """Batched map matches a direct operation exactly."""
        data = np.arange(40, dtype=np.float64).reshape(10, 4)
        direct = identity_plus_marker(data)
        batched = BatchExecutor(batch_size=3).map_array(identity_plus_marker, data)
        assert batched.shape == direct.shape
        np.testing.assert_array_equal(batched, direct)

    def test_map_array_preserves_dtype(self) -> None:
        """Output dtype comes from the operation output."""
        data = np.arange(12, dtype=np.float32).reshape(4, 3)

        def to_float64(chunk: np.ndarray) -> np.ndarray:
            return chunk.astype(np.float64)

        batched = BatchExecutor(batch_size=2).map_array(to_float64, data)
        assert batched.dtype == np.float64

    def test_map_array_rejects_non_numpy_output(self) -> None:
        """Operations must return numpy arrays."""

        def bad_operation(chunk: np.ndarray) -> list[float]:
            return [float(chunk.shape[0])]

        with pytest.raises(TypeError, match="must return numpy.ndarray"):
            BatchExecutor(batch_size=2).map_array(bad_operation, np.ones((4, 2)))  # pyright: ignore[reportArgumentType]

    def test_map_array_rejects_row_count_changes(self) -> None:
        """Each chunk output must preserve chunk row count."""

        def drops_row(chunk: np.ndarray) -> np.ndarray:
            return chunk[:1]

        with pytest.raises(ValueError, match="preserve the first-axis length"):
            BatchExecutor(batch_size=3).map_array(drops_row, np.ones((6, 2)))


class TestBatchExecutorAdapterBatching:
    """Adapter encode/decode batching."""

    def test_encode_matches_direct_random_projection(self, synthetic_data: np.ndarray) -> None:
        """Batched adapter.encode matches the direct encode path."""
        adapter = RandomProjection(input_dim=6, latent_dim=3, random_state=42)
        direct = adapter.encode(synthetic_data)
        batched = BatchExecutor(batch_size=4).encode(adapter, synthetic_data)
        assert batched.shape == direct.shape
        np.testing.assert_allclose(batched, direct)

    def test_decode_matches_direct_random_projection(self, synthetic_data: np.ndarray) -> None:
        """Batched adapter.decode matches the direct decode path."""
        adapter = RandomProjection(input_dim=6, latent_dim=3, random_state=42)
        latent = adapter.encode(synthetic_data)
        direct = adapter.decode(latent)
        batched = BatchExecutor(batch_size=4).decode(adapter, latent)
        assert batched.shape == direct.shape
        np.testing.assert_allclose(batched, direct)

    @pytest.mark.parametrize("batch_size", [1, 3, 10, 99])
    def test_encode_edge_batch_sizes_match_direct(self, synthetic_data: np.ndarray, batch_size: int) -> None:
        """Batch size 1, remainder, exact data length, and larger-than-data all match direct encode."""
        adapter = RandomProjection(input_dim=6, latent_dim=3, random_state=123)
        direct = adapter.encode(synthetic_data)
        batched = BatchExecutor(batch_size=batch_size).encode(adapter, synthetic_data)
        assert batched.shape == direct.shape
        np.testing.assert_allclose(batched, direct)


class TestBatchExecutorMethodBatching:
    """Layer A method transform batching."""

    def test_transform_matches_direct_pca(self, synthetic_data: np.ndarray) -> None:
        """Batched PCA.transform matches direct PCA.transform."""
        method = PCA(n_components=2)
        method.fit(synthetic_data)
        direct = method.transform(synthetic_data)
        batched = BatchExecutor(batch_size=4).transform(method, synthetic_data)
        assert batched.shape == direct.shape
        np.testing.assert_array_almost_equal(batched, direct)

    def test_transform_preserves_remainder_order(self) -> None:
        """PCA transform with a remainder batch preserves row order."""
        data = np.arange(60, dtype=np.float64).reshape(10, 6)
        method = PCA(n_components=2)
        method.fit(data)
        direct = method.transform(data)
        batched = BatchExecutor(batch_size=6).transform(method, data)
        np.testing.assert_array_almost_equal(batched, direct)
