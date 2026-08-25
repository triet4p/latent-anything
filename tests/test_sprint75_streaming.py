"""Sprint 75 bounded rollout streaming and eager-equivalence tests."""

from __future__ import annotations

import numpy as np

from latent_anything import DeterministicLatentTransition, LatentSpace, RolloutPipeline


def _pipeline() -> RolloutPipeline:
    states = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    actions = np.ones((4, 1), dtype=np.float64)
    transition = DeterministicLatentTransition(LatentSpace(2, source_model="sprint75-test"), 1).fit(
        states, actions, states + np.array([1.0, 0.0])
    )
    return RolloutPipeline(transition)


def test_long_chunked_rollout_matches_eager_without_accumulating_stream_outputs() -> None:
    horizon = 257
    chunk_rows = 17
    actions = np.ones((horizon, 1), dtype=np.float64)
    eager = _pipeline().run(np.zeros(2), actions).to_numpy()[1:]
    yielded_rows = 0
    max_chunk_bytes = 0
    streamed_chunks = _pipeline().stream(
        np.zeros(2),
        (np.ones((min(chunk_rows, horizon - start), 1)) for start in range(0, horizon, chunk_rows)),
        max_chunk_rows=chunk_rows,
    )
    for chunk in streamed_chunks:
        values = chunk.to_numpy()
        yielded_rows += values.shape[0]
        max_chunk_bytes = max(max_chunk_bytes, values.nbytes)
        np.testing.assert_array_equal(values, eager[yielded_rows - values.shape[0] : yielded_rows])
        del values

    assert yielded_rows == horizon
    assert max_chunk_bytes <= chunk_rows * 2 * np.dtype(np.float64).itemsize
