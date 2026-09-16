"""Focused tests for the bounded dictionary-learning evidence lane."""

from __future__ import annotations

import numpy as np
import pytest

from latent_anything.dictionary_learning import DictionaryLearning, DictionaryLearningConfig


def _fixture(seed: int = 79) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dictionary = rng.normal(size=(8, 12))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    codes = np.zeros((600, 8), dtype=np.float64)
    for row in codes:
        active = rng.choice(8, size=2, replace=False)
        row[active] = rng.uniform(0.4, 1.0, size=2)
    return codes @ dictionary + rng.normal(scale=0.01, size=(600, 12))


def test_heldout_fit_is_sparse_and_matches_public_reconstruction() -> None:
    data = _fixture()
    learner = DictionaryLearning(DictionaryLearningConfig(alpha=0.05, transform_n_nonzero_coefs=2))
    evaluation = learner.fit(data)

    assert learner.train_indices_ is not None
    assert learner.validation_indices_ is not None
    train_reconstruction = learner.reconstruct(data[learner.train_indices_])
    val_reconstruction = learner.reconstruct(data[learner.validation_indices_])

    assert evaluation.n_train == 480
    assert evaluation.n_val == 120
    assert evaluation.val_reconstruction_mse < evaluation.val_baseline_mse * 0.5
    assert evaluation.val_mean_l0 <= 2.0
    assert np.isfinite(evaluation.val_reconstruction_mse)
    np.testing.assert_allclose(
        evaluation.train_reconstruction_mse,
        np.mean((data[learner.train_indices_] - train_reconstruction) ** 2),
    )
    np.testing.assert_allclose(
        evaluation.val_reconstruction_mse,
        np.mean((data[learner.validation_indices_] - val_reconstruction) ** 2),
    )


def test_fit_is_deterministic_and_does_not_mutate_input() -> None:
    data = _fixture()
    before = data.copy()
    config = DictionaryLearningConfig(alpha=0.05, transform_n_nonzero_coefs=2)
    first = DictionaryLearning(config)
    second = DictionaryLearning(config)
    first_eval = first.fit(data)
    second_eval = second.fit(data)

    np.testing.assert_array_equal(data, before)
    np.testing.assert_array_equal(first.train_indices_, second.train_indices_)
    np.testing.assert_allclose(first.components_, second.components_, atol=1e-12)
    np.testing.assert_allclose(first_eval.val_codes, second_eval.val_codes, atol=1e-12)


def test_invalid_data_and_unfitted_use_fail_closed() -> None:
    learner = DictionaryLearning()
    with pytest.raises(RuntimeError, match="must be fitted"):
        learner.transform(np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        learner.fit(np.array([[0.0, np.nan], [1.0, 2.0]]))
