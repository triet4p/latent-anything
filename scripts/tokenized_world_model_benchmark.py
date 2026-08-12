"""Generate reproducible CPU evidence for tokenized world-model dynamics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from sklearn.datasets import load_digits  # pyright: ignore[reportMissingTypeStubs]

from latent_anything.adapters import VQVAE
from latent_anything.tokenized_world_model import TokenizedWorldModel


class _DigitsDataset(Protocol):
    images: np.ndarray
    target: np.ndarray


def make_token_sequences(
    *,
    seed: int,
    episodes: int,
    horizon: int,
    vocab_size: int,
    tokens_per_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic action-controlled codebook task."""

    rng = np.random.default_rng(seed)
    observations = np.empty((episodes, horizon + 1, tokens_per_frame), dtype=np.int64)
    actions = rng.choice(np.asarray([-1.0, 1.0]), size=(episodes, horizon, 1))
    initial = rng.integers(0, vocab_size, size=(episodes, 1))
    observations[:, 0] = np.repeat(initial, tokens_per_frame, axis=1)
    for step in range(horizon):
        drift = np.where(actions[:, step, 0] > 0.0, 1, 2)[:, None]
        observations[:, step + 1] = np.repeat(
            (observations[:, step, :1] + drift) % vocab_size, tokens_per_frame, axis=1
        )
    return observations, actions.astype(np.float64)


def brightness_proxy(values: np.ndarray) -> np.ndarray:
    """Return a decoder-backed binary task proxy for each image."""

    return np.mean(values, axis=(-3, -2, -1)) > 0.25


def main() -> None:
    """Train on one pinned digits slice and evaluate a held-out trajectory slice."""

    digits = cast(_DigitsDataset, load_digits())
    images = (digits.images / 16.0).astype(np.float64)[:, None, :, :]
    train_images = images[:256]
    tokenizer = VQVAE(codebook_size=8, embedding_dim=6, random_state=72, n_epochs=4)
    tokenizer.fit(train_images)
    train_tokens, train_actions = make_token_sequences(
        seed=72,
        episodes=128,
        horizon=6,
        vocab_size=tokenizer.codebook_size,
        tokens_per_frame=tokenizer.sequence_length,
    )
    heldout_tokens, heldout_actions = make_token_sequences(
        seed=1702,
        episodes=32,
        horizon=8,
        vocab_size=tokenizer.codebook_size,
        tokens_per_frame=tokenizer.sequence_length,
    )
    model = TokenizedWorldModel(
        tokenizer,
        action_dim=1,
        hidden_dim=32,
        epochs=180,
        learning_rate=0.01,
        seed=72,
        model_revision="compact-tokenized-world-model-v1",
    )
    model.fit_tokens(train_tokens, train_actions)
    report = model.evaluate(heldout_tokens, heldout_actions, task_proxy=brightness_proxy)
    predicted_rollouts = np.stack(
        [model.rollout(tokens[0], action).to_numpy() for tokens, action in zip(heldout_tokens, heldout_actions)], axis=0
    )
    sampled_rollouts = np.stack(
        [
            model.rollout(tokens[0], action, sampling="sample", temperature=1.8, seed=1702 + index).to_numpy()
            for index, (tokens, action) in enumerate(zip(heldout_tokens, heldout_actions))
        ],
        axis=0,
    )
    target_decoded = model.decode(heldout_tokens.reshape(-1, model.tokens_per_frame)).reshape(
        heldout_tokens.shape[0], heldout_tokens.shape[1], 1, 8, 8
    )
    predicted_decoded = model.decode(predicted_rollouts[:, 1:].reshape(-1, model.tokens_per_frame)).reshape(
        heldout_tokens.shape[0], heldout_tokens.shape[1] - 1, 1, 8, 8
    )
    decoded_mse = [
        float(np.mean(np.square(predicted_decoded[:, step] - target_decoded[:, step + 1])))
        for step in range(heldout_actions.shape[1])
    ]
    task_accuracy = [
        float(np.mean((predicted_rollouts[:, step + 1, 0] % 2) == (heldout_tokens[:, step + 1, 0] % 2)))
        for step in range(heldout_actions.shape[1])
    ]
    sampled_errors = [
        float(np.mean(sampled_rollouts[:, step + 1] != heldout_tokens[:, step + 1]))
        for step in range(heldout_actions.shape[1])
    ]
    sampled_failure_horizon = next(
        (index + 1 for index, value in enumerate(sampled_errors) if value > 0.5),
        None,
    )
    tokenizer_train_codes = tokenizer.encode(train_images)
    seed_a = model.rollout(heldout_tokens[0, 0], heldout_actions[0], sampling="sample", seed=72)
    seed_b = model.rollout(heldout_tokens[0, 0], heldout_actions[0], sampling="sample", seed=72)
    payload = report.to_dict()
    payload.update(
        {
            "dataset": "sklearn.datasets.load_digits",
            "dataset_revision": VQVAE.dataset_revision,
            "evidence_tier": "D2_synthetic_cpu",
            "fit_metadata": dict(model.fit_metadata),
            "tokenizer_code_usage": tokenizer.codebook_diagnostics(tokenizer_train_codes),
            "train_code_usage": model.code_usage(train_tokens),
            "heldout_code_usage": model.code_usage(heldout_tokens),
            "decoded_consistency_mse_by_horizon": list(report.free_running.decoded_mse_by_horizon or decoded_mse),
            "task_proxy_accuracy_by_horizon": list(report.free_running.task_proxy_accuracy_by_horizon or task_accuracy),
            "sampled_rollout": {
                "temperature": 1.8,
                "token_error_by_horizon": sampled_errors,
                "failure_horizon": sampled_failure_horizon,
            },
            "seeded_rollout_bit_exact": bool(np.array_equal(seed_a.to_numpy(), seed_b.to_numpy())),
            "rollout_example": seed_a.to_numpy().tolist(),
            "acceptance": {
                "teacher_forced_perplexity_finite": bool(np.isfinite(report.teacher_forced.perplexity)),
                "free_running_horizon_complete": report.free_running.horizon == heldout_actions.shape[1],
                "decoded_consistency_available": report.free_running.decoded_mse_by_horizon is not None,
                "task_proxy_available": report.free_running.task_proxy_accuracy_by_horizon is not None,
                "seeded_rollout_reproducible": bool(np.array_equal(seed_a.to_numpy(), seed_b.to_numpy())),
            },
            "failure_analysis": (
                "Teacher-forced likelihood is local to ground-truth contexts, while free-running drift feeds each "
                "predicted frame back into the next query. The recorded failure_horizon and per-horizon decoded/task "
                "metrics are the acceptance evidence; this compact CPU run does not claim real-model or CUDA scale."
            ),
        }
    )
    output = Path("artifacts/tokenized_world_model_evidence.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = {
        "dataset": payload["dataset"],
        "dataset_revision": payload["dataset_revision"],
        "model_revision": model.model_revision,
        "codebook_version": model.codebook_version,
        "seed": 72,
        "heldout_seed": 1702,
        "train_episodes": int(train_tokens.shape[0]),
        "train_horizon": int(train_actions.shape[1]),
        "heldout_episodes": int(heldout_tokens.shape[0]),
        "heldout_horizon": int(heldout_actions.shape[1]),
        "tokenizer": {
            "codebook_size": tokenizer.codebook_size,
            "embedding_dim": tokenizer.embedding_dim,
            "n_epochs": tokenizer.n_epochs,
            "learning_rate": tokenizer.learning_rate,
        },
        "dynamics": model.to_config().model_dump(mode="json"),
        "offline": True,
    }
    output.with_name("tokenized_world_model_evidence_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "teacher_forced": report.teacher_forced.to_dict(),
                "free_running": report.free_running.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
