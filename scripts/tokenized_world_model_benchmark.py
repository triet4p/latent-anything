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


def make_observation_sequences(
    *,
    frames: np.ndarray,
    seed: int,
    episodes: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build image trajectories whose frames are later encoded by the VQ-VAE."""

    rng = np.random.default_rng(seed)
    if frames.ndim != 4 or tuple(frames.shape[1:]) != (1, 8, 8):
        raise ValueError("frames must have shape (n, 1, 8, 8)")
    observations = np.empty((episodes, horizon + 1, 1, 8, 8), dtype=np.float64)
    actions = rng.choice(np.asarray([-1.0, 1.0]), size=(episodes, horizon, 1))
    observations[:, 0] = frames[rng.integers(0, len(frames), size=episodes)]
    for step in range(horizon):
        for episode in range(episodes):
            shift = 1 if actions[episode, step, 0] > 0.0 else -1
            observations[episode, step + 1] = np.roll(observations[episode, step], shift=shift, axis=-1)
    return observations, actions.astype(np.float64)


def brightness_proxy(values: np.ndarray) -> np.ndarray:
    """Return a decoder-backed binary task proxy for each image."""

    return np.mean(values, axis=(-3, -2, -1)) > 0.25


def main() -> None:
    """Train on one pinned digits slice and evaluate a held-out trajectory slice."""

    digits = cast(_DigitsDataset, load_digits())
    images = (digits.images / 16.0).astype(np.float64)[:, None, :, :]
    train_images = images[:256]
    heldout_images = images[256:320]
    tokenizer = VQVAE(codebook_size=8, embedding_dim=6, random_state=72, n_epochs=4)
    tokenizer.fit(train_images)
    train_observations, train_actions = make_observation_sequences(
        frames=train_images,
        seed=72,
        episodes=128,
        horizon=6,
    )
    heldout_observations, heldout_actions = make_observation_sequences(
        frames=heldout_images,
        seed=1702,
        episodes=32,
        horizon=8,
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
    model.fit(train_observations, train_actions)
    report = model.evaluate(heldout_observations, heldout_actions, task_proxy=brightness_proxy)
    train_tokens = model.encode(train_observations.reshape(-1, 1, 8, 8)).reshape(
        train_observations.shape[0], train_observations.shape[1], model.tokens_per_frame
    )
    heldout_tokens = model.encode(heldout_observations.reshape(-1, 1, 8, 8)).reshape(
        heldout_observations.shape[0], heldout_observations.shape[1], model.tokens_per_frame
    )
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
    target_decoded = heldout_observations
    predicted_decoded = model.decode(predicted_rollouts[:, 1:].reshape(-1, model.tokens_per_frame)).reshape(
        heldout_tokens.shape[0], heldout_tokens.shape[1] - 1, 1, 8, 8
    )
    decoded_mse = [
        float(np.mean(np.square(predicted_decoded[:, step] - target_decoded[:, step + 1])))
        for step in range(heldout_actions.shape[1])
    ]
    task_accuracy = [
        float(np.mean(brightness_proxy(predicted_decoded[:, step]) == brightness_proxy(target_decoded[:, step + 1])))
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
    tokenizer_code_usage = tokenizer.codebook_diagnostics(tokenizer_train_codes)
    train_code_usage = model.code_usage(train_tokens)
    heldout_code_usage = model.code_usage(heldout_tokens)
    train_counts = np.asarray(cast(list[int], train_code_usage["counts"]), dtype=np.int64)
    heldout_counts = np.asarray(cast(list[int], heldout_code_usage["counts"]), dtype=np.int64)
    active_train_codes = int(np.count_nonzero(train_counts))
    active_heldout_codes = int(np.count_nonzero(heldout_counts))
    nontrivial_token_usage = bool(
        float(tokenizer_code_usage["codebook_perplexity"]) > 1.0
        and float(tokenizer_code_usage["dead_code_rate"]) < 1.0
        and active_train_codes >= 2
        and active_heldout_codes >= 2
    )
    tokenizer_used_for_fit = bool(
        model.fit_metadata.get("input_representation") == "raw_observations"
        and model.fit_metadata.get("codebook_version") == tokenizer.codebook_version
    )
    heldout_observations_encoded_before_evaluation = bool(
        report.provenance.get("input_representation") == "raw_observations"
        and report.provenance.get("codebook_version") == tokenizer.codebook_version
    )
    acceptance = {
        "teacher_forced_perplexity_finite": bool(np.isfinite(report.teacher_forced.perplexity)),
        "free_running_horizon_complete": report.free_running.horizon == heldout_actions.shape[1],
        "decoded_consistency_available": report.free_running.decoded_mse_by_horizon is not None,
        "task_proxy_available": report.free_running.task_proxy_accuracy_by_horizon is not None,
        "seeded_rollout_reproducible": bool(np.array_equal(seed_a.to_numpy(), seed_b.to_numpy())),
        "tokenizer_used_for_fit": tokenizer_used_for_fit,
        "heldout_observations_encoded_before_evaluation": heldout_observations_encoded_before_evaluation,
        "nontrivial_token_usage": nontrivial_token_usage,
    }
    evidence_status = "D2" if all(acceptance.values()) else "D1"
    dominant_train_code = int(np.argmax(train_counts))
    payload = report.to_dict()
    payload.update(
        {
            "dataset": "sklearn.datasets.load_digits",
            "dataset_revision": VQVAE.dataset_revision,
            "evidence_status": evidence_status,
            "evidence_tier": f"{evidence_status}_end_to_end_encoded_observation_cpu",
            "benchmark_input": "raw_digits_trajectories_encoded_by_fitted_vq_vae",
            "fit_metadata": dict(model.fit_metadata),
            "tokenizer_code_usage": tokenizer_code_usage,
            "train_code_usage": train_code_usage,
            "heldout_code_usage": heldout_code_usage,
            "decoded_consistency_mse_by_horizon": list(report.free_running.decoded_mse_by_horizon or decoded_mse),
            "task_proxy_accuracy_by_horizon": list(report.free_running.task_proxy_accuracy_by_horizon or task_accuracy),
            "sampled_rollout": {
                "temperature": 1.8,
                "token_error_by_horizon": sampled_errors,
                "failure_horizon": sampled_failure_horizon,
            },
            "seeded_rollout_bit_exact": bool(np.array_equal(seed_a.to_numpy(), seed_b.to_numpy())),
            "rollout_example": seed_a.to_numpy().tolist(),
            "acceptance": acceptance,
            "failure_analysis": (
                "The benchmark fits from raw image trajectories after encoding every frame with the fitted VQ-VAE "
                "and measures that provenance from fit/evaluation metadata. However, the fitted tokenizer is "
                f"collapsed: tokenizer perplexity={float(tokenizer_code_usage['codebook_perplexity']):.6g}, "
                f"dead-code rate={float(tokenizer_code_usage['dead_code_rate']):.6g}, "
                f"active train codes={active_train_codes}, active held-out codes={active_heldout_codes}, "
                f"with every training token mapped to code {dominant_train_code}. Perfect token accuracy and "
                "exact-frame metrics therefore measure constant-token prediction, not meaningful dynamics. The "
                f"nontrivial_token_usage gate is {nontrivial_token_usage}, so this reproduction remains "
                f"{evidence_status} "
                "evidence despite the end-to-end wiring; it makes no real-model or CUDA claim."
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
        "sequence_generation": "horizontal_roll_actions_on_digits_images",
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
