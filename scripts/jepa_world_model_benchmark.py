"""Generate reproducible CPU evidence for the compact JEPA world model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from latent_anything.adapters.jepa import JEPAEvaluationReport, JEPAWorldModelAdapter


def make_sequences(seed: int, *, episodes: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic action-conditioned latent-dynamics fixture."""

    rng = np.random.default_rng(seed)
    observations = np.zeros((episodes, horizon + 1, 3), dtype=np.float64)
    observations[:, 0] = rng.normal(scale=0.5, size=(episodes, 3))
    actions = rng.normal(scale=0.4, size=(episodes, horizon, 1))
    for index in range(horizon):
        observations[:, index + 1] = 0.7 * observations[:, index] + 0.2 * np.repeat(actions[:, index], 3, axis=1)
    return observations, actions


def main() -> None:
    train_observations, train_actions = make_sequences(71, episodes=24, horizon=6)
    heldout_observations, heldout_actions = make_sequences(1701, episodes=12, horizon=6)
    adapter = JEPAWorldModelAdapter(
        observation_dim=3,
        latent_dim=2,
        action_dim=1,
        hidden_dim=12,
        epochs=80,
        learning_rate=0.03,
        seed=71,
        source_space_identity="compact-jepa-lewm-v1:synthetic-controlled-latent-dynamics-v1",
    ).fit(train_observations, train_actions)

    heldout_current = heldout_observations[:, :-1].reshape(-1, 3)
    heldout_next = heldout_observations[:, 1:].reshape(-1, 3)
    heldout_actions_flat = heldout_actions.reshape(-1, 1)
    prediction = adapter.evaluate_one_step(heldout_current, heldout_actions_flat, heldout_next)
    initial = adapter.encode(heldout_observations[:, 0])
    target = adapter.encode_target(heldout_observations.reshape(-1, 3)).reshape(heldout_observations.shape[0], -1, 2)
    target_states = np.concatenate((initial[:, None, :], target[:, 1:, :]), axis=1)
    rollout = adapter.evaluate_rollout(initial, heldout_actions, target_states)
    report = JEPAEvaluationReport(
        prediction=prediction,
        rollout=rollout,
        provenance={
            "model_revision": adapter.model_revision,
            "dataset_revision": adapter.dataset_revision,
            "implementation": "latent_anything.adapters.jepa.JEPAWorldModelAdapter",
            "train_seed": 71,
            "heldout_seed": 1701,
            "decoder": "absent",
            "evidence_tier": "D2_synthetic_cpu",
        },
    )
    payload = report.to_dict()
    payload["fit_metadata"] = dict(adapter.fit_metadata)
    payload["train_target_health"] = adapter.evaluate_latent_health(
        train_observations[:, 1:].reshape(-1, 3), target=True
    ).to_dict()
    payload["heldout_target_health"] = adapter.evaluate_latent_health(
        heldout_observations[:, 1:].reshape(-1, 3), target=True
    ).to_dict()
    output = Path("artifacts/jepa_world_model_evidence.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = {
        "model_revision": adapter.model_revision,
        "dataset_revision": adapter.dataset_revision,
        "observation_dim": adapter.observation_dim,
        "latent_dim": adapter.latent_dim,
        "action_dim": adapter.action_dim,
        "fit_config": adapter.to_config().model_dump(mode="json"),
        "heldout_episodes": int(heldout_observations.shape[0]),
        "heldout_horizon": int(heldout_actions.shape[1]),
    }
    Path("artifacts/jepa_world_model_evidence_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "metrics": report.to_metrics()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
