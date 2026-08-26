"""Private torch-backed fitting helpers for the compact RSSM implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class RSSMFittedParameters:
    """NumPy parameters and diagnostics produced by one bounded fit."""

    recurrent_weights: np.ndarray
    recurrent_bias: np.ndarray
    emission_weights: np.ndarray
    emission_bias: np.ndarray
    scale: np.ndarray
    predictions: np.ndarray
    final_loss: float


def fit_rssm_parameters(
    states: np.ndarray,
    actions: np.ndarray,
    mask: np.ndarray,
    *,
    hidden_dim: int,
    state_dim: int,
    action_dim: int,
    epochs: int,
    learning_rate: float,
    variance_floor: float,
    device: str,
    seed: int,
) -> RSSMFittedParameters:
    """Fit the recurrent/emission torch layers and return NumPy parameters."""

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch_device = torch.device(device)
    hidden_dim_input = hidden_dim + state_dim + action_dim
    recurrent = torch.nn.Linear(hidden_dim_input, hidden_dim, device=torch_device)
    emission = torch.nn.Linear(hidden_dim + state_dim + action_dim + 1, state_dim, device=torch_device)
    optimizer = torch.optim.Adam([*recurrent.parameters(), *emission.parameters()], lr=learning_rate)
    states_tensor = torch.as_tensor(states, dtype=torch.float32, device=torch_device)
    actions_tensor = torch.as_tensor(actions, dtype=torch.float32, device=torch_device)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=torch_device)
    final_loss = float("nan")
    for _ in range(epochs):
        optimizer.zero_grad()
        hidden = torch.zeros((states.shape[0], hidden_dim), device=torch_device)
        total_loss = torch.zeros((), device=torch_device)
        valid_count = torch.zeros((), device=torch_device)
        for index in range(actions.shape[1]):
            valid = mask_tensor[:, index]
            proposed = torch.tanh(
                recurrent(torch.cat((hidden, states_tensor[:, index], actions_tensor[:, index]), dim=1))
            )
            hidden = torch.where(valid[:, None], proposed, hidden)
            prediction = emission(
                torch.cat(
                    (
                        hidden,
                        states_tensor[:, index],
                        actions_tensor[:, index],
                        torch.ones((states.shape[0], 1), device=torch_device),
                    ),
                    dim=1,
                )
            )
            residual = torch.square(prediction - states_tensor[:, index + 1])
            total_loss = total_loss + torch.sum(residual * valid[:, None])
            valid_count = valid_count + torch.sum(valid)
        loss = total_loss / torch.clamp(valid_count * state_dim, min=1.0)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())

    recurrent_weights = recurrent.weight.detach().cpu().numpy().T.astype(np.float64)
    recurrent_bias = recurrent.bias.detach().cpu().numpy().astype(np.float64)
    emission_weights = emission.weight.detach().cpu().numpy().T.astype(np.float64)
    emission_bias = emission.bias.detach().cpu().numpy().astype(np.float64)
    _, predictions = teacher_forced_predictions(
        states,
        actions,
        mask,
        recurrent_weights=recurrent_weights,
        recurrent_bias=recurrent_bias,
        emission_weights=emission_weights,
        emission_bias=emission_bias,
        hidden_dim=hidden_dim,
        state_dim=state_dim,
    )
    residuals = states[:, 1:, :] - predictions
    valid_residuals = residuals[mask]
    variances = np.maximum(np.mean(np.square(valid_residuals), axis=0), variance_floor)
    return RSSMFittedParameters(
        recurrent_weights=recurrent_weights,
        recurrent_bias=recurrent_bias,
        emission_weights=emission_weights,
        emission_bias=emission_bias,
        scale=np.sqrt(variances),
        predictions=predictions,
        final_loss=final_loss,
    )


def teacher_forced_predictions(
    states: np.ndarray,
    actions: np.ndarray,
    mask: np.ndarray,
    *,
    recurrent_weights: np.ndarray,
    recurrent_bias: np.ndarray,
    emission_weights: np.ndarray,
    emission_bias: np.ndarray,
    hidden_dim: int,
    state_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic hidden paths and teacher-forced mean predictions."""

    hidden = np.zeros((states.shape[0], hidden_dim), dtype=np.float64)
    predictions = np.empty((states.shape[0], actions.shape[1], state_dim), dtype=np.float64)
    hidden_paths = np.empty((states.shape[0], actions.shape[1], hidden_dim), dtype=np.float64)
    for index in range(actions.shape[1]):
        proposed = np.tanh(
            np.concatenate((hidden, states[:, index], actions[:, index]), axis=1) @ recurrent_weights + recurrent_bias
        )
        hidden = np.where(mask[:, index, None], proposed, hidden)
        predictions[:, index] = (
            np.concatenate((hidden, states[:, index], actions[:, index], np.ones((states.shape[0], 1))), axis=1)
            @ emission_weights
            + emission_bias
        )
        hidden_paths[:, index] = hidden
    return hidden_paths, predictions
