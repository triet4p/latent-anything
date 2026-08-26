"""Private torch fitting and EMA helpers for the compact JEPA adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class JEPATrainingResult:
    final_loss: float
    final_prediction: np.ndarray
    final_target: np.ndarray
    training_steps: int


def fit_jepa_parameters(
    context_encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    observations: np.ndarray,
    actions: np.ndarray,
    mask: np.ndarray,
    *,
    observation_dim: int,
    action_dim: int,
    latent_dim: int,
    epochs: int,
    learning_rate: float,
    ema_momentum: float,
    variance_loss_weight: float,
    minimum_latent_std: float,
    variance_floor: float,
    device: str,
    seed: int,
    initial_training_steps: int,
) -> JEPATrainingResult:
    """Fit context/predictor modules and update the target encoder by EMA."""

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch_device = torch.device(device)
    optimizer = torch.optim.Adam(context_encoder.parameters(), lr=learning_rate)
    optimizer.add_param_group({"params": predictor.parameters()})
    observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=torch_device)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=torch_device)
    mask_tensor = torch.as_tensor(mask.reshape(-1), dtype=torch.bool, device=torch_device)
    final_loss = float("nan")
    final_prediction = torch.empty((0, latent_dim), device=torch_device)
    final_target = torch.empty((0, latent_dim), device=torch_device)
    training_steps = initial_training_steps
    for _ in range(epochs):
        optimizer.zero_grad()
        context = context_encoder(observation_tensor[:, :-1].reshape(-1, observation_dim))
        with torch.no_grad():
            target = target_encoder(observation_tensor[:, 1:].reshape(-1, observation_dim))
        predicted = predictor(context, action_tensor.reshape(-1, action_dim))
        valid_predicted = predicted[mask_tensor]
        valid_target = target[mask_tensor]
        prediction_loss = torch.mean(torch.square(valid_predicted - valid_target))
        context_std = torch.sqrt(torch.var(context[mask_tensor], dim=0, unbiased=False) + variance_floor)
        variance_penalty = torch.mean(torch.relu(minimum_latent_std - context_std))
        loss = prediction_loss + variance_loss_weight * variance_penalty
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for target_parameter, context_parameter in zip(
                target_encoder.parameters(), context_encoder.parameters(), strict=True
            ):
                target_parameter.mul_(ema_momentum).add_(context_parameter, alpha=1.0 - ema_momentum)
        training_steps += 1
        final_loss = float(loss.detach().cpu().item())
        final_prediction = valid_predicted.detach()
        final_target = valid_target.detach()
    return JEPATrainingResult(
        final_loss=final_loss,
        final_prediction=final_prediction.cpu().numpy(),
        final_target=final_target.cpu().numpy(),
        training_steps=training_steps,
    )
