"""Private token-dynamics training, sampling, and result helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim

from latent_anything._tokenized_dynamics import AutoregressiveDynamics


@dataclass(frozen=True, slots=True)
class TokenTrainingResult:
    loss: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class TokenSamplingResult:
    tokens: np.ndarray
    token_log_likelihood: np.ndarray


def fit_token_dynamics(
    dynamics: AutoregressiveDynamics,
    current: np.ndarray,
    target: np.ndarray,
    actions: np.ndarray,
    *,
    vocab_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> TokenTrainingResult:
    """Fit the concrete GRU on already validated integer transitions."""

    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    torch.set_num_threads(1)
    source_tensor = torch.from_numpy(current.astype(np.int64))  # pyright: ignore[reportUnknownMemberType]
    target_tensor = torch.from_numpy(target.astype(np.int64))  # pyright: ignore[reportUnknownMemberType]
    action_tensor = torch.from_numpy(actions.astype(np.float32))  # pyright: ignore[reportUnknownMemberType]
    optimizer = optim.Adam(dynamics.parameters(), lr=learning_rate)
    last_loss = 0.0
    last_accuracy = 0.0
    for _ in range(epochs):
        hidden = dynamics.encode_context(source_tensor, action_tensor)
        logits = dynamics.decode_teacher_forced(hidden, action_tensor, target_tensor)
        loss = nn.functional.cross_entropy(logits.reshape(-1, vocab_size), target_tensor.reshape(-1))
        optimizer.zero_grad()
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        with torch.no_grad():
            last_loss = float(loss.detach())  # pyright: ignore[reportUnknownMemberType]
            last_accuracy = float((logits.argmax(dim=-1) == target_tensor).float().mean())  # pyright: ignore[reportUnknownMemberType]
    return TokenTrainingResult(last_loss, last_accuracy)


def sample_next_tokens(
    dynamics: AutoregressiveDynamics,
    current: np.ndarray,
    actions: np.ndarray,
    *,
    vocab_size: int,
    tokens_per_frame: int,
    sampling: str,
    temperature: float,
    top_k: int | None,
    seed: int | None,
) -> TokenSamplingResult:
    """Generate one next frame using greedy or seeded categorical sampling."""

    generator = np.random.default_rng(seed)
    current_tensor = torch.from_numpy(current.astype(np.int64))  # pyright: ignore[reportUnknownMemberType]
    action_tensor = torch.from_numpy(actions.astype(np.float32))  # pyright: ignore[reportUnknownMemberType]
    with torch.no_grad():
        hidden = dynamics.encode_context(current_tensor, action_tensor)
        prefix = torch.empty((current.shape[0], 0), dtype=torch.long)
        output = np.empty((current.shape[0], tokens_per_frame), dtype=np.int64)
        log_likelihood = np.empty_like(output, dtype=np.float64)
        for position in range(tokens_per_frame):
            logits = dynamics.decode_one(hidden, action_tensor, prefix)
            scaled = logits / temperature
            if top_k is not None and top_k < vocab_size:
                values, indices = torch.topk(scaled, top_k, dim=-1)
                filtered = torch.full_like(scaled, -torch.inf)
                filtered.scatter_(1, indices, values)
                scaled = filtered
            log_probs = torch.log_softmax(scaled, dim=-1)
            if sampling == "greedy":
                next_token = torch.argmax(log_probs, dim=-1)
            else:
                probabilities = torch.softmax(scaled, dim=-1).cpu().numpy()
                sampled = [generator.choice(vocab_size, p=row) for row in probabilities]
                next_token = torch.from_numpy(np.asarray(sampled, dtype=np.int64))
            output[:, position] = next_token.cpu().numpy()
            log_likelihood[:, position] = log_probs.gather(1, next_token[:, None]).squeeze(1).cpu().numpy()
            prefix = torch.cat((prefix, next_token[:, None]), dim=1)
    return TokenSamplingResult(output, log_likelihood)
