"""Private autoregressive GRU dynamics used by the tokenized world model."""

from __future__ import annotations

import torch
from torch import nn


class AutoregressiveDynamics(nn.Module):
    """Concrete GRU encoder/decoder; token IDs remain categorical at the facade."""

    def __init__(self, vocab_size: int, action_dim: int, hidden_dim: int, pad_token_id: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size + 1, hidden_dim, padding_idx=pad_token_id)
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.bos = nn.Parameter(torch.zeros(hidden_dim))

    def encode_context(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        action_embedding = self.action_projection(actions).unsqueeze(1)
        values = self.token_embedding(tokens) + action_embedding
        _, hidden = self.encoder(values)
        return hidden

    def decode_teacher_forced(
        self, hidden: torch.Tensor, actions: torch.Tensor, target_tokens: torch.Tensor
    ) -> torch.Tensor:
        action_embedding = self.action_projection(actions).unsqueeze(1)
        prefix = torch.cat(
            (self.bos.expand(target_tokens.shape[0], 1, -1), self.token_embedding(target_tokens[:, :-1])), dim=1
        )
        values = prefix + action_embedding
        decoded, _ = self.decoder(values, hidden)
        return self.output(decoded)

    def decode_one(self, hidden: torch.Tensor, actions: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        action_embedding = self.action_projection(actions)
        value = self.bos.expand(prefix.shape[0], -1) if prefix.shape[1] == 0 else self.token_embedding(prefix[:, -1])
        decoded, _ = self.decoder((value + action_embedding).unsqueeze(1), hidden)
        return self.output(decoded[:, 0])
