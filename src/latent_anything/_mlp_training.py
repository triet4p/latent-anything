"""Torch-only training mechanics for the public MLP probe facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch has incomplete type stubs — these warnings are noise)


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs deterministically."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[reportAttributeAccessIssue]
    torch.backends.cudnn.benchmark = False  # type: ignore[reportAttributeAccessIssue]


class _MLP(torch.nn.Module):  # type: ignore[reportMissingTypeStubs]
    """Small bounded MLP used only inside nonlinear probing."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_sizes: list[int] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        hidden = hidden_sizes or [64]
        layers: list[torch.nn.Module] = []
        prev = n_features
        act_fn = torch.nn.ReLU() if activation == "relu" else torch.nn.Tanh()

        for h in hidden:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(act_fn)
            prev = h

        layers.append(torch.nn.Linear(prev, n_classes))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass returning logits."""
        return self.net(x)

    def count_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture(self) -> dict[str, Any]:
        """Return the stable architecture summary used by result schemas."""
        return {
            "type": "MLP",
            "hidden_sizes": [int(layer.out_features) for layer in self.net if isinstance(layer, torch.nn.Linear)][:-1],
            "n_hidden_layers": sum(1 for _ in [layer for layer in self.net if isinstance(layer, torch.nn.Linear)]) - 1,
            "n_params": self.count_params(),
        }


@dataclass(frozen=True)
class MLPTrainingResult:
    """Torch-free result boundary returned to the facade."""

    accuracy: float
    val_accuracy: float
    predictions: np.ndarray
    probabilities: np.ndarray
    n_epochs: int
    stopped_early: bool
    architecture: dict[str, Any]
    n_params: int
    optimizer: str


def train_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray | None,
    val_y: np.ndarray | None,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    n_features: int,
    n_classes: int,
    hidden_sizes: list[int],
    activation: str,
    max_epochs: int,
    early_stopping_patience: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    random_state: int,
) -> MLPTrainingResult:
    """Train and evaluate one deterministic MLP on preprocessed NumPy data."""
    _seed_everything(random_state)
    has_val = val_x is not None and val_y is not None and val_x.shape[0] > 0

    model = _MLP(
        n_features=n_features,
        n_classes=n_classes,
        hidden_sizes=hidden_sizes,
        activation=activation,
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    x_train_t = torch.as_tensor(train_x, dtype=torch.float32)
    y_train_t = torch.as_tensor(train_y, dtype=torch.long)
    x_test_t = torch.as_tensor(test_x, dtype=torch.float32)
    x_val_t = torch.as_tensor(val_x, dtype=torch.float32) if has_val and val_x is not None else None
    y_val_t = torch.as_tensor(val_y, dtype=torch.long) if has_val and val_y is not None else None

    n = len(train_x)
    best_val_acc = -1.0
    patience_counter = 0
    n_epochs_run = 0
    stopped_early = False

    for epoch in range(max_epochs):
        perm = torch.randperm(n)
        x_shuffled = x_train_t[perm]
        y_shuffled = y_train_t[perm]

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_x = x_shuffled[start:end]
            batch_y = y_shuffled[start:end]

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

        if has_val and x_val_t is not None and y_val_t is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(x_val_t)
                val_preds_t = val_logits.argmax(dim=1)
                val_acc_epoch = float((val_preds_t == y_val_t).float().mean())
            model.train()

            if val_acc_epoch > best_val_acc:
                best_val_acc = val_acc_epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                n_epochs_run = epoch + 1
                stopped_early = True
                break

        n_epochs_run = epoch + 1

    model.eval()
    with torch.no_grad():
        test_logits = model(x_test_t)
        test_probs_t = torch.softmax(test_logits, dim=1)
        test_preds_t = test_logits.argmax(dim=1)

        test_preds: np.ndarray = np.asarray(test_preds_t.numpy())
        test_probs: np.ndarray = np.asarray(test_probs_t.numpy())
        accuracy = float(np.mean(test_preds == test_y))

        val_accuracy = 0.0
        if has_val and x_val_t is not None and y_val_t is not None:
            val_logits = model(x_val_t)
            val_preds_t = val_logits.argmax(dim=1)
            val_preds_np: np.ndarray = np.asarray(val_preds_t.numpy())
            val_accuracy = float(np.mean(val_preds_np == val_y))

    return MLPTrainingResult(
        accuracy=accuracy,
        val_accuracy=val_accuracy,
        predictions=test_preds,
        probabilities=test_probs,
        n_epochs=n_epochs_run,
        stopped_early=stopped_early,
        architecture=model.architecture(),
        n_params=model.count_params(),
        optimizer="AdamW",
    )
