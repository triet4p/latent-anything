"""Private CPU probe and tokenizer controls for M14 L04.8."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from scripts._m14_l04_contract_common import canonical_json_bytes

L2_C = 1.0
GPT2_VOCAB_SIZE = 50257
LBFGS_MAX_ITER = 100
LBFGS_TOLERANCE_GRAD = 1e-9
LBFGS_TOLERANCE_CHANGE = 1e-12
CONVERGENCE_GRAD_TOL = 1e-6


@dataclass(frozen=True)
class ProbeConfig:
    l2_c: float = L2_C
    max_iter: int = LBFGS_MAX_ITER
    tolerance_grad: float = LBFGS_TOLERANCE_GRAD
    tolerance_change: float = LBFGS_TOLERANCE_CHANGE
    convergence_grad_tol: float = CONVERGENCE_GRAD_TOL


@dataclass(frozen=True)
class LogisticProbe:
    weights: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray
    config: ProbeConfig

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.weights.size:
            raise ValueError("feature dimensions do not match probe")
        standardized = (values - self.mean) / self.scale
        logits = standardized @ self.weights + self.intercept
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))
        if not np.isfinite(probabilities).all():
            raise ValueError("probe produced non-finite probabilities")
        return probabilities


def _standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("features must be a finite non-empty matrix")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return (values - mean) / scale, mean, scale


def fit_logistic_probe(
    features: np.ndarray, labels: np.ndarray, *, torch: Any, config: ProbeConfig | None = None
) -> LogisticProbe:
    """Fit the frozen balanced, intercept-bearing CPU float64 logistic probe."""
    config = ProbeConfig() if config is None else config
    standardized, mean, scale = _standardize(features)
    y = np.asarray(labels, dtype=np.float64)
    if y.ndim != 1 or y.size != standardized.shape[0] or np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("labels must be binary and aligned with features")
    counts = np.bincount(y.astype(np.int64), minlength=2)
    if np.any(counts == 0):
        raise ValueError("balanced logistic probe needs both classes in the training partition")
    sample_weights = y * (y.size / (2.0 * counts[1])) + (1.0 - y) * (y.size / (2.0 * counts[0]))
    x_t = torch.as_tensor(standardized, dtype=torch.float64, device="cpu")
    y_t = torch.as_tensor(y, dtype=torch.float64, device="cpu")
    weight_t = torch.zeros(standardized.shape[1], dtype=torch.float64, device="cpu", requires_grad=True)
    intercept_t = torch.zeros((), dtype=torch.float64, device="cpu", requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight_t, intercept_t],
        max_iter=config.max_iter,
        tolerance_grad=config.tolerance_grad,
        tolerance_change=config.tolerance_change,
        history_size=100,
        line_search_fn="strong_wolfe",
    )
    weights_t = torch.as_tensor(sample_weights, dtype=torch.float64, device="cpu")
    last_gradient = float("inf")

    def closure() -> Any:
        nonlocal last_gradient
        optimizer.zero_grad()
        logits = x_t @ weight_t + intercept_t
        loss = torch.nn.functional.softplus(logits) - y_t * logits
        loss = (loss * weights_t).mean() + 0.5 * torch.sum(weight_t * weight_t) / config.l2_c
        loss.backward()
        gradient_parts = [weight_t.grad.detach(), intercept_t.grad.detach().reshape(1)]
        last_gradient = float(torch.cat(gradient_parts).abs().max().detach().cpu())
        return loss

    optimizer.step(closure)
    if not np.isfinite(last_gradient) or last_gradient > config.convergence_grad_tol:
        raise ValueError(f"logistic probe did not converge: gradient={last_gradient:.3e}")
    weights = weight_t.detach().cpu().numpy().astype(np.float64, copy=True)
    intercept = float(intercept_t.detach().cpu())
    if not np.isfinite(weights).all() or not np.isfinite(intercept):
        raise ValueError("logistic probe parameters are non-finite")
    return LogisticProbe(weights, intercept, mean, scale, config)


def binary_token_bow(
    rows: list[dict[str, Any]], vocab_size: int, *, excluded_token_ids: set[int] | frozenset[int] = frozenset()
) -> np.ndarray:
    """Build a binary input-token bag, excluding padding and output classes."""
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    result = np.zeros((len(rows), vocab_size), dtype=np.float64)
    for row_index, row in enumerate(rows):
        ids = np.asarray(row["input_ids"], dtype=np.int64)
        mask = np.asarray(row["attention_mask"], dtype=np.int64)
        if ids.ndim != 1 or mask.shape != ids.shape:
            raise ValueError("token IDs and attention mask are misaligned")
        for token_id in ids[mask.astype(bool)]:
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError("token ID is outside the pinned vocabulary")
            if int(token_id) not in excluded_token_ids:
                result[row_index, int(token_id)] = 1.0
    return result


def matrix_digest(matrix: np.ndarray, *, purpose: str) -> str:
    """Digest a private matrix with explicit shape/dtype/order metadata."""
    values = np.ascontiguousarray(matrix)
    payload = {
        "purpose": purpose,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "order": "C",
        "bytes_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def excluded_columns_digest(row_count: int, token_ids: list[int], *, dtype: str = "float64") -> str:
    """Digest the expected all-zero excluded-token columns without retaining them."""
    if row_count < 0 or sorted(token_ids) != token_ids:
        raise ValueError("excluded-column digest inputs are invalid")
    zeros = np.zeros((row_count, len(token_ids)), dtype=np.dtype(dtype))
    return matrix_digest(zeros, purpose=f"excluded-token-columns:{token_ids}")


def tokenizer_vocab_size(tokenizer: Any, config: Any) -> int:
    value = getattr(tokenizer, "vocab_size", None)
    if value is None:
        value = getattr(config, "vocab_size", None)
    if value is None:
        raise ValueError("pinned tokenizer vocabulary size is unavailable")
    return int(value)


__all__ = [
    "CONVERGENCE_GRAD_TOL",
    "GPT2_VOCAB_SIZE",
    "LBFGS_MAX_ITER",
    "LBFGS_TOLERANCE_CHANGE",
    "LBFGS_TOLERANCE_GRAD",
    "L2_C",
    "LogisticProbe",
    "ProbeConfig",
    "binary_token_bow",
    "excluded_columns_digest",
    "fit_logistic_probe",
    "matrix_digest",
    "tokenizer_vocab_size",
]
