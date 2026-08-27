"""Leakage-safe glyph prompts and deterministic grouped split construction."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from sklearn.datasets import load_digits  # pyright: ignore[reportMissingTypeStubs]
from sklearn.model_selection import StratifiedGroupKFold  # pyright: ignore[reportMissingTypeStubs]


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}:{array.shape}".encode("ascii")
    return digest_bytes(header + b"\0" + array.tobytes())


def glyph_prompt(image: np.ndarray) -> str:
    """Render an 8x8 image without its digit identity or numeric target."""
    pixels = np.asarray(image)
    if pixels.shape != (8, 8):
        raise ValueError(f"expected an 8x8 image, got {pixels.shape}")
    rows = ["".join("#" if float(pixel) >= 8.0 else "." for pixel in row) for row in pixels]
    return "Classify this 8x8 glyph:\n" + "\n".join(rows)


def _prompt_digest(prompt: str) -> str:
    return digest_bytes(prompt.encode("utf-8"))


def validate_group_labels(prompt_digests: np.ndarray, labels: np.ndarray) -> None:
    """Reject one prompt group carrying contradictory labels."""
    labels_by_group: dict[str, set[int]] = {}
    for digest, label in zip(prompt_digests, labels, strict=True):
        labels_by_group.setdefault(str(digest), set()).add(int(label))
    conflicts = {key: sorted(value) for key, value in labels_by_group.items() if len(value) > 1}
    if conflicts:
        raise ValueError(f"prompt digest maps to multiple labels: {conflicts}")


def grouped_digit_split(seed: int = 79) -> dict[str, Any]:
    """Build deterministic 60/20/20 partitions, grouping equal prompts."""
    dataset = cast(Any, load_digits())
    images = np.asarray(dataset.images, dtype=np.float64)
    labels = np.asarray(dataset.target, dtype=np.int64)
    prompts = np.asarray([glyph_prompt(image) for image in images], dtype=object)
    prompt_digests = np.asarray([_prompt_digest(str(prompt)) for prompt in prompts], dtype=str)
    validate_group_labels(prompt_digests, labels)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_id = np.full(len(labels), -1, dtype=np.int64)
    dummy = np.zeros((len(labels), 1), dtype=np.float32)
    for fold, (_train, test) in enumerate(splitter.split(dummy, labels, prompt_digests)):
        fold_id[test] = fold
    if np.any(fold_id < 0):
        raise RuntimeError("group splitter left samples without a fold")
    partitions = {
        name: np.isin(fold_id, folds) for name, folds in {"train": (0, 1, 2), "val": (3,), "test": (4,)}.items()
    }
    groups = {name: set(prompt_digests[mask].tolist()) for name, mask in partitions.items()}
    if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]:
        raise AssertionError("prompt groups overlap across partitions")
    return {
        "images": images,
        "labels": labels,
        "prompts": prompts,
        "prompt_digests": prompt_digests,
        "fold_id": fold_id,
        "partitions": partitions,
        "metadata": split_metadata(images, labels, prompts, prompt_digests, partitions, fold_id),
    }


def split_metadata(
    images: np.ndarray,
    labels: np.ndarray,
    prompts: np.ndarray,
    prompt_digests: np.ndarray,
    partitions: Mapping[str, np.ndarray],
    fold_id: np.ndarray,
) -> dict[str, Any]:
    """Return auditable counts, class distributions, and input digests."""
    return {
        "dataset": "sklearn.datasets.load_digits",
        "total_samples": int(len(labels)),
        "partition_counts": {name: int(mask.sum()) for name, mask in partitions.items()},
        "class_counts": {
            name: dict(sorted((str(k), int(v)) for k, v in Counter(labels[mask]).items()))
            for name, mask in partitions.items()
        },
        "group_counts": {name: int(np.unique(prompt_digests[mask]).size) for name, mask in partitions.items()},
        "content_sha256": array_digest(images),
        "label_sha256": array_digest(labels),
        "prompt_sha256": digest_bytes("\n".join(str(item) for item in prompts).encode()),
        "prompt_digest_sha256": array_digest(prompt_digests),
        "fold_sha256": array_digest(fold_id),
        "group_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
    }
