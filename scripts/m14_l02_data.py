"""M14 L02 data loading, train-only fitting, and held-out path construction."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from sklearn.datasets import load_digits  # pyright: ignore[reportMissingTypeStubs]

from latent_anything import LatentSpace
from latent_anything.adapters.conv_vae import ConvVAE
from latent_anything.density import GaussianMixtureDensity, GMMConfig
from latent_anything.methods import Lerp
from scripts.m14_l02_plan import section

Array = np.ndarray


def array_digest(value: Array) -> str:
    """Hash array shape, dtype, and bytes without changing the input."""
    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}:{array.shape}".encode("ascii")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def load_and_split_digits(plan: Mapping[str, Any]) -> tuple[Array, Array, Array, Array, Array, Array, str]:
    """Load real digits and make the deterministic train/held-out split."""
    data, model = section(plan, "data"), section(plan, "model")
    dataset = cast(Any, load_digits())
    images = np.asarray(dataset.images, dtype=np.float64)[:, None, :, :] / 16.0
    labels = np.asarray(dataset.target, dtype=np.int64)
    content_digest = hashlib.sha256((array_digest(images) + array_digest(labels)).encode("ascii")).hexdigest()
    rng = np.random.default_rng(int(model.get("random_state", 42)))
    indices = rng.permutation(len(images))
    train_count = int(data.get("train_samples", round(len(images) * 0.8)))
    if not 1 <= train_count < len(images):
        raise ValueError("L02 split must leave a non-empty held-out partition")
    train_indices, heldout_indices = indices[:train_count], indices[train_count:]
    return (
        images[train_indices],
        images[heldout_indices],
        labels[train_indices],
        labels[heldout_indices],
        train_indices,
        heldout_indices,
        content_digest,
    )


def fit_train_only_conv_vae(train_images: Array, plan: Mapping[str, Any]) -> ConvVAE:
    """Fit the existing ConvVAE only on declared training images."""
    model = section(plan, "model")
    adapter = ConvVAE(
        latent_dim=int(model["latent_dim"]), random_state=int(model["random_state"]), n_epochs=int(model["epochs"])
    )
    adapter.fit(np.array(train_images, dtype=np.float64, copy=True))
    return adapter


def fit_train_only_density(train_latents: Array, plan: Mapping[str, Any]) -> GaussianMixtureDensity:
    """Fit the density oracle only on train latents."""
    spec = section(plan, "density")
    config = spec.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("L02 density config must be an object")
    density = GaussianMixtureDensity(GMMConfig(**dict(config)))
    return density.fit(
        np.array(train_latents, dtype=np.float64, copy=True),
        source_representation_identity=str(spec["identity"]),
        geometry="euclidean",
        provenance={"fit_scope": "train latents only"},
    )


def _select_pair_indices(labels: Array, count: int, seed: int) -> list[tuple[int, int, bool]]:
    rng = np.random.default_rng(seed)
    candidates: dict[bool, list[tuple[int, int]]] = {True: [], False: []}
    for first in range(len(labels)):
        for second in range(first + 1, len(labels)):
            candidates[bool(labels[first] == labels[second])].append((first, second))
    per_class = count // 2
    selected: list[tuple[int, int, bool]] = []
    for same in (True, False):
        if len(candidates[same]) < per_class:
            raise ValueError("held-out labels do not provide enough balanced pairs")
        chosen = rng.choice(len(candidates[same]), size=per_class, replace=False)
        selected.extend((candidates[same][int(index)][0], candidates[same][int(index)][1], same) for index in chosen)
    return selected


def resample_rows(values: Array, count: int) -> Array:
    coordinates, target = np.linspace(0.0, 1.0, len(values)), np.linspace(0.0, 1.0, count)
    return np.column_stack([np.interp(target, coordinates, values[:, column]) for column in range(values.shape[1])])


def build_heldout_latent_paths(
    heldout_latents: Array,
    heldout_images: Array,
    labels: Array,
    plan: Mapping[str, Any],
    latent_space: LatentSpace | None = None,
) -> dict[str, Any]:
    """Build held-out interpolation pair paths before scoring."""
    model, execution = section(plan, "model"), section(plan, "execution")
    records = plan.get("records")
    if (
        not isinstance(records, list)
        or not records
        or not isinstance(records[0], Mapping)
        or not isinstance(records[0].get("acceptance"), Mapping)
    ):
        raise ValueError("manifold record must declare pair_count_min")
    pair_count = int(records[0]["acceptance"]["pair_count_min"])
    pairs = _select_pair_indices(labels, pair_count, int(model["random_state"]))
    euclidean = latent_space or LatentSpace(dim=int(model["latent_dim"]), source_model="conv_vae_8x8")
    spherical = LatentSpace(dim=int(model["latent_dim"]), geometry="unit_norm", source_model="conv_vae_8x8")
    method = Lerp(space=euclidean)
    n_points = int(execution["path_points"])
    pair_paths: list[dict[str, Any]] = []
    for first, second, same in pairs:
        a, b = (
            np.array(heldout_latents[first], dtype=np.float64, copy=True),
            np.array(heldout_latents[second], dtype=np.float64, copy=True),
        )
        lerp = np.asarray([method(a, b, float(t)) for t in np.linspace(0.0, 1.0, n_points)])
        a_spherical, b_spherical = spherical.normalize(a), spherical.normalize(b)
        slerp = np.asarray(
            [spherical.interpolate(a_spherical, b_spherical, float(t)) for t in np.linspace(0.0, 1.0, n_points)]
        )
        pair_paths.append(
            {
                "a": a,
                "b": b,
                "pair_indices": (first, second),
                "same_label": same,
                "lerp": lerp,
                "slerp": slerp,
                "pixels_a": np.array(heldout_images[first], dtype=np.float64, copy=True),
                "pixels_b": np.array(heldout_images[second], dtype=np.float64, copy=True),
            }
        )
    return {"pairs": pair_paths, "euclidean_space": euclidean, "spherical_space": spherical}
