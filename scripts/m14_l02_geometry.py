"""Executable orchestration for the bounded M14 L02 geometry lane.

Importing this module is side-effect free. Only ``main`` performs the explicit
real-data run and writes the future artifact/run record.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from latent_anything.adapters.conv_vae import ConvVAE
from scripts.m14_l02_data import (
    array_digest,
    build_heldout_latent_paths,
    fit_train_only_conv_vae,
    fit_train_only_density,
    load_and_split_digits,
)
from scripts.m14_l02_envelope import build_payload, input_digests, write_artifact_and_run_record
from scripts.m14_l02_metrics import evaluate_independent_records
from scripts.m14_l02_plan import load_plan, section


def run_l02_benchmark(plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Execute the bounded lane when explicitly called by ``main``."""
    active_plan = load_plan() if plan is None else plan
    (
        train_images,
        heldout_images,
        train_labels,
        heldout_labels,
        train_indices,
        heldout_indices,
        content_digest,
    ) = load_and_split_digits(active_plan)
    source_arrays = {
        "train_images": train_images,
        "heldout_images": heldout_images,
        "train_labels": train_labels,
        "heldout_labels": heldout_labels,
    }
    before = input_digests(source_arrays)
    adapter: ConvVAE = fit_train_only_conv_vae(train_images, active_plan)
    train_latents = adapter.encode_value(train_images).to_numpy()
    heldout_latents = adapter.encode_value(heldout_images).to_numpy()
    density = fit_train_only_density(train_latents, active_plan)
    paths = build_heldout_latent_paths(
        heldout_latents, heldout_images, heldout_labels, active_plan, adapter.latent_space
    )
    records = evaluate_independent_records(paths, density, active_plan)
    after = input_digests(source_arrays)
    split_metadata = {
        "dataset": section(active_plan, "data")["dataset"],
        "license": section(active_plan, "data")["license"],
        "content_sha256": content_digest,
        "total_samples": len(train_indices) + len(heldout_indices),
        "train_samples": len(train_indices),
        "heldout_samples": len(heldout_indices),
        "train_index_sha256": array_digest(train_indices),
        "heldout_index_sha256": array_digest(heldout_indices),
    }
    return build_payload(
        active_plan,
        records,
        adapter,
        density,
        split_metadata,
        before,
        after,
        {"train_latents": array_digest(train_latents), "heldout_latents": array_digest(heldout_latents)},
    )


def main() -> None:
    """Run the approved lane and write its shared artifact/run record."""
    plan = load_plan()
    write_artifact_and_run_record(plan, run_l02_benchmark(plan))


if __name__ == "__main__":
    main()
