"""Synthetic, model-free power sensitivity simulation for L04.9 v2."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from scripts._m14_l049_v2_schema import BOOTSTRAP_REPLICATES, POWER_SIMULATION_SEED, canonical_json_bytes

POWER_SIMULATION_SCHEMA = "m14-l04.9-v2-power-v1"
POWER_ASSUMPTIONS: dict[str, Any] = {
    "groups": 24,
    "simulations": 2000,
    "effect_mean": 0.16,
    "effect_sd": 0.30,
    "null_mean": 0.0,
    "decision_lower_ci": 0.05,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
}


def _lower_ci(values: np.ndarray, rng: np.random.Generator) -> float:
    """Compute the preregistered lower bound using all declared resamples."""
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))]
    return float(np.quantile(np.mean(draws, axis=1), 0.05))


def run_power_simulation(seed: int = POWER_SIMULATION_SEED) -> dict[str, Any]:
    """Estimate sensitivity from synthetic group effects only.

    This is a preregistration sensitivity check, not model evidence. The
    intentionally modest power is retained as an explicit false-negative risk.
    """
    rng = np.random.default_rng(int(seed))
    alternatives = np.asarray(
        [
            _lower_ci(
                rng.normal(
                    POWER_ASSUMPTIONS["effect_mean"], POWER_ASSUMPTIONS["effect_sd"], POWER_ASSUMPTIONS["groups"]
                ),
                rng,
            )
            > POWER_ASSUMPTIONS["decision_lower_ci"]
            for _ in range(POWER_ASSUMPTIONS["simulations"])
        ],
        dtype=np.float64,
    )
    nulls = np.asarray(
        [
            _lower_ci(
                rng.normal(POWER_ASSUMPTIONS["null_mean"], POWER_ASSUMPTIONS["effect_sd"], POWER_ASSUMPTIONS["groups"]),
                rng,
            )
            > POWER_ASSUMPTIONS["decision_lower_ci"]
            for _ in range(POWER_ASSUMPTIONS["simulations"])
        ],
        dtype=np.float64,
    )
    power = float(np.mean(alternatives))
    false_negative_risk = float(1.0 - power)
    false_positive_rate = float(np.mean(nulls))
    return {
        "schema_version": POWER_SIMULATION_SCHEMA,
        "seed": int(seed),
        "assumptions": dict(POWER_ASSUMPTIONS),
        "result": {
            "power": power,
            "false_negative_risk": false_negative_risk,
            "false_positive_rate": false_positive_rate,
            "accepted_false_negative_risk": True,
        },
    }


def power_digest(result: Mapping[str, Any]) -> str:
    unsigned = dict(result)
    unsigned.pop("digest_sha256", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def frozen_power_result() -> dict[str, Any]:
    result = run_power_simulation()
    result["digest_sha256"] = power_digest(result)
    return result


def validate_power_result(result: Mapping[str, Any]) -> list[str]:
    """Independently validate the frozen assumptions, 2,000 draws, and digest."""
    errors: list[str] = []
    if result.get("schema_version") != POWER_SIMULATION_SCHEMA or result.get("seed") != POWER_SIMULATION_SEED:
        errors.append("power simulation schema or seed is invalid")
    if result.get("assumptions") != POWER_ASSUMPTIONS:
        errors.append("power simulation assumptions are invalid")
    expected = frozen_power_result()
    if result.get("result") != expected.get("result"):
        errors.append("power simulation result was not independently recomputed")
    try:
        recomputed_digest = power_digest(result)
    except (TypeError, ValueError, OverflowError):
        recomputed_digest = None
    if result.get("digest_sha256") != recomputed_digest or result.get("digest_sha256") != expected["digest_sha256"]:
        errors.append("power simulation digest is invalid")
    return errors


__all__ = [
    "POWER_ASSUMPTIONS",
    "POWER_SIMULATION_SCHEMA",
    "frozen_power_result",
    "power_digest",
    "run_power_simulation",
    "validate_power_result",
]
