"""Deterministic non-parametric bootstrap intervals."""

from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    values: list[float], samples: int = 1000, seed: int = 42
) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    array = np.asarray(values, dtype=float)
    if samples <= 0:
        mean = float(np.mean(array))
        return {"mean": mean, "ci_low": mean, "ci_high": mean}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(np.mean(array)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }

