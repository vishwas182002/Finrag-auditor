"""Latency summaries with explicit percentile interpolation."""

from __future__ import annotations

import numpy as np


def latency_summary(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for row in rows for key in row})
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if key in row], dtype=float)
        if len(values):
            summary[key] = {
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
                "mean_ms": float(np.mean(values)),
            }
    return summary

