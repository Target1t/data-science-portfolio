"""Paired hierarchical bootstrap utilities for forecast comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def group_losses(base: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paired absolute errors by SKC, or by SKU if SKC is absent."""

    keys = ["sku_id"]
    if "style_id" in base.columns and "style_id" in candidate.columns:
        keys = ["style_id", "sku_id"]
    left = base.loc[:, [*keys, "actual_qty", "prediction_qty"]].rename(
        columns={"prediction_qty": "base_prediction"}
    )
    right = candidate.loc[:, [*keys, "prediction_qty"]].rename(
        columns={"prediction_qty": "candidate_prediction"}
    )
    frame = left.merge(right, on=keys, how="inner", validate="one_to_one")
    frame["base_error"] = (frame.base_prediction - frame.actual_qty).abs()
    frame["candidate_error"] = (frame.candidate_prediction - frame.actual_qty).abs()
    group_key = "style_id" if "style_id" in frame else "sku_id"
    return frame.groupby(group_key, as_index=False).agg(
        actual=("actual_qty", "sum"),
        base_error=("base_error", "sum"),
        candidate_error=("candidate_error", "sum"),
    )


def hierarchical_bootstrap_delta(
    grouped_anchors: list[pd.DataFrame],
    iterations: int = 2000,
    seed: int = 2028,
) -> dict[str, float]:
    """Bootstrap candidate-minus-base WMAPE across origins and SKCs."""

    if not grouped_anchors:
        raise ValueError("At least one paired anchor is required")
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=float)
    anchor_count = len(grouped_anchors)
    for iteration in range(iterations):
        actual = base_error = candidate_error = 0.0
        for anchor_index in rng.integers(0, anchor_count, size=anchor_count):
            groups = grouped_anchors[int(anchor_index)]
            sampled = groups.iloc[rng.integers(0, len(groups), size=len(groups))]
            actual += float(sampled.actual.sum())
            base_error += float(sampled.base_error.sum())
            candidate_error += float(sampled.candidate_error.sum())
        deltas[iteration] = (candidate_error - base_error) / max(actual, 1e-12)
    return {
        "mean_delta": float(deltas.mean()),
        "ci_lower": float(np.quantile(deltas, 0.025)),
        "ci_upper": float(np.quantile(deltas, 0.975)),
        "probability_candidate_better": float((deltas < 0).mean()),
        "iterations": int(iterations),
    }
