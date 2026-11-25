"""Prequential calibration for immutable rolling-origin prediction contexts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationChoice:
    threshold_offset: float
    scale_multiplier: float
    inner_wmape: float
    inner_ratio: float


def apply_calibration(
    context: pd.DataFrame,
    quadrant_thresholds: dict[str, float],
    threshold_offset: float,
    scale_multiplier: float,
) -> pd.DataFrame:
    """Apply a gate/scale choice to stored out-of-sample probabilities and quantities."""

    frame = context.copy()
    replenish = pd.to_numeric(frame["replenish_days_90"], errors="coerce").fillna(0).gt(0)
    future = pd.to_numeric(frame["future_days_90"], errors="coerce").fillna(0).gt(0)
    threshold = np.select(
        [replenish & future, replenish, future],
        [
            quadrant_thresholds["repl1_fut1"],
            quadrant_thresholds["repl1_fut0"],
            quadrant_thresholds["repl0_fut1"],
        ],
        default=quadrant_thresholds["repl0_fut0"],
    )
    threshold = np.clip(threshold + float(threshold_offset), 0.0, 1.0)
    quantity = pd.to_numeric(frame["conditional_qty"], errors="coerce").fillna(0).clip(lower=0)
    quantity = quantity * float(scale_multiplier)
    probability = pd.to_numeric(frame["prob"], errors="coerce").fillna(0).clip(0, 1)
    frame["prediction_qty"] = np.where(probability.to_numpy() >= threshold, quantity, 0.0)
    return frame


def pooled_loss(contexts: list[pd.DataFrame]) -> tuple[float, float]:
    """Return pooled WMAPE and forecast-to-actual ratio."""

    frame = pd.concat(contexts, ignore_index=True)
    actual = pd.to_numeric(frame["actual_qty"], errors="coerce").fillna(0).clip(lower=0)
    prediction = pd.to_numeric(frame["prediction_qty"], errors="coerce").fillna(0).clip(lower=0)
    total = max(float(actual.sum()), 1e-12)
    return float((prediction - actual).abs().sum() / total), float(prediction.sum() / total)


def choose_calibration(
    history: list[pd.DataFrame],
    quadrant_thresholds: dict[str, float],
    threshold_offsets: list[float],
    scale_multipliers: list[float],
    ratio_floor: float | None = None,
) -> CalibrationChoice:
    """Choose the lowest prior-origin WMAPE with deterministic conservative tie breaks."""

    choices: list[CalibrationChoice] = []
    for offset in threshold_offsets:
        for multiplier in scale_multipliers:
            calibrated = [
                apply_calibration(frame, quadrant_thresholds, offset, multiplier)
                for frame in history
            ]
            wmape, ratio = pooled_loss(calibrated)
            choices.append(CalibrationChoice(offset, multiplier, wmape, ratio))
    eligible = choices
    if ratio_floor is not None:
        above_floor = [item for item in choices if item.inner_ratio >= ratio_floor]
        if above_floor:
            eligible = above_floor
        else:
            best_ratio = max(item.inner_ratio for item in choices)
            eligible = [item for item in choices if item.inner_ratio == best_ratio]
    return min(
        eligible,
        key=lambda item: (
            item.inner_wmape,
            abs(1.0 - item.inner_ratio),
            abs(item.threshold_offset),
            abs(item.scale_multiplier - 1.0),
        ),
    )
