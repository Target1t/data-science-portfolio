"""Leakage-safe segmented calibration for stored out-of-sample forecasts.

The calibrator only learns from completed earlier origins.  It deliberately
uses a small, auditable grid instead of another high-capacity expert model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


QUANTITY_BINS = (-np.inf, 2.0, 5.0, 10.0, 25.0, np.inf)
QUANTITY_LABELS = ("q_lt2", "q_2_5", "q_5_10", "q_10_25", "q_ge25")


@dataclass(frozen=True)
class SegmentChoice:
    threshold_offset: float
    scale_multiplier: float
    rows: int
    actual_volume: float
    inner_mae: float
    inner_ratio: float


def add_segments(frame: pd.DataFrame) -> pd.DataFrame:
    """Add scoring-time observable regime labels."""

    result = frame.copy()
    replenish = pd.to_numeric(result["replenish_days_90"], errors="coerce").fillna(0).gt(0)
    future = pd.to_numeric(result["future_days_90"], errors="coerce").fillna(0).gt(0)
    result["_quadrant"] = np.select(
        [replenish & future, replenish, future],
        ["repl1_fut1", "repl1_fut0", "repl0_fut1"],
        default="repl0_fut0",
    )
    quantity = pd.to_numeric(result["conditional_qty"], errors="coerce").fillna(0).clip(lower=0)
    result["_quantity_bin"] = pd.cut(
        quantity,
        bins=QUANTITY_BINS,
        labels=QUANTITY_LABELS,
        right=False,
    ).astype(str)
    if "listing_age_days" in result:
        age = pd.to_numeric(result["listing_age_days"], errors="coerce").fillna(np.inf)
        result["_age_regime"] = np.where(age.le(90), "cold", "mature")
    else:
        result["_age_regime"] = "unknown"
    return result


def segment_labels(frame: pd.DataFrame, specification: str) -> pd.Series:
    """Return labels for one of the prespecified calibration structures."""

    enriched = add_segments(frame)
    if specification == "global":
        return pd.Series("global", index=enriched.index, dtype=str)
    if specification == "quadrant":
        return enriched["_quadrant"].astype(str)
    if specification == "quantity":
        return enriched["_quantity_bin"].astype(str)
    if specification == "quadrant_quantity":
        return enriched["_quadrant"].astype(str) + "|" + enriched["_quantity_bin"].astype(str)
    if specification == "age_quantity":
        return enriched["_age_regime"].astype(str) + "|" + enriched["_quantity_bin"].astype(str)
    raise ValueError(f"Unknown segment specification: {specification}")


def _base_threshold(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    enriched = add_segments(frame)
    return enriched["_quadrant"].map(thresholds).astype(float).to_numpy()


def _candidate_loss(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    offset: float,
    multiplier: float,
) -> float:
    probability = pd.to_numeric(frame["prob"], errors="coerce").fillna(0).clip(0, 1).to_numpy()
    quantity = pd.to_numeric(frame["conditional_qty"], errors="coerce").fillna(0).clip(lower=0).to_numpy()
    actual = pd.to_numeric(frame["actual_qty"], errors="coerce").fillna(0).clip(lower=0).to_numpy()
    gate = probability >= np.clip(_base_threshold(frame, thresholds) + float(offset), 0, 1)
    prediction = np.where(gate, quantity * float(multiplier), 0.0)
    return float(np.abs(prediction - actual).sum())


def choose_segmented_calibration(
    history: list[pd.DataFrame],
    thresholds: dict[str, float],
    specification: str,
    threshold_offsets: list[float],
    scale_multipliers: list[float],
    min_rows: int = 200,
    min_actual_volume: float = 100.0,
    ratio_floor: float | None = None,
) -> dict[str, SegmentChoice]:
    """Choose independent MAE-minimizing settings, falling back to global."""

    pooled = pd.concat(history, ignore_index=True)
    labels = segment_labels(pooled, specification)
    pooled = pooled.assign(_segment=labels.to_numpy())

    def choose(frame: pd.DataFrame) -> SegmentChoice:
        actual = pd.to_numeric(frame["actual_qty"], errors="coerce").fillna(0).clip(lower=0).to_numpy()
        probability = pd.to_numeric(frame["prob"], errors="coerce").fillna(0).clip(0, 1).to_numpy()
        quantity = pd.to_numeric(frame["conditional_qty"], errors="coerce").fillna(0).clip(lower=0).to_numpy()
        base_threshold = _base_threshold(frame, thresholds)
        actual_volume = float(actual.sum())
        candidates = []
        for offset in threshold_offsets:
            gate = probability >= np.clip(base_threshold + float(offset), 0, 1)
            for multiplier in scale_multipliers:
                prediction = np.where(gate, quantity * float(multiplier), 0.0)
                candidates.append((
                    float(np.abs(prediction - actual).sum()),
                    float(prediction.sum() / max(actual_volume, 1e-12)),
                    abs(float(offset)),
                    abs(float(multiplier) - 1.0),
                    float(offset),
                    float(multiplier),
                ))
        eligible = candidates
        if ratio_floor is not None:
            eligible = [item for item in candidates if item[1] >= ratio_floor]
            if not eligible:
                best_ratio = max(item[1] for item in candidates)
                eligible = [item for item in candidates if item[1] == best_ratio]
        best = min(eligible, key=lambda item: (item[0], abs(1.0 - item[1]), item[2], item[3]))
        return SegmentChoice(best[4], best[5], len(frame), actual_volume, best[0], best[1])

    global_choice = choose(pooled)
    choices = {"__fallback__": global_choice}
    for label, segment in pooled.groupby("_segment", sort=True):
        actual_volume = float(pd.to_numeric(segment["actual_qty"], errors="coerce").fillna(0).clip(lower=0).sum())
        if len(segment) >= min_rows and actual_volume >= min_actual_volume:
            choices[str(label)] = choose(segment)
    return choices


def apply_segmented_calibration(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    specification: str,
    choices: dict[str, SegmentChoice],
) -> pd.DataFrame:
    """Apply fitted segment choices without using any future outcomes."""

    result = frame.copy()
    labels = segment_labels(result, specification)
    fallback = choices["__fallback__"]
    offsets = labels.map(lambda value: choices.get(str(value), fallback).threshold_offset).to_numpy()
    multipliers = labels.map(lambda value: choices.get(str(value), fallback).scale_multiplier).to_numpy()
    probability = pd.to_numeric(result["prob"], errors="coerce").fillna(0).clip(0, 1).to_numpy()
    quantity = pd.to_numeric(result["conditional_qty"], errors="coerce").fillna(0).clip(lower=0).to_numpy()
    gate = probability >= np.clip(_base_threshold(result, thresholds) + offsets, 0, 1)
    result["prediction_qty"] = np.where(gate, quantity * multipliers, 0.0)
    result["calibration_segment"] = labels.to_numpy()
    result["calibration_threshold_offset"] = offsets
    result["calibration_scale_multiplier"] = multipliers
    return result
