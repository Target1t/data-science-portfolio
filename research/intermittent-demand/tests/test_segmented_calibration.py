import pandas as pd

from phase9.segmented_calibration import (
    apply_segmented_calibration,
    choose_segmented_calibration,
    segment_labels,
)


THRESHOLDS = {
    "repl0_fut0": 0.30,
    "repl0_fut1": 0.25,
    "repl1_fut0": 0.35,
    "repl1_fut1": 0.20,
}


def _frame():
    return pd.DataFrame({
        "actual_qty": [0.0, 4.0, 8.0, 30.0],
        "prob": [0.1, 0.4, 0.5, 0.8],
        "conditional_qty": [1.0, 3.0, 7.0, 28.0],
        "prediction_qty": [0.0, 3.0, 7.0, 28.0],
        "replenish_days_90": [0, 0, 1, 1],
        "future_days_90": [0, 1, 0, 1],
        "listing_age_days": [20, 100, 200, 40],
    })


def test_segment_labels_use_only_observable_fields():
    labels = segment_labels(_frame(), "quadrant_quantity")
    assert labels.tolist() == [
        "repl0_fut0|q_lt2", "repl0_fut1|q_2_5",
        "repl1_fut0|q_5_10", "repl1_fut1|q_ge25",
    ]


def test_segmented_calibration_can_select_and_apply():
    history = [_frame()] * 3
    choices = choose_segmented_calibration(
        history, THRESHOLDS, "quantity", [0.0], [1.0, 1.2],
        min_rows=1, min_actual_volume=0,
    )
    result = apply_segmented_calibration(_frame(), THRESHOLDS, "quantity", choices)
    assert "calibration_segment" in result
    assert result.loc[0, "prediction_qty"] == 0.0
    assert result.loc[3, "prediction_qty"] >= 28.0


def test_ratio_floor_prevents_overly_conservative_choice():
    choices = choose_segmented_calibration(
        [_frame()] * 3, THRESHOLDS, "global", [0.0], [0.5, 1.0],
        min_rows=1, min_actual_volume=0, ratio_floor=0.7,
    )
    assert choices["__fallback__"].inner_ratio >= 0.7
