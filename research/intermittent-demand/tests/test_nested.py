import pandas as pd

from phase9.nested import apply_calibration, choose_calibration


THRESHOLDS = {
    "repl0_fut0": 0.30,
    "repl0_fut1": 0.20,
    "repl1_fut0": 0.40,
    "repl1_fut1": 0.25,
}


def _context() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "actual_qty": [0.0, 2.0, 4.0, 5.0],
            "conditional_qty": [1.0, 2.0, 4.0, 5.0],
            "prob": [0.29, 0.21, 0.39, 0.26],
            "replenish_days_90": [0, 0, 1, 1],
            "future_days_90": [0, 1, 0, 1],
        }
    )


def test_apply_calibration_uses_quadrant_gates_and_scale() -> None:
    result = apply_calibration(_context(), THRESHOLDS, threshold_offset=0.0, scale_multiplier=2.0)
    assert result["prediction_qty"].tolist() == [0.0, 4.0, 0.0, 10.0]


def test_choose_calibration_uses_prior_context_loss() -> None:
    choice = choose_calibration([_context()], THRESHOLDS, [0.0, -0.02], [1.0])
    assert choice.threshold_offset == -0.02


def test_choose_calibration_respects_ratio_floor() -> None:
    choice = choose_calibration(
        [_context()], THRESHOLDS, [0.0], [0.5, 1.0], ratio_floor=0.7
    )
    assert choice.scale_multiplier == 1.0
