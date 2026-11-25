import pandas as pd

from phase9.metrics import phase7_scorecard


def test_scorecard_reports_phase7_core_and_tail_metrics() -> None:
    context = pd.DataFrame(
        {
            "sku_id": ["a", "b", "c", "d"],
            "actual_qty": [20.0, 0.0, 5.0, 1.0],
            "prediction_qty": [15.0, 4.0, 0.0, 1.0],
            "prob": [0.9, 0.8, 0.1, 0.7],
            "category": ["coat", "coat", "tee", "tee"],
            "replenish_days_90": [4, 0, 0, 2],
            "future_days_90": [1, 0, 2, 1],
        }
    )

    score = phase7_scorecard(context)

    required = {
        "auc",
        "average_precision",
        "brier_score",
        "ece_10",
        "f1",
        "global_ratio",
        "global_wmape",
        "false_zero_rate",
        "top20_true_volume_capture",
        "blockbuster_under_wape",
        "4_25_under_wape",
        "repl0_fut1_rows",
        "category_worst5_ratio",
    }
    assert required.issubset(score)
    assert score["global_ratio"] == 20.0 / 26.0
