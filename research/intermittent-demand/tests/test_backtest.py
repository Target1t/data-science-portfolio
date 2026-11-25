from pathlib import Path

from phase9.backtest import training_anchor_dates


def test_training_anchors_end_before_evaluation_purge_window() -> None:
    anchors = training_anchor_dates(
        first_observed="2025-01-01", evaluation_anchor="2026-04-12", config={"lookback_days": 90, "forecast_days": 30, "purge_days": 30}
    )

    assert anchors
    assert max(anchors) + __import__("pandas").Timedelta(days=30) <= __import__("pandas").Timestamp("2026-04-12") - __import__("pandas").Timedelta(days=30)
