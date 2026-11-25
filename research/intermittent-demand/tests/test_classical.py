import pandas as pd

from phase9.classical import croston_daily, forecast_frame, tsb_daily


def test_croston_and_sba_on_daily_constant_demand() -> None:
    dates = pd.Series(["2026-01-01", "2026-01-02"])
    qty = pd.Series([2.0, 2.0])

    assert croston_daily(dates, qty, alpha=0.1) == 2.0
    assert croston_daily(dates, qty, alpha=0.1, bias_adjusted=True) == 1.9


def test_tsb_probability_decays_after_zero_demand_days() -> None:
    dates = pd.Series(["2026-01-01"])
    qty = pd.Series([2.0])

    forecast = tsb_daily(dates, qty, anchor=pd.Timestamp("2026-01-12"), beta=0.1)

    assert abs(forecast - 2.0 * 0.9**10) < 1e-12


def test_single_event_initializes_from_observation_start() -> None:
    dates = pd.Series(["2026-01-10"])
    qty = pd.Series([10.0])
    start = pd.Timestamp("2026-01-01")

    assert croston_daily(dates, qty, observation_start=start) == 1.0
    assert tsb_daily(
        dates, qty, anchor=pd.Timestamp("2026-01-11"), observation_start=start
    ) == 1.0


def test_forecast_frame_returns_horizon_totals() -> None:
    history = pd.DataFrame({
        "sku_id": ["a", "a", "b"],
        "date": ["2026-01-01", "2026-01-02", "2026-01-01"],
        "qty_replenish": [2.0, 2.0, 0.0],
    })

    result = forecast_frame(history, pd.Timestamp("2026-01-03"), horizon_days=30)

    assert result.loc[result.sku_id.eq("a"), "croston"].iloc[0] == 60.0
    assert result.loc[result.sku_id.eq("b"), "tsb"].iloc[0] == 0.0
