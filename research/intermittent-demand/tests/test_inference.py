from phase9.inference import forecast_window


def test_final_forecast_window_starts_after_latest_observed_date() -> None:
    assert forecast_window("2026-08-16", 30) == ("2026-08-17", "2026-09-15")
