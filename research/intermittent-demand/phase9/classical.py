"""Classical intermittent-demand baselines for paper comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _positive_events(dates: pd.Series, quantities: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({
        "date": pd.to_datetime(dates, errors="coerce"),
        "qty": pd.to_numeric(quantities, errors="coerce"),
    })
    frame = frame.dropna(subset=["date"]).assign(qty=lambda x: x.qty.fillna(0.0).clip(lower=0.0))
    frame = frame.loc[frame.qty.gt(0)].groupby("date", as_index=False).qty.sum().sort_values("date")
    return frame.date.to_numpy(), frame.qty.to_numpy(dtype=float)


def croston_daily(
    dates: pd.Series,
    quantities: pd.Series,
    alpha: float = 0.1,
    bias_adjusted: bool = False,
    observation_start: pd.Timestamp | None = None,
) -> float:
    """Return Croston or SBA expected demand per day."""

    event_dates, event_qty = _positive_events(dates, quantities)
    if len(event_qty) == 0:
        return 0.0
    size = float(event_qty[0])
    last = pd.Timestamp(event_dates[0])
    start = pd.Timestamp(observation_start) if observation_start is not None and pd.notna(observation_start) else last
    interval = float(max((last - start).days + 1, 1))
    for raw_date, qty in zip(event_dates[1:], event_qty[1:]):
        date = pd.Timestamp(raw_date)
        gap = max((date - last).days, 1)
        size += alpha * (float(qty) - size)
        interval += alpha * (float(gap) - interval)
        last = date
    forecast = size / max(interval, 1e-9)
    if bias_adjusted:
        forecast *= 1.0 - alpha / 2.0
    return max(float(forecast), 0.0)


def tsb_daily(
    dates: pd.Series,
    quantities: pd.Series,
    anchor: pd.Timestamp,
    alpha: float = 0.1,
    beta: float = 0.1,
    observation_start: pd.Timestamp | None = None,
) -> float:
    """Return TSB probability-times-size demand per day without a dense panel."""

    event_dates, event_qty = _positive_events(dates, quantities)
    if len(event_qty) == 0:
        return 0.0
    size = float(event_qty[0])
    last = pd.Timestamp(event_dates[0])
    start = pd.Timestamp(observation_start) if observation_start is not None and pd.notna(observation_start) else last
    initial_interval = float(max((last - start).days + 1, 1))
    probability = 1.0 / initial_interval
    for raw_date, qty in zip(event_dates[1:], event_qty[1:]):
        date = pd.Timestamp(raw_date)
        zero_days = max((date - last).days - 1, 0)
        probability *= (1.0 - beta) ** zero_days
        probability = (1.0 - beta) * probability + beta
        size += alpha * (float(qty) - size)
        last = date
    zero_days_after_last = max((pd.Timestamp(anchor).normalize() - last).days - 1, 0)
    probability *= (1.0 - beta) ** zero_days_after_last
    return max(float(probability * size), 0.0)


def forecast_frame(
    history: pd.DataFrame,
    anchor: pd.Timestamp,
    horizon_days: int = 30,
    alpha: float = 0.1,
    beta: float = 0.1,
) -> pd.DataFrame:
    """Forecast all observed SKUs with Croston, SBA and TSB."""

    required = {"sku_id", "date", "qty_replenish"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"History missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for sku_id, group in history.groupby("sku_id", sort=False):
        observation_start = None
        if "listing_date" in group:
            listing_dates = pd.to_datetime(group.listing_date, errors="coerce").dropna()
            if not listing_dates.empty:
                observation_start = listing_dates.min()
        croston = croston_daily(
            group.date, group.qty_replenish, alpha=alpha, observation_start=observation_start
        )
        sba = croston_daily(
            group.date, group.qty_replenish, alpha=alpha, bias_adjusted=True,
            observation_start=observation_start,
        )
        tsb = tsb_daily(
            group.date, group.qty_replenish, anchor=anchor, alpha=alpha, beta=beta,
            observation_start=observation_start,
        )
        rows.append({
            "sku_id": str(sku_id),
            "croston": croston * horizon_days,
            "sba": sba * horizon_days,
            "tsb": tsb * horizon_days,
        })
    return pd.DataFrame(rows)
