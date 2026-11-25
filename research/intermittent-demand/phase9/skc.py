"""Auditable SKU-to-SKC aggregation and bounded historical calibration."""

from __future__ import annotations

import pandas as pd


def build_skc_forecast(sku_forecast: pd.DataFrame, factors: dict[int, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"style_id", "forecast_qty_h30"}; missing = required.difference(sku_forecast.columns)
    if missing: raise ValueError(f"SKU forecast missing columns: {sorted(missing)}")
    metadata = [c for c in ("product_name", "category", "sub_category", "season", "band", "series", "price_tag", "color_id", "listing_date", "forecast_start", "forecast_end") if c in sku_forecast]
    grouped = sku_forecast.groupby("style_id", as_index=False).agg(**{c: (c, "first") for c in metadata}, sku_sum_qty=("forecast_qty_h30", "sum"))
    grouped["prediction_band"] = pd.cut(grouped.sku_sum_qty, [0, 3, 10, float("inf")], right=False, labels=False, include_lowest=True).astype(int)
    grouped["skc_calibration_factor"] = grouped.prediction_band.map(factors).fillna(1.0).astype(float)
    grouped["forecast_qty_h30"] = (grouped.sku_sum_qty * grouped.skc_calibration_factor).round(6)
    client = grouped.loc[:, ["style_id", *metadata, "forecast_qty_h30"]].copy()
    return client, grouped
