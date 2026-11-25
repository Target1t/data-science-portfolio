"""Leakage-safe, SKU-grain Phase9 feature construction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


STATIC_COLUMNS = (
    "style_id",
    "product_name",
    "category",
    "sub_category",
    "season",
    "band",
    "series",
    "price_tag",
    "size_id",
    "color_id",
    "listing_date",
)


def _as_dates(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], errors="coerce").dt.normalize()


def _window_features(history: pd.DataFrame, anchor: pd.Timestamp, days: int) -> pd.DataFrame:
    start = anchor - pd.Timedelta(days=days)
    window = history.loc[history["date"].ge(start)].copy()
    grouped = window.groupby("sku_id", as_index=False).agg(
        **{
            f"replenish_sum_{days}": ("qty_replenish", "sum"),
            f"future_sum_{days}": ("qty_future", "sum"),
            f"replenish_days_{days}": ("qty_replenish", lambda values: int((values > 0).sum())),
            f"future_days_{days}": ("qty_future", lambda values: int((values > 0).sum())),
            f"replenish_max_{days}": ("qty_replenish", "max"),
            f"future_max_{days}": ("qty_future", "max"),
        }
    )
    return grouped


def _style_window_features(history: pd.DataFrame, anchor: pd.Timestamp, days: int) -> pd.DataFrame:
    """Aggregate sibling SKU activity without exposing the style identity itself."""

    start = anchor - pd.Timedelta(days=days)
    window = history.loc[history["date"].ge(start)]
    return window.groupby("style_id", as_index=False).agg(
        **{
            f"skc_replenish_sum_{days}": ("qty_replenish", "sum"),
            f"skc_future_sum_{days}": ("qty_future", "sum"),
            f"skc_replenish_days_{days}": ("qty_replenish", lambda values: int((values > 0).sum())),
            f"skc_future_days_{days}": ("qty_future", lambda values: int((values > 0).sum())),
            f"skc_active_skus_{days}": ("sku_id", "nunique"),
        }
    )


def _add_sibling_features(base: pd.DataFrame, days: int) -> pd.DataFrame:
    """Remove the target SKU from SKC totals and expose its historical share."""

    derived: dict[str, pd.Series] = {}
    for signal in ("replenish", "future"):
        sku_sum = base[f"{signal}_sum_{days}"].fillna(0.0)
        skc_sum = base[f"skc_{signal}_sum_{days}"].fillna(0.0)
        derived[f"sibling_{signal}_sum_{days}"] = (skc_sum - sku_sum).clip(lower=0.0)
        derived[f"sku_{signal}_share_skc_{days}"] = pd.Series(
            np.where(skc_sum > 0, sku_sum / skc_sum, 0.0), index=base.index
        )
    derived[f"sibling_active_skus_{days}"] = (
        base[f"skc_active_skus_{days}"].fillna(0.0) - 1.0
    ).clip(lower=0.0)
    return pd.concat([base, pd.DataFrame(derived, index=base.index)], axis=1)


def _add_engineered_features(
    base: pd.DataFrame,
    history: pd.DataFrame,
    anchor: pd.Timestamp,
    include_operational: bool = False,
    windows: tuple[int, ...] = (7, 30, 90),
) -> pd.DataFrame:
    """Add leakage-safe calendar, recency, rate, and momentum features."""

    positive_repl = history.loc[history["qty_replenish"].gt(0)].groupby("sku_id")["date"].max()
    positive_future = history.loc[history["qty_future"].gt(0)].groupby("sku_id")["date"].max()
    derived: dict[str, pd.Series | float] = {
        "replenish_recency_days": base["sku_id"].map(positive_repl).rsub(anchor).dt.days.fillna(999.0),
        "future_recency_days": base["sku_id"].map(positive_future).rsub(anchor).dt.days.fillna(999.0),
    }
    for signal in ("replenish", "future"):
        for days in windows:
            sum_col, active_col = f"{signal}_sum_{days}", f"{signal}_days_{days}"
            if sum_col in base and active_col in base:
                derived[f"{signal}_active_mean_{days}"] = base[sum_col] / base[active_col].clip(lower=1.0)
        for short, long in zip(windows[:-1], windows[1:]):
            derived[f"{signal}_pace_{short}_vs_{long}"] = (
                base[f"{signal}_sum_{short}"] / short
            ) / (base[f"{signal}_sum_{long}"] / long).clip(lower=0.1)
            derived[f"{signal}_sum_band_{short + 1}_{long}"] = (
                base[f"{signal}_sum_{long}"] - base[f"{signal}_sum_{short}"]
            ).clip(lower=0.0)
            derived[f"{signal}_days_band_{short + 1}_{long}"] = (
                base[f"{signal}_days_{long}"] - base[f"{signal}_days_{short}"]
            ).clip(lower=0.0)
    month_angle = 2.0 * np.pi * (anchor.month - 1) / 12.0
    derived["forecast_month_sin"] = np.sin(month_angle)
    derived["forecast_month_cos"] = np.cos(month_angle)
    derived["forecast_quarter"] = float(anchor.quarter)
    base = pd.concat([base, pd.DataFrame(derived, index=base.index)], axis=1)
    if not include_operational:
        return base
    for days in windows:
        start = anchor - pd.Timedelta(days=days)
        window = history.loc[history["date"].ge(start)]
        aggregations: dict[str, tuple[str, str]] = {f"order_rows_{days}": ("sku_id", "size")}
        if "buyer_id" in window:
            aggregations[f"active_buyers_{days}"] = ("buyer_id", "nunique")
        for source in ("qty_shipped", "qty_inbound", "qty_debt"):
            if source in window:
                aggregations[f"{source.removeprefix('qty_')}_sum_{days}"] = (source, "sum")
                aggregations[f"{source.removeprefix('qty_')}_max_{days}"] = (source, "max")
        operational = window.groupby("sku_id", as_index=False).agg(**aggregations)
        base = base.merge(operational, on="sku_id", how="left")
    return base


def build_anchor_samples(
    gold: pd.DataFrame, qfo: pd.DataFrame, anchor: pd.Timestamp, config: dict[str, Any]
) -> pd.DataFrame:
    """Build one row per SKU using only facts available before ``anchor``.

    The target is the next ``forecast_days`` of positive replenishment. The
    optional purge is handled by the backtest trainer when it selects fitting
    anchors; it does not change the feature or label window of this sample.
    """

    required = {"date", "sku_id", "qty_replenish", "qty_future", *STATIC_COLUMNS}
    missing = required.difference(gold.columns)
    if missing:
        raise ValueError(f"Gold table missing columns: {sorted(missing)}")
    anchor = pd.Timestamp(anchor).normalize()
    lookback_days = int(config["lookback_days"])
    forecast_days = int(config["forecast_days"])
    frame = gold.copy()
    frame["date"] = _as_dates(frame, "date")
    frame["qty_replenish"] = pd.to_numeric(frame["qty_replenish"], errors="coerce").fillna(0.0).clip(lower=0.0)
    frame["qty_future"] = pd.to_numeric(frame["qty_future"], errors="coerce").fillna(0.0).clip(lower=0.0)
    history = frame.loc[frame["date"].lt(anchor)].copy()
    if history.empty:
        return pd.DataFrame()
    history["listing_date"] = _as_dates(history, "listing_date")
    history = history.loc[history["listing_date"].le(anchor) | history["listing_date"].isna()].copy()
    universe = history.sort_values("date").groupby("sku_id", as_index=False).tail(1)
    base = universe.loc[:, ["sku_id", *STATIC_COLUMNS]].copy()
    feature_windows = tuple(sorted(set(int(value) for value in config.get(
        "feature_windows", (7, 30, lookback_days)
    ))))
    for days in feature_windows:
        base = base.merge(_window_features(history, anchor, days), on="sku_id", how="left")
        if bool(config.get("hierarchical_features", False)) or bool(config.get("sibling_features", False)):
            base = base.merge(_style_window_features(history, anchor, days), on="style_id", how="left")
        if bool(config.get("sibling_features", False)):
            base = _add_sibling_features(base, days)
    if bool(config.get("engineered_features", False)):
        base = _add_engineered_features(
            base,
            history,
            anchor,
            include_operational=bool(config.get("operational_features", False)),
            windows=feature_windows,
        )
    target_end = anchor + pd.Timedelta(days=forecast_days)
    target = (
        frame.loc[frame["date"].ge(anchor) & frame["date"].lt(target_end)]
        .groupby("sku_id", as_index=False)["qty_replenish"]
        .sum()
        .rename(columns={"qty_replenish": "actual_qty"})
    )
    base = base.merge(target, on="sku_id", how="left")
    qfo_frame = qfo.loc[:, ["sku_id", "qty_first_order", "qfo_available_date"]].copy()
    qfo_frame["qfo_available_date"] = _as_dates(qfo_frame, "qfo_available_date")
    qfo_frame["qfo_available"] = qfo_frame["qfo_available_date"].lt(anchor).astype(float)
    qfo_frame["qfo_qty"] = pd.to_numeric(qfo_frame["qty_first_order"], errors="coerce").fillna(0.0)
    # Quantity is not known before its source availability date.  Keeping the
    # flag alone gated while exposing the quantity would leak future metadata.
    qfo_frame["qfo_qty"] *= qfo_frame["qfo_available"]
    base = base.merge(qfo_frame.loc[:, ["sku_id", "qfo_available", "qfo_qty"]], on="sku_id", how="left")
    base["listing_date"] = _as_dates(base, "listing_date")
    base["listing_age_days"] = (anchor - base["listing_date"]).dt.days.clip(lower=0).fillna(0.0)
    base["price_tag"] = pd.to_numeric(base["price_tag"], errors="coerce").fillna(0.0)
    numeric = [
        column for column in base.columns
        if any(column.endswith(f"_{days}") for days in feature_windows)
    ]
    numeric += ["actual_qty", "qfo_available", "qfo_qty", "listing_age_days", "price_tag"]
    base[numeric] = base[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    base = pd.concat([
        base,
        pd.DataFrame({
            "feature_max_date": history["date"].max(),
            "target_start": anchor,
            "target_end": target_end,
        }, index=base.index),
    ], axis=1)
    return base.sort_values("sku_id").reset_index(drop=True)


def model_feature_columns(samples: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Return configured model features while excluding identities and dates."""

    excluded = set(config.get("exclude_features", ())) | {
        "actual_qty", "feature_max_date", "target_start", "target_end", "listing_date", "product_name"
    }
    categorical = set(config.get("categorical_features", ())) if config.get("include_categorical_features", False) else set()
    return [
        column
        for column in samples.columns
        if column not in excluded and (pd.api.types.is_numeric_dtype(samples[column]) or column in categorical)
    ]
