"""Phase7-compatible detailed replenishment scorecard."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else np.nan


def _expected_calibration_error(truth: pd.Series, probability: pd.Series, bins: int = 10) -> float:
    """Equal-width expected calibration error for binary occurrence probabilities."""

    index = np.minimum((probability.clip(0, 1).to_numpy() * bins).astype(int), bins - 1)
    error = 0.0
    for bin_index in range(bins):
        mask = index == bin_index
        if mask.any():
            error += float(mask.mean()) * abs(
                float(truth.to_numpy()[mask].mean()) - float(probability.to_numpy()[mask].mean())
            )
    return error


def _aggregate(context: pd.DataFrame) -> pd.DataFrame:
    fields = {"actual_qty": "sum", "prediction_qty": "sum", "replenish_days_90": "max"}
    result = context.groupby("sku_id", as_index=False).agg(fields)
    result = result.loc[result["actual_qty"].gt(0)].copy()
    result["sku_ratio"] = result["prediction_qty"].div(result["actual_qty"])
    return result


def _slice(sku: pd.DataFrame, mask: pd.Series) -> dict[str, float]:
    sub = sku.loc[mask].copy()
    empty = {key: np.nan for key in ("ratio", "wmape_like", "sku_p50", "sku_trimmed_mean_10", "within_20pct_rate", "within_30pct_rate", "within_50pct_rate", "catastrophic_under_rate", "catastrophic_over_rate", "under_wape", "over_wape")}
    if sub.empty:
        return {"rows": 0, **empty}
    true = float(sub.actual_qty.sum())
    pred = float(sub.prediction_qty.sum())
    ratio = sub.sku_ratio
    lower = np.clip(sub.actual_qty - sub.prediction_qty, 0, None).sum()
    upper = np.clip(sub.prediction_qty - sub.actual_qty, 0, None).sum()
    return {
        "rows": int(len(sub)), "ratio": _safe_div(pred, true),
        "wmape_like": _safe_div(np.abs(sub.prediction_qty - sub.actual_qty).sum(), true),
        "sku_p50": float(ratio.median()), "sku_trimmed_mean_10": float(ratio.sort_values().iloc[int(len(ratio) * .1): max(int(len(ratio) * .9), 1)].mean()),
        "within_20pct_rate": float(((ratio >= .8) & (ratio <= 1.2)).mean()),
        "within_30pct_rate": float(((ratio >= .7) & (ratio <= 1.3)).mean()),
        "within_50pct_rate": float(((ratio >= .5) & (ratio <= 1.5)).mean()),
        "catastrophic_under_rate": float((ratio < .3).mean()), "catastrophic_over_rate": float((ratio > 2.).mean()),
        "under_wape": _safe_div(lower, true), "over_wape": _safe_div(upper, true),
    }


def _capture(sku: pd.DataFrame, fraction: float) -> float:
    if sku.empty or sku.actual_qty.sum() <= 0:
        return np.nan
    n = max(1, int(np.ceil(len(sku) * fraction)))
    return float(sku.nlargest(n, ["prediction_qty", "actual_qty"]).actual_qty.sum() / sku.actual_qty.sum())


def phase7_scorecard(context: pd.DataFrame) -> dict[str, object]:
    """Return the Phase7 detailed metric family from one SKU prediction context."""
    required = {"sku_id", "actual_qty", "prediction_qty", "prob", "replenish_days_90", "future_days_90", "category"}
    missing = required.difference(context.columns)
    if missing:
        raise ValueError(f"Context missing required columns: {sorted(missing)}")
    frame = context.copy()
    for column in ("actual_qty", "prediction_qty", "prob", "replenish_days_90", "future_days_90"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    actual = frame.actual_qty.clip(lower=0)
    pred = frame.prediction_qty.clip(lower=0)
    truth = actual.gt(0).astype(int)
    classified = pred.gt(0).astype(int)
    total_true, total_pred = float(actual.sum()), float(pred.sum())
    sku = _aggregate(frame)
    full = _slice(sku, pd.Series(True, index=sku.index))
    under = np.clip(actual - pred, 0, None).sum(); over = np.clip(pred - actual, 0, None).sum()
    score: dict[str, object] = {
        "rows": int(len(frame)), "auc": float(roc_auc_score(truth, frame.prob)) if truth.nunique() > 1 else np.nan,
        "average_precision": float(average_precision_score(truth, frame.prob)) if truth.nunique() > 1 else np.nan,
        "brier_score": float(brier_score_loss(truth, frame.prob.clip(0, 1))),
        "ece_10": _expected_calibration_error(truth, frame.prob),
        "f1": float(f1_score(truth, classified, zero_division=0)), "global_ratio": _safe_div(total_pred, total_true),
        "global_wmape": _safe_div(np.abs(pred - actual).sum(), total_true), "sku_p50": full["sku_p50"],
        "positive_sku_rows": int(len(sku)), "sku_trimmed_mean_10": full["sku_trimmed_mean_10"],
        "within_20pct_rate": full["within_20pct_rate"], "within_30pct_rate": full["within_30pct_rate"], "within_50pct_rate": full["within_50pct_rate"],
        "catastrophic_under_rate": full["catastrophic_under_rate"], "catastrophic_over_rate": full["catastrophic_over_rate"],
        "under_wape": _safe_div(under, total_true), "over_wape": _safe_div(over, total_true),
        "rank_corr_positive_skus": float(sku.prediction_qty.corr(sku.actual_qty, method="spearman")) if len(sku) > 1 else np.nan,
        "top10_true_volume_capture": _capture(sku, .10), "top20_true_volume_capture": _capture(sku, .20),
        "false_zero_rate": float(pred.loc[truth.eq(1)].eq(0).mean()) if truth.any() else np.nan,
        "false_positive_rate_zero_true": float(pred.loc[truth.eq(0)].gt(0).mean()) if (~truth.astype(bool)).any() else np.nan,
        "zero_true_pred_ge_3_rate": float(pred.loc[truth.eq(0)].ge(3).mean()) if (~truth.astype(bool)).any() else np.nan,
        "true_gt_10_pred_le_1_rate": float(pred.loc[actual.gt(10)].le(1).mean()) if actual.gt(10).any() else np.nan,
    }
    masks = {
        "ice": sku.replenish_days_90.eq(0), "4_25": sku.actual_qty.between(4, 25),
        "ice_4_25": sku.replenish_days_90.eq(0) & sku.actual_qty.between(4, 25),
        "1_3": sku.actual_qty.between(1, 3), "4_10": sku.actual_qty.between(4, 10),
        "11_25": sku.actual_qty.between(11, 25), "blockbuster": sku.actual_qty.gt(25),
    }
    for prefix, mask in masks.items():
        score.update({f"{prefix}_{key}": value for key, value in _slice(sku, mask).items()})
    score["blockbuster_top10_true_volume_capture"] = _capture(sku.loc[masks["blockbuster"]], .10)
    score["blockbuster_top20_true_volume_capture"] = _capture(sku.loc[masks["blockbuster"]], .20)
    quadrant = np.select([(frame.replenish_days_90 > 0) & (frame.future_days_90 > 0), (frame.replenish_days_90 > 0), (frame.future_days_90 > 0)], ["repl1_fut1", "repl1_fut0", "repl0_fut1"], default="repl0_fut0")
    frame["signal_quadrant"] = quadrant
    for name in ("repl0_fut0", "repl0_fut1", "repl1_fut0", "repl1_fut1"):
        sub = frame.loc[frame.signal_quadrant.eq(name)]
        sub_true = float(sub.actual_qty.sum())
        score.update({f"{name}_rows": int(len(sub)), f"{name}_ratio": _safe_div(float(sub.prediction_qty.sum()), sub_true), f"{name}_under_wape": _safe_div(np.clip(sub.actual_qty - sub.prediction_qty, 0, None).sum(), sub_true)})
    categories = frame.groupby(frame.category.fillna("Unknown").astype(str)).agg(actual_qty=("actual_qty", "sum"), prediction_qty=("prediction_qty", "sum"))
    categories = categories.loc[categories.actual_qty.gt(0)].copy()
    if categories.empty:
        score["category_worst5_ratio"] = "[]"; score["category_worst5_under_wape"] = "[]"
    else:
        categories["ratio"] = categories.prediction_qty / categories.actual_qty
        categories["under_wape"] = np.clip(categories.actual_qty - categories.prediction_qty, 0, None) / categories.actual_qty
        score["category_worst5_ratio"] = json.dumps(categories.assign(distance=np.abs(np.log(categories.ratio.clip(lower=1e-9)))).nlargest(5, "distance").reset_index().to_dict("records"), ensure_ascii=False)
        score["category_worst5_under_wape"] = json.dumps(categories.nlargest(5, "under_wape").reset_index().to_dict("records"), ensure_ascii=False)
    return score
