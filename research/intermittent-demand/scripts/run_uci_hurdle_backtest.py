from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from phase9.data_contract import create_run_dir
from phase9.metrics import phase7_scorecard
from phase9.model import HurdleModel


WINDOWS = (7, 28, 84)


def build_sample(events: pd.DataFrame, series: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    sample = series.copy()
    for days in WINDOWS:
        history = events.loc[
            events.date.ge(origin - pd.Timedelta(days=days)) & events.date.lt(origin)
        ]
        self_qty = history.groupby("series_id").quantity.sum()
        self_days = history.groupby("series_id").date.nunique()
        group_qty = history.groupby("group_id").quantity.sum()
        group_active = history.groupby("group_id").series_id.nunique()
        sample[f"self_qty_{days}"] = sample.series_id.map(self_qty).fillna(0.0)
        sample[f"self_days_{days}"] = sample.series_id.map(self_days).fillna(0.0)
        total = sample.group_id.map(group_qty).fillna(0.0)
        active = sample.group_id.map(group_active).fillna(0.0)
        sample[f"sibling_qty_{days}"] = (total - sample[f"self_qty_{days}"]).clip(lower=0.0)
        sample[f"sibling_active_skus_{days}"] = (
            active - sample[f"self_qty_{days}"].gt(0).astype(float)
        ).clip(lower=0.0)
        sample[f"sku_qty_share_skc_{days}"] = np.where(
            total.gt(0), sample[f"self_qty_{days}"] / total, 0.0
        )
    future = events.loc[
        events.date.ge(origin) & events.date.lt(origin + pd.Timedelta(days=28))
    ].groupby("series_id").quantity.sum()
    sample["actual_qty"] = sample.series_id.map(future).fillna(0.0)
    sample["listing_age_days"] = (origin - sample.first_date).dt.days.clip(lower=0)
    sample["replenish_days_90"] = sample["self_days_84"]
    sample["future_days_90"] = 0.0
    sample["target_start"] = origin
    sample["sku_id"] = sample.series_id
    sample["style_id"] = sample.group_id
    sample["category"] = sample.country
    return sample


def model_config() -> dict[str, object]:
    return {
        "gate_threshold": .25,
        "learning_rate": .05,
        "n_estimators": 250,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": .9,
        "colsample_bytree": .9,
        "random_state": 2028,
        "regression_objective": "regression_l1",
        "quantity_scale": 1.0,
        "primary_weight": 1.0,
        "probability_power": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--self-run-id", required=True)
    parser.add_argument("--peer-run-id", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    events = pd.read_csv(args.data, parse_dates=["date"], dtype={
        "series_id": str, "group_id": str, "customer_id": str, "product_id": str,
    })
    series = events.sort_values("date").groupby("series_id", as_index=False).agg(
        group_id=("group_id", "first"),
        country=("country", "first"),
        first_date=("date", "min"),
    )
    origins = list(pd.date_range("2010-04-01", "2011-11-01", freq="MS"))
    evaluation = list(pd.date_range("2011-03-01", "2011-11-01", freq="MS"))
    samples = {origin: build_sample(events, series, origin) for origin in origins}

    config = model_config()
    self_features = ["country", "listing_age_days"] + [
        f"self_{measure}_{days}" for days in WINDOWS for measure in ("qty", "days")
    ]
    peer_features = self_features + [
        feature for days in WINDOWS for feature in (
            f"sibling_qty_{days}", f"sibling_active_skus_{days}", f"sku_qty_share_skc_{days}"
        )
    ]
    runs = {
        "self": (create_run_dir(root, args.self_run_id), self_features),
        "peer": (create_run_dir(root, args.peer_run_id), peer_features),
    }
    scorecards: dict[str, list[dict[str, object]]] = {"self": [], "peer": []}
    for run_dir, _ in runs.values():
        (run_dir / "anchor_contexts").mkdir()
        (run_dir / "config.json").write_text(json.dumps({
            **config, "anchors": [str(value.date()) for value in evaluation], "forecast_days": 28,
            "dataset": "UCI Online Retail II",
        }, indent=2), encoding="utf-8")

    for origin in evaluation:
        eligible = [value for value in origins if value <= origin - pd.Timedelta(days=28)]
        training = pd.concat([samples[value] for value in eligible], ignore_index=True)
        test = samples[origin]
        for name, (run_dir, features) in runs.items():
            model = HurdleModel.fit(training, features, "actual_qty", config)
            prediction = model.predict(test)
            reliability = [
                column for column in test.columns
                if column.startswith("sibling_") or "_share_skc_" in column
            ]
            context = test.loc[:, [
                "sku_id", "style_id", "category", "actual_qty", "replenish_days_90",
                "future_days_90", "listing_age_days", *reliability,
            ]].join(prediction)
            context.to_csv(
                run_dir / "anchor_contexts" / f"{origin:%Y%m%d}.csv",
                index=False, encoding="utf-8-sig",
            )
            score = phase7_scorecard(context)
            score.update({"anchor_date": str(origin.date()), "model": name})
            scorecards[name].append(score)

    for name, (run_dir, _) in runs.items():
        card = pd.DataFrame(scorecards[name])
        card.to_csv(run_dir / "scorecard.csv", index=False, encoding="utf-8-sig")
        summary = {
            "origins": len(card),
            "mean_wmape": float(card.global_wmape.mean()),
            "mean_ratio": float(card.global_ratio.mean()),
            "mean_auc": float(card.auc.mean()),
            "mean_f1": float(card.f1.mean()),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(name, json.dumps(summary))


if __name__ == "__main__":
    main()
