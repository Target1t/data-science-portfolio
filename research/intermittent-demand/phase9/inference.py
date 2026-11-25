"""Final Phase9 SKU training and inference."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .data_contract import Phase9Paths, create_run_dir, validate_snapshot
from .features import build_anchor_samples, model_feature_columns
from .backtest import training_anchor_dates
from .model import HurdleModel


def forecast_window(observed_through: str, forecast_days: int) -> tuple[str, str]:
    start = pd.Timestamp(observed_through) + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=forecast_days - 1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def train_final(project_root: Path, config_path: Path, run_id: str) -> Path:
    root = project_root.resolve(); config = json.loads(config_path.read_text(encoding="utf-8")); paths = Phase9Paths.from_root(root)
    manifest = validate_snapshot(paths); run_dir = create_run_dir(root, run_id); shutil.copy2(config_path, run_dir / "config.json")
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    gold = pd.read_csv(paths.gold, dtype={"sku_id": str, "style_id": str}); qfo = pd.read_csv(paths.qfo, dtype={"sku_id": str})
    observed = pd.to_datetime(gold.date).max().normalize(); train_anchor = observed - pd.Timedelta(days=int(config["forecast_days"]) - 1)
    first_observed = pd.to_datetime(gold.date).min().normalize()
    anchors = training_anchor_dates(
        str(first_observed.date()),
        str((observed + pd.Timedelta(days=int(config["purge_days"]))).date()),
        config,
    )
    if train_anchor not in anchors:
        anchors.append(train_anchor)
    anchors = sorted(set(anchor for anchor in anchors if anchor + pd.Timedelta(days=int(config["forecast_days"]) - 1) <= observed))
    samples = pd.concat([build_anchor_samples(gold, qfo, anchor, config) for anchor in anchors], ignore_index=True)
    features = model_feature_columns(samples, config)
    model = HurdleModel.fit(samples, features, "actual_qty", config); model.save(run_dir / "model.pkl")
    (run_dir / "training_manifest.json").write_text(json.dumps({"observed_through": observed.date().isoformat(), "training_anchor": train_anchor.date().isoformat(), "training_anchors": [a.date().isoformat() for a in anchors], "training_rows": len(samples), "feature_cols": features}, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def predict_final(run_dir: Path) -> Path:
    run_dir = run_dir.resolve(); root = run_dir.parents[2]; config = json.loads((run_dir / "config.json").read_text(encoding="utf-8")); paths = Phase9Paths.from_root(root)
    gold = pd.read_csv(paths.gold, dtype={"sku_id": str, "style_id": str}); qfo = pd.read_csv(paths.qfo, dtype={"sku_id": str})
    observed = pd.to_datetime(gold.date).max().normalize(); samples = build_anchor_samples(gold, qfo, observed + pd.Timedelta(days=1), config)
    # Inference needs no future label; build_anchor_samples produces an all-zero unavailable target after the observed boundary.
    meta = json.loads((run_dir / "training_manifest.json").read_text(encoding="utf-8")); model = HurdleModel.load(run_dir / "model.pkl")
    prediction = model.predict(samples.loc[:, meta["feature_cols"]]); start, end = forecast_window(str(observed.date()), int(config["forecast_days"]))
    output = samples.loc[:, ["sku_id", "style_id", "product_name", "category", "sub_category", "season", "band", "series", "price_tag", "size_id", "color_id", "listing_date"]].join(prediction)
    output = output.rename(columns={"prediction_qty": "forecast_qty_h30", "prob": "ai_pred_prob"}); output["forecast_start"] = start; output["forecast_end"] = end
    path = run_dir / "sku_forecast_h30.csv"; output.to_csv(path, index=False, encoding="utf-8-sig"); return path
