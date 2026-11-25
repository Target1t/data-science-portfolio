"""Strict rolling Phase9 backtest using the Phase7 scorecard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data_contract import Phase9Paths, create_run_dir, validate_snapshot
from .features import build_anchor_samples, model_feature_columns
from .metrics import phase7_scorecard
from .model import HurdleModel


def training_anchor_dates(first_observed: str, evaluation_anchor: str, config: dict[str, int]) -> list[pd.Timestamp]:
    start = pd.Timestamp(first_observed).normalize() + pd.Timedelta(days=int(config["lookback_days"]))
    latest = pd.Timestamp(evaluation_anchor).normalize() - pd.Timedelta(days=int(config["forecast_days"]) + int(config["purge_days"]))
    return list(pd.date_range(start=start, end=latest, freq=f"{int(config.get('training_step_days', 30))}D"))


def run_backtest(
    project_root: Path,
    config_path: Path,
    run_id: str,
    anchors: list[str] | None = None,
) -> Path:
    root = project_root.resolve(); config = json.loads(config_path.read_text(encoding="utf-8"))
    if anchors is not None:
        config["anchors"] = anchors
    paths = Phase9Paths.from_root(root); manifest = validate_snapshot(paths); run_dir = create_run_dir(root, run_id)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    gold = pd.read_csv(paths.gold, dtype={"sku_id": str, "style_id": str}); qfo = pd.read_csv(paths.qfo, dtype={"sku_id": str})
    first_observed = pd.to_datetime(gold.date).min(); scores = []
    context_dir = run_dir / "anchor_contexts"; context_dir.mkdir()
    for raw_anchor in config["anchors"]:
        anchor = pd.Timestamp(raw_anchor); train = pd.concat([build_anchor_samples(gold, qfo, date, config) for date in training_anchor_dates(str(first_observed.date()), raw_anchor, config)], ignore_index=True)
        features = model_feature_columns(train, config); model = HurdleModel.fit(train, features, "actual_qty", config)
        test = build_anchor_samples(gold, qfo, anchor, config); prediction = model.predict(test)
        context_columns = [
            "sku_id", "style_id", "category", "actual_qty",
            "replenish_days_90", "future_days_90",
        ]
        if "listing_age_days" in test:
            context_columns.append("listing_age_days")
        context_columns.extend(
            column for column in features
            if column not in context_columns
            and (column.startswith("sibling_") or "_share_skc_" in column)
        )
        context = test.loc[:, context_columns].join(prediction)
        context.to_csv(context_dir / f"{anchor:%Y%m%d}.csv", index=False, encoding="utf-8-sig")
        score = phase7_scorecard(context); score["anchor_date"] = anchor.date().isoformat(); scores.append(score)
    scorecard = pd.DataFrame(scores); scorecard.to_csv(run_dir / "scorecard.csv", index=False, encoding="utf-8-sig")
    means = scorecard.select_dtypes(include="number").mean().sort_index()
    (run_dir / "summary.md").write_text("# Phase9 回测摘要\n\n" + "\n".join(f"- `{key}`: {value:.6f}" for key, value in means.items()) + "\n", encoding="utf-8")
    return run_dir
