from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase9.classical import forecast_frame
from phase9.data_contract import Phase9Paths, create_run_dir, validate_snapshot
from phase9.features import build_anchor_samples
from phase9.metrics import phase7_scorecard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/optimized_v3.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    paths = Phase9Paths.from_root(root)
    manifest = validate_snapshot(paths)
    run_dir = create_run_dir(root, args.run_id)
    experiment = {
        "source_config": str(args.config),
        "alpha": args.alpha,
        "beta": args.beta,
        "anchors": config["anchors"],
    }
    (run_dir / "config.json").write_text(json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    gold = pd.read_csv(paths.gold, dtype={"sku_id": str, "style_id": str})
    qfo = pd.read_csv(paths.qfo, dtype={"sku_id": str})
    gold["date"] = pd.to_datetime(gold.date, errors="coerce")
    context_dir = run_dir / "anchor_contexts"
    context_dir.mkdir()
    scores: list[dict[str, object]] = []
    for raw_anchor in config["anchors"]:
        anchor = pd.Timestamp(raw_anchor)
        test = build_anchor_samples(gold, qfo, anchor, config)
        history = gold.loc[
            gold.date.lt(anchor), ["sku_id", "date", "qty_replenish", "listing_date"]
        ]
        predictions = forecast_frame(
            history,
            anchor,
            horizon_days=int(config["forecast_days"]),
            alpha=args.alpha,
            beta=args.beta,
        ).set_index("sku_id")
        base = test.loc[:, [
            "sku_id", "style_id", "category", "actual_qty", "replenish_days_90", "future_days_90"
        ]].copy()
        for method in ("croston", "sba", "tsb"):
            context = base.copy()
            context["prediction_qty"] = context.sku_id.map(predictions[method]).fillna(0.0).clip(lower=0.0)
            context["prob"] = context.prediction_qty.gt(0).astype(float)
            context["conditional_qty"] = context.prediction_qty
            context.to_csv(context_dir / f"{anchor:%Y%m%d}_{method}.csv", index=False, encoding="utf-8-sig")
            score = phase7_scorecard(context)
            score.update({"anchor_date": anchor.date().isoformat(), "method": method})
            scores.append(score)
    pd.DataFrame(scores).to_csv(run_dir / "scorecard.csv", index=False, encoding="utf-8-sig")
    print(run_dir)


if __name__ == "__main__":
    main()
