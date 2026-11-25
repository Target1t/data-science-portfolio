from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase9.metrics import phase7_scorecard


def _thresholds(frame: pd.DataFrame, values: dict[str, float]) -> np.ndarray:
    replenish = pd.to_numeric(frame["replenish_days_90"], errors="coerce").fillna(0).gt(0)
    future = pd.to_numeric(frame["future_days_90"], errors="coerce").fillna(0).gt(0)
    return np.select(
        [replenish & future, replenish, future],
        [values["repl1_fut1"], values["repl1_fut0"], values["repl0_fut1"]],
        default=values["repl0_fut0"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--multiscale-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--occurrence-weight", type=float, default=0.0)
    parser.add_argument("--quantity-weight", type=float, default=0.25)
    args = parser.parse_args()
    output = args.output_run.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    context_dir = output / "anchor_contexts"
    context_dir.mkdir()
    reference_dir = output / "base_reference_contexts"
    reference_dir.mkdir()

    base_config = json.loads((args.base_run / "config.json").read_text(encoding="utf-8"))
    threshold_values = {
        key: float(value) for key, value in base_config["quadrant_thresholds"].items()
    }
    scores = []
    for multiscale_path in sorted((args.multiscale_run / "anchor_contexts").glob("*.csv")):
        base_path = args.base_run / "anchor_contexts" / multiscale_path.name
        base = pd.read_csv(base_path, dtype={"sku_id": str, "style_id": str})
        multiscale = pd.read_csv(multiscale_path, dtype={"sku_id": str, "style_id": str})
        if base["sku_id"].tolist() != multiscale["sku_id"].tolist():
            raise ValueError(f"SKU alignment mismatch at {multiscale_path.name}")
        result = base.copy()
        if "style_id" not in result and "style_id" in multiscale:
            result.insert(1, "style_id", multiscale["style_id"])
        result["base_prob"] = base["prob"]
        result["multiscale_prob"] = multiscale["prob"]
        result["base_conditional_qty"] = base["conditional_qty"]
        result["multiscale_conditional_qty"] = multiscale["conditional_qty"]
        result["prob"] = (
            (1.0 - args.occurrence_weight) * base["prob"]
            + args.occurrence_weight * multiscale["prob"]
        )
        result["conditional_qty"] = (
            (1.0 - args.quantity_weight) * base["conditional_qty"]
            + args.quantity_weight * multiscale["conditional_qty"]
        )
        result["prediction_qty"] = np.where(
            result["prob"].to_numpy() >= _thresholds(result, threshold_values),
            result["conditional_qty"].clip(lower=0).to_numpy(),
            0.0,
        )
        reference = result.copy()
        reference["prob"] = base["prob"]
        reference["conditional_qty"] = base["conditional_qty"]
        reference["prediction_qty"] = base["prediction_qty"]
        reference.to_csv(
            reference_dir / multiscale_path.name, index=False, encoding="utf-8-sig"
        )
        result.to_csv(context_dir / multiscale_path.name, index=False, encoding="utf-8-sig")
        score = phase7_scorecard(result)
        score["anchor_date"] = pd.Timestamp(multiscale_path.stem).strftime("%Y-%m-%d")
        scores.append(score)

    scorecard = pd.DataFrame(scores)
    scorecard.to_csv(output / "scorecard.csv", index=False, encoding="utf-8-sig")
    numeric_means = scorecard.select_dtypes(include="number").mean().to_dict()
    manifest = {
        "base_run": str(args.base_run.resolve()),
        "multiscale_run": str(args.multiscale_run.resolve()),
        "occurrence_weight": args.occurrence_weight,
        "quantity_weight": args.quantity_weight,
        "anchors": len(scorecard),
        "mean_metrics": numeric_means,
    }
    (output / "ensemble_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "config.json").write_text(
        json.dumps(base_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "anchors": len(scorecard),
        "global_wmape": numeric_means["global_wmape"],
        "global_ratio": numeric_means["global_ratio"],
        "f1": numeric_means["f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
