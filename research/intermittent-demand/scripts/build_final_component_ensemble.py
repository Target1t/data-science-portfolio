from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase9.data_contract import create_run_dir
from phase9.skc import build_skc_forecast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--base-final", type=Path, required=True)
    parser.add_argument("--multiscale-final", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--quantity-weight", type=float, default=0.25)
    args = parser.parse_args()
    root = args.project_root.resolve()
    base_dir = args.base_final.resolve()
    multiscale_dir = args.multiscale_final.resolve()
    output = create_run_dir(root, args.run_id)

    base = pd.read_csv(base_dir / "sku_forecast_h30.csv", dtype={"sku_id": str, "style_id": str})
    multiscale = pd.read_csv(
        multiscale_dir / "sku_forecast_h30.csv", dtype={"sku_id": str, "style_id": str}
    )
    if base["sku_id"].tolist() != multiscale["sku_id"].tolist():
        raise ValueError("Final SKU forecasts are not aligned")
    if not base[["forecast_start", "forecast_end"]].equals(
        multiscale[["forecast_start", "forecast_end"]]
    ):
        raise ValueError("Forecast windows do not match")

    result = base.copy()
    result["base_conditional_qty"] = base["conditional_qty"]
    result["multiscale_conditional_qty"] = multiscale["conditional_qty"]
    result["conditional_qty"] = (
        (1.0 - args.quantity_weight) * base["conditional_qty"]
        + args.quantity_weight * multiscale["conditional_qty"]
    ).clip(lower=0)
    # The occurrence decision is deliberately inherited from v4.
    result["forecast_qty_h30"] = result["conditional_qty"].where(
        base["forecast_qty_h30"].gt(0), 0.0
    )
    result.to_csv(output / "sku_forecast_h30.csv", index=False, encoding="utf-8-sig")

    shutil.copy2(base_dir / "model.pkl", output / "base_model.pkl")
    shutil.copy2(multiscale_dir / "model.pkl", output / "multiscale_model.pkl")
    shutil.copy2(base_dir / "data_manifest.json", output / "data_manifest.json")
    shutil.copy2(base_dir / "config.json", output / "base_config.json")
    shutil.copy2(multiscale_dir / "config.json", output / "multiscale_config.json")
    base_training = json.loads((base_dir / "training_manifest.json").read_text(encoding="utf-8"))
    multiscale_training = json.loads(
        (multiscale_dir / "training_manifest.json").read_text(encoding="utf-8")
    )
    training = {
        **base_training,
        "ensemble": True,
        "base_feature_count": len(base_training["feature_cols"]),
        "multiscale_feature_count": len(multiscale_training["feature_cols"]),
        "feature_cols": base_training["feature_cols"],
    }
    (output / "training_manifest.json").write_text(
        json.dumps(training, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ensemble = {
        "architecture": "component_quantity_ensemble",
        "occurrence_source": "base_v4",
        "base_final": str(base_dir),
        "multiscale_final": str(multiscale_dir),
        "base_quantity_weight": 1.0 - args.quantity_weight,
        "multiscale_quantity_weight": args.quantity_weight,
    }
    (output / "ensemble_manifest.json").write_text(
        json.dumps(ensemble, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "config.json").write_text(
        json.dumps(ensemble, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    client, audit = build_skc_forecast(result, {0: 1.1, 1: 0.9, 2: 1.1})
    client.to_csv(output / "skc_forecast_h30.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output / "skc_forecast_h30_audit.csv", index=False, encoding="utf-8-sig")
    (output / "skc_manifest.json").write_text(
        json.dumps({
            "route": "sku_derived_skc_band_calibrated",
            "factors": {"0": 1.1, "1": 0.9, "2": 1.1},
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
