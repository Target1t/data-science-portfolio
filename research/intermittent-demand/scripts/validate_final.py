from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final Phase9 artifacts.")
    parser.add_argument("--run-dir", type=Path, required=True)
    run = parser.parse_args().run_dir.resolve()

    required = [
        "config.json",
        "data_manifest.json",
        "training_manifest.json",
        "sku_forecast_h30.csv",
        "skc_forecast_h30.csv",
        "skc_forecast_h30_audit.csv",
        "skc_manifest.json",
    ]
    if (run / "ensemble_manifest.json").is_file():
        required.extend(["base_model.pkl", "multiscale_model.pkl", "ensemble_manifest.json"])
    else:
        required.append("model.pkl")
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing final artifact(s): {missing}")

    sku = pd.read_csv(run / "sku_forecast_h30.csv")
    skc = pd.read_csv(run / "skc_forecast_h30.csv")
    audit = pd.read_csv(run / "skc_forecast_h30_audit.csv")
    training = json.loads((run / "training_manifest.json").read_text(encoding="utf-8"))

    expected = audit["sku_sum_qty"] * audit["skc_calibration_factor"]
    identity_error = float((expected - audit["forecast_qty_h30"]).abs().max())
    if identity_error > 1e-5:
        raise ValueError(f"SKC aggregation identity failed: max error={identity_error}")
    if (sku["forecast_qty_h30"] < 0).any() or (skc["forecast_qty_h30"] < 0).any():
        raise ValueError("Negative forecasts found")

    summary = {
        "run_dir": str(run),
        "sku_rows": int(len(sku)),
        "positive_skus": int((sku["forecast_qty_h30"] > 0).sum()),
        "sku_total": float(sku["forecast_qty_h30"].sum()),
        "skc_rows": int(len(skc)),
        "skc_total": float(skc["forecast_qty_h30"].sum()),
        "skc_identity_max_abs_error": identity_error,
        "forecast_start": str(sku["forecast_start"].iloc[0]),
        "forecast_end": str(sku["forecast_end"].iloc[0]),
        "training_rows": int(training["training_rows"]),
        "training_anchors": int(len(training["training_anchors"])),
        "feature_count": int(len(training["feature_cols"])),
        "ensemble": bool(training.get("ensemble", False)),
    }
    output = json.dumps(summary, ensure_ascii=False, indent=2)
    (run / "validation_summary.json").write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
