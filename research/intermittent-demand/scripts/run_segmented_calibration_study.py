from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase9.metrics import phase7_scorecard
from phase9.nested import pooled_loss
from phase9.segmented_calibration import apply_segmented_calibration, choose_segmented_calibration


def _load(run: Path) -> tuple[dict[str, object], list[Path], list[pd.DataFrame]]:
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    paths = sorted((run / "anchor_contexts").glob("*.csv"))
    contexts = [pd.read_csv(path, dtype={"sku_id": str, "style_id": str}) for path in paths]
    return config, paths, contexts


def _evaluate(
    run: Path,
    specification: str,
    history_window: int | None,
    min_history: int,
    offsets: list[float],
    multipliers: list[float],
    ratio_floor: float | None,
) -> tuple[dict[str, object], list[pd.DataFrame], list[dict[str, object]]]:
    config, paths, contexts = _load(run)
    thresholds = {key: float(value) for key, value in config["quadrant_thresholds"].items()}
    calibrated_contexts = []
    selections = []
    for index in range(min_history, len(contexts)):
        start = 0 if history_window is None else max(0, index - history_window)
        history = contexts[start:index]
        choices = choose_segmented_calibration(
            history, thresholds, specification, offsets, multipliers,
            ratio_floor=ratio_floor,
        )
        calibrated = apply_segmented_calibration(
            contexts[index], thresholds, specification, choices
        )
        calibrated_contexts.append(calibrated)
        for label, choice in choices.items():
            selections.append({
                "anchor_date": paths[index].stem,
                "specification": specification,
                "history_window": history_window or "all",
                "segment": label,
                **choice.__dict__,
            })
    raw_outer = contexts[min_history:]
    raw_wmape, raw_ratio = pooled_loss(raw_outer)
    wmape, ratio = pooled_loss(calibrated_contexts)
    raw_scores = [phase7_scorecard(frame) for frame in raw_outer]
    scores = [phase7_scorecard(frame) for frame in calibrated_contexts]
    result = {
        "source_run": run.name,
        "specification": specification,
        "history_window": history_window or "all",
        "ratio_floor": ratio_floor if ratio_floor is not None else "none",
        "outer_origins": len(calibrated_contexts),
        "pooled_wmape": wmape,
        "pooled_ratio": ratio,
        "mean_anchor_wmape": sum(item["global_wmape"] for item in scores) / len(scores),
        "raw_pooled_wmape": raw_wmape,
        "raw_pooled_ratio": raw_ratio,
        "raw_mean_anchor_wmape": sum(item["global_wmape"] for item in raw_scores) / len(raw_scores),
        "delta_pooled_wmape": wmape - raw_wmape,
        "delta_mean_anchor_wmape": (
            sum(item["global_wmape"] for item in scores) -
            sum(item["global_wmape"] for item in raw_scores)
        ) / len(scores),
        "pooled_f1": phase7_scorecard(pd.concat(calibrated_contexts, ignore_index=True))["f1"],
    }
    return result, calibrated_contexts, selections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monthly-run", type=Path, required=True)
    parser.add_argument("--shifted-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-history", type=int, default=3)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)

    specifications = ["global", "quadrant", "quantity", "quadrant_quantity", "age_quantity"]
    windows = [None, 3, 5]
    ratio_floors = [None, 0.82, 0.85, 0.90]
    offsets = [-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04]
    multipliers = [0.80, 0.90, 1.00, 1.10, 1.20, 1.30]
    results = []
    cached = {}
    selection_rows = []
    for grid, run in (("monthly", args.monthly_run.resolve()), ("shifted", args.shifted_run.resolve())):
        for specification in specifications:
            for window in windows:
                for ratio_floor in ratio_floors:
                    result, contexts, selections = _evaluate(
                        run, specification, window, args.min_history, offsets, multipliers,
                        ratio_floor,
                    )
                    result["grid"] = grid
                    results.append(result)
                    floor_key = ratio_floor if ratio_floor is not None else "none"
                    cached[(grid, specification, window or "all", floor_key)] = contexts
                    selection_rows.extend({"grid": grid, "ratio_floor": floor_key, **row} for row in selections)

    table = pd.DataFrame(results)
    table.to_csv(output / "study_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selection_rows).to_csv(output / "all_selections.csv", index=False, encoding="utf-8-sig")

    paired = table.pivot_table(
        index=["specification", "history_window", "ratio_floor"],
        columns="grid",
        values=["delta_pooled_wmape", "pooled_ratio"],
    ).reset_index()
    paired.columns = [
        "specification", "history_window", "ratio_floor",
        "monthly_delta", "shifted_delta", "monthly_ratio", "shifted_ratio",
    ]
    paired["worst_grid_delta"] = paired[["monthly_delta", "shifted_delta"]].max(axis=1)
    paired["average_delta"] = paired[["monthly_delta", "shifted_delta"]].mean(axis=1)
    eligible = paired.loc[
        paired["monthly_ratio"].ge(0.82) & paired["shifted_ratio"].ge(0.82)
    ]
    winner = eligible.sort_values(["worst_grid_delta", "average_delta"]).iloc[0]
    best_specification = str(winner["specification"])
    best_window = winner["history_window"]
    best_window_key = "all" if str(best_window) == "all" else int(best_window)
    best_floor = winner["ratio_floor"]
    best_dir = output / "best_contexts"
    for grid in ("monthly", "shifted"):
        destination = best_dir / grid
        destination.mkdir(parents=True)
        for frame, source_path in zip(
            cached[(grid, best_specification, best_window_key, best_floor)],
            sorted((args.monthly_run if grid == "monthly" else args.shifted_run).resolve().joinpath("anchor_contexts").glob("*.csv"))[args.min_history:],
        ):
            frame.to_csv(destination / source_path.name, index=False, encoding="utf-8-sig")
    summary = {
        "winner": {
            "specification": best_specification,
            "history_window": str(best_window),
            "ratio_floor": str(best_floor),
            "monthly_delta_pooled_wmape": float(winner["monthly_delta"]),
            "shifted_delta_pooled_wmape": float(winner["shifted_delta"]),
            "monthly_ratio": float(winner["monthly_ratio"]),
            "shifted_ratio": float(winner["shifted_ratio"]),
        },
        "candidate_count": len(paired),
        "selection_rule": "minimize worst-grid delta, then average delta",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
