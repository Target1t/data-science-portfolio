from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase9.metrics import phase7_scorecard
from phase9.nested import apply_calibration, choose_calibration, pooled_loss


def _numbers(raw: str) -> list[float]:
    return [float(value) for value in raw.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--threshold-offsets", default="-0.01,-0.005,0,0.005,0.01")
    parser.add_argument("--scale-multipliers", default="0.98,0.99,1,1.01,1.02")
    parser.add_argument("--ratio-floor", type=float)
    args = parser.parse_args()

    source = args.source_run.resolve()
    output = args.output_run.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    context_output = output / "anchor_contexts"
    context_output.mkdir()

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    thresholds = {key: float(value) for key, value in config["quadrant_thresholds"].items()}
    offsets = _numbers(args.threshold_offsets)
    multipliers = _numbers(args.scale_multipliers)
    paths = sorted((source / "anchor_contexts").glob("*.csv"))
    if len(paths) <= args.min_history:
        raise ValueError("Not enough origins for prequential calibration")

    raw_contexts = [pd.read_csv(path, dtype={"sku_id": str, "style_id": str}) for path in paths]
    selections: list[dict[str, object]] = []
    scores: list[dict[str, object]] = []
    outer_contexts: list[pd.DataFrame] = []
    for index in range(args.min_history, len(paths)):
        choice = choose_calibration(
            raw_contexts[:index], thresholds, offsets, multipliers, args.ratio_floor
        )
        calibrated = apply_calibration(
            raw_contexts[index], thresholds, choice.threshold_offset, choice.scale_multiplier
        )
        calibrated.to_csv(context_output / paths[index].name, index=False, encoding="utf-8-sig")
        score = phase7_scorecard(calibrated)
        anchor = pd.Timestamp(paths[index].stem).strftime("%Y-%m-%d")
        score["anchor_date"] = anchor
        scores.append(score)
        outer_contexts.append(calibrated)
        selections.append(
            {
                "anchor_date": anchor,
                "history_origins": index,
                "threshold_offset": choice.threshold_offset,
                "scale_multiplier": choice.scale_multiplier,
                "inner_wmape": choice.inner_wmape,
                "inner_ratio": choice.inner_ratio,
            }
        )

    scorecard = pd.DataFrame(scores)
    scorecard.to_csv(output / "scorecard.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selections).to_csv(output / "selections.csv", index=False, encoding="utf-8-sig")
    wmape, ratio = pooled_loss(outer_contexts)
    pooled = phase7_scorecard(pd.concat(outer_contexts, ignore_index=True))
    summary = {
        "source_run": str(source),
        "outer_origins": len(outer_contexts),
        "warmup_origins": args.min_history,
        "threshold_offsets": offsets,
        "scale_multipliers": multipliers,
        "ratio_floor": args.ratio_floor,
        "pooled_wmape": wmape,
        "pooled_ratio": ratio,
        "pooled_scorecard": pooled,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("outer_origins", "pooled_wmape", "pooled_ratio")}, indent=2))


if __name__ == "__main__":
    main()
