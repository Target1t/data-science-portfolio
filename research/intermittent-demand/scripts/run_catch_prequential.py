from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from phase9.catch import DirectLossGate, PreferenceGate, blend_probability, blend_quantity
from phase9.data_contract import create_run_dir
from phase9.metrics import phase7_scorecard


def paired_context(self_path: Path, peer_path: Path) -> pd.DataFrame:
    self_frame = pd.read_csv(self_path, dtype={"sku_id": str})
    peer_frame = pd.read_csv(peer_path, dtype={"sku_id": str})
    keys = [column for column in ("sku_id", "style_id") if column in self_frame and column in peer_frame]
    peer_identifiers = [column for column in ("style_id",) if column in peer_frame and column not in keys]
    reliability = [
        column for column in peer_frame.columns
        if (column.startswith("sibling_") or "_share_skc_" in column)
        and column not in self_frame.columns
    ]
    keep = keys + peer_identifiers + ["prob", "conditional_qty"] + reliability
    peer = peer_frame.loc[:, keep].rename(columns={
        "prob": "peer_probability", "conditional_qty": "peer_quantity",
    })
    result = self_frame.merge(peer, on=keys, how="inner", validate="one_to_one")
    result = result.rename(columns={"prob": "self_probability", "conditional_qty": "self_quantity"})
    return result


def thresholds(frame: pd.DataFrame, config: dict[str, object]) -> np.ndarray:
    values = dict(config.get("quadrant_thresholds", {}))
    default = float(config["gate_threshold"])
    replenish = frame.replenish_days_90.to_numpy(float) > 0
    future = frame.future_days_90.to_numpy(float) > 0
    return np.select(
        [replenish & future, replenish, future],
        [
            float(values.get("repl1_fut1", default)),
            float(values.get("repl1_fut0", default)),
            float(values.get("repl0_fut1", default)),
        ],
        default=float(values.get("repl0_fut0", default)),
    )


def scored_context(frame: pd.DataFrame, probability: np.ndarray, quantity: np.ndarray, threshold: np.ndarray) -> pd.DataFrame:
    context = frame.loc[:, [column for column in (
        "sku_id", "style_id", "category", "actual_qty", "replenish_days_90",
        "future_days_90", "listing_age_days",
    ) if column in frame]].copy()
    context["prob"] = probability
    context["conditional_qty"] = quantity
    context["prediction_qty"] = np.where(probability >= threshold, np.maximum(quantity, 0.0), 0.0)
    return context


def constant_blend_loss(
    frames: list[pd.DataFrame], config: dict[str, object], occurrence_weight: float, magnitude_weight: float
) -> tuple[float, float]:
    absolute_error = 0.0
    actual_total = 0.0
    prediction_total = 0.0
    for frame in frames:
        probability = blend_probability(
            frame.self_probability, frame.peer_probability, np.full(len(frame), occurrence_weight)
        )
        quantity = blend_quantity(
            frame.self_quantity, frame.peer_quantity, np.full(len(frame), magnitude_weight)
        )
        prediction = np.where(probability >= thresholds(frame, config), np.maximum(quantity, 0.0), 0.0)
        actual = frame.actual_qty.to_numpy(float)
        absolute_error += float(np.abs(prediction - actual).sum())
        actual_total += float(actual.sum())
        prediction_total += float(prediction.sum())
    return absolute_error / actual_total, prediction_total / actual_total


def select_constant_weights(
    frames: list[pd.DataFrame], config: dict[str, object], grid: list[float]
) -> tuple[float, float, float, float]:
    candidates: list[tuple[float, float, float, float]] = []
    for occurrence_weight in grid:
        for magnitude_weight in grid:
            loss, ratio = constant_blend_loss(frames, config, occurrence_weight, magnitude_weight)
            candidates.append((loss, abs(1.0 - ratio), occurrence_weight, magnitude_weight))
    loss, _, occurrence_weight, magnitude_weight = min(candidates)
    _, ratio = constant_blend_loss(frames, config, occurrence_weight, magnitude_weight)
    return occurrence_weight, magnitude_weight, loss, ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-run", required=True)
    parser.add_argument("--peer-run", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--gate-feature-set", choices=("all", "prediction", "reliability"), default="all")
    parser.add_argument("--gate-objective", choices=("preference", "direct"), default="preference")
    parser.add_argument("--occurrence-gate-objective", choices=("preference", "direct"))
    parser.add_argument("--magnitude-gate-objective", choices=("preference", "direct"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    runs = root / "artifacts" / "runs"
    self_dir = runs / args.self_run
    peer_dir = runs / args.peer_run
    config = json.loads((self_dir / "config.json").read_text(encoding="utf-8"))
    self_paths = sorted((self_dir / "anchor_contexts").glob("*.csv"))
    peer_paths = {path.name: path for path in (peer_dir / "anchor_contexts").glob("*.csv")}
    pairs = [(path, peer_paths[path.name]) for path in self_paths if path.name in peer_paths]
    if len(pairs) <= args.warmup:
        raise ValueError("Insufficient aligned out-of-sample origins")

    frames = [paired_context(self_path, peer_path) for self_path, peer_path in pairs]
    output = create_run_dir(root, args.run_id)
    context_dir = output / "anchor_contexts"
    context_dir.mkdir()
    constant_context_dir = output / "constant_anchor_contexts"
    constant_context_dir.mkdir()
    fixed_context_dir = output / "fixed_cs_hcl_anchor_contexts"
    fixed_context_dir.mkdir()
    records: list[dict[str, object]] = []
    constant_records: list[dict[str, object]] = []
    gate_records: list[dict[str, object]] = []
    coefficient_records: list[pd.DataFrame] = []
    fixed_records: list[dict[str, object]] = []

    for index in range(args.warmup, len(frames)):
        history = pd.concat(frames[:index], ignore_index=True)
        current = frames[index]
        occurrence_objective = args.occurrence_gate_objective or args.gate_objective
        magnitude_objective = args.magnitude_gate_objective or args.gate_objective
        occurrence_class = DirectLossGate if occurrence_objective == "direct" else PreferenceGate
        magnitude_class = DirectLossGate if magnitude_objective == "direct" else PreferenceGate
        occurrence_gate = occurrence_class.fit(history, "occurrence", args.gate_feature_set)
        magnitude_gate = magnitude_class.fit(history, "magnitude", args.gate_feature_set)
        occurrence_weight = occurrence_gate.predict(current)
        magnitude_weight = magnitude_gate.predict(current)
        probability = blend_probability(current.self_probability, current.peer_probability, occurrence_weight)
        quantity = blend_quantity(current.self_quantity, current.peer_quantity, magnitude_weight)
        context = scored_context(current, probability, quantity, thresholds(current, config))
        anchor = pairs[index][0].stem
        context.to_csv(context_dir / f"{anchor}.csv", index=False, encoding="utf-8-sig")
        score = phase7_scorecard(context)
        score.update({"anchor_date": anchor, "model": "CATCH"})
        records.append(score)

        fixed_context = scored_context(
            current,
            current.peer_probability.to_numpy(float),
            current.self_quantity.to_numpy(float),
            thresholds(current, config),
        )
        fixed_context.to_csv(fixed_context_dir / f"{anchor}.csv", index=False, encoding="utf-8-sig")
        fixed_score = phase7_scorecard(fixed_context)
        fixed_score.update({"anchor_date": anchor, "model": "fixed_CS_HCL_clean"})
        fixed_records.append(fixed_score)

        constant_occurrence, constant_magnitude, inner_loss, inner_ratio = select_constant_weights(
            frames[:index], config, [0.0, 0.25, 0.5, 0.75, 1.0]
        )
        constant_probability = blend_probability(
            current.self_probability, current.peer_probability,
            np.full(len(current), constant_occurrence),
        )
        constant_quantity = blend_quantity(
            current.self_quantity, current.peer_quantity,
            np.full(len(current), constant_magnitude),
        )
        constant_context = scored_context(
            current, constant_probability, constant_quantity, thresholds(current, config)
        )
        constant_context.to_csv(
            constant_context_dir / f"{anchor}.csv", index=False, encoding="utf-8-sig"
        )
        constant_score = phase7_scorecard(constant_context)
        constant_score.update({
            "anchor_date": anchor,
            "model": "nested_constant_blend",
            "occurrence_weight": constant_occurrence,
            "magnitude_weight": constant_magnitude,
            "inner_wmape": inner_loss,
            "inner_ratio": inner_ratio,
        })
        constant_records.append(constant_score)
        gate_records.append({
            "anchor_date": anchor,
            "training_origins": index,
            "occurrence_weight_mean": float(np.mean(occurrence_weight)),
            "occurrence_weight_p10": float(np.quantile(occurrence_weight, .1)),
            "occurrence_weight_p90": float(np.quantile(occurrence_weight, .9)),
            "magnitude_weight_mean": float(np.mean(magnitude_weight)),
            "magnitude_weight_p10": float(np.quantile(magnitude_weight, .1)),
            "magnitude_weight_p90": float(np.quantile(magnitude_weight, .9)),
        })
        for component, fitted_gate in (
            ("occurrence", occurrence_gate), ("magnitude", magnitude_gate)
        ):
            coefficients = fitted_gate.coefficients()
            coefficients.insert(0, "component", component)
            coefficients.insert(0, "anchor_date", anchor)
            coefficient_records.append(coefficients)

    scorecard = pd.DataFrame(records)
    scorecard.to_csv(output / "scorecard.csv", index=False, encoding="utf-8-sig")
    constant_scorecard = pd.DataFrame(constant_records)
    constant_scorecard.to_csv(output / "constant_scorecard.csv", index=False, encoding="utf-8-sig")
    fixed_scorecard = pd.DataFrame(fixed_records)
    fixed_scorecard.to_csv(output / "fixed_cs_hcl_scorecard.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(gate_records).to_csv(output / "gate_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.concat(coefficient_records, ignore_index=True).to_csv(
        output / "gate_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "self_run": args.self_run,
        "peer_run": args.peer_run,
        "gate_feature_set": args.gate_feature_set,
        "gate_objective": args.gate_objective,
        "occurrence_gate_objective": args.occurrence_gate_objective or args.gate_objective,
        "magnitude_gate_objective": args.magnitude_gate_objective or args.gate_objective,
        "outer_origins": len(scorecard),
        "mean_wmape": float(scorecard.global_wmape.mean()),
        "mean_ratio": float(scorecard.global_ratio.mean()),
        "mean_auc": float(scorecard.auc.mean()),
        "mean_f1": float(scorecard.f1.mean()),
        "fixed_cs_hcl_mean_wmape": float(fixed_scorecard.global_wmape.mean()),
        "fixed_cs_hcl_mean_ratio": float(fixed_scorecard.global_ratio.mean()),
        "constant_mean_wmape": float(constant_scorecard.global_wmape.mean()),
        "constant_mean_ratio": float(constant_scorecard.global_ratio.mean()),
        "constant_mean_auc": float(constant_scorecard.auc.mean()),
        "constant_mean_f1": float(constant_scorecard.f1.mean()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
