from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from phase9.catch import DirectLossGate, PreferenceGate, blend_probability, blend_quantity


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


def simulate_origin(
    rng: np.random.Generator,
    origin: int,
    occurrence_peer: str,
    magnitude_peer: str,
    groups: int,
    items_per_group: int,
) -> pd.DataFrame:
    rows = groups * items_per_group
    group_id = np.repeat(np.arange(groups), items_per_group)
    breadth = np.repeat(rng.integers(1, 11, size=groups), items_per_group).astype(float)
    reliability = np.sqrt(np.maximum(breadth, 1.0) / 5.5)
    group_occurrence = np.repeat(rng.normal(0, .8, size=groups), items_per_group)
    item_occurrence = rng.normal(0, .65, size=rows)
    seasonal = .25 * np.sin(2 * np.pi * origin / 6)
    occurrence_logit = -1.65 + group_occurrence + item_occurrence + seasonal
    true_probability = sigmoid(occurrence_logit)
    occurred = rng.binomial(1, true_probability)

    group_magnitude = np.repeat(rng.normal(0, .4, size=groups), items_per_group)
    item_magnitude = rng.normal(0, .45, size=rows)
    log_quantity = 1.35 + group_magnitude + item_magnitude
    positive_quantity = np.maximum(1.0, np.exp(log_quantity + rng.normal(0, .35, size=rows)))
    actual = occurred * positive_quantity

    self_probability = sigmoid(occurrence_logit + rng.normal(0, .75, size=rows))
    occurrence_sigma = .30 / reliability if occurrence_peer == "good" else 1.15 * reliability
    peer_probability = sigmoid(occurrence_logit + rng.normal(0, occurrence_sigma, size=rows))
    self_quantity = np.exp(log_quantity + rng.normal(0, .55, size=rows))
    magnitude_sigma = .22 / reliability if magnitude_peer == "good" else .95 * reliability
    peer_quantity = np.exp(log_quantity + rng.normal(0, magnitude_sigma, size=rows))

    return pd.DataFrame({
        "sku_id": [f"g{group}-i{item}" for group in range(groups) for item in range(items_per_group)],
        "style_id": [f"g{value}" for value in group_id],
        "category": np.where(group_id % 2, "B", "A"),
        "actual_qty": actual,
        "self_probability": self_probability,
        "peer_probability": peer_probability,
        "self_quantity": self_quantity,
        "peer_quantity": peer_quantity,
        "replenish_days_90": rng.poisson(3, size=rows),
        "future_days_90": rng.poisson(2, size=rows),
        "listing_age_days": rng.integers(1, 500, size=rows),
        "sibling_active_skus_90": breadth,
        "sibling_replenish_sum_90": breadth * rng.gamma(1.5, 2.0, size=rows),
        "sku_replenish_share_skc_90": 1.0 / (breadth + 1.0),
    })


def evaluate(frame: pd.DataFrame, probability: np.ndarray, quantity: np.ndarray) -> dict[str, float]:
    truth = frame.actual_qty.gt(0).astype(int).to_numpy()
    positive = truth == 1
    prediction = np.where(probability >= .30, np.maximum(quantity, 0.0), 0.0)
    total = float(frame.actual_qty.sum())
    return {
        "wmape": float(np.abs(prediction - frame.actual_qty.to_numpy()).sum() / total),
        "occurrence_logloss": float(log_loss(truth, np.clip(probability, 1e-6, 1 - 1e-6))),
        "positive_log_mae": float(np.abs(
            np.log1p(quantity[positive]) - np.log1p(frame.actual_qty.to_numpy()[positive])
        ).mean()),
    }


def run_replicate(
    seed: int, occurrence_peer: str, magnitude_peer: str, gate_objective: str
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    frames = [simulate_origin(rng, origin, occurrence_peer, magnitude_peer, 60, 8) for origin in range(9)]
    collected: dict[str, list[dict[str, float]]] = {name: [] for name in ("self", "peer", "fixed_cs_hcl", "half", "CATCH")}
    weights: list[tuple[float, float]] = []
    for index in range(3, len(frames)):
        history = pd.concat(frames[:index], ignore_index=True)
        current = frames[index]
        gate_class = DirectLossGate if gate_objective == "direct" else PreferenceGate
        occurrence_gate = gate_class.fit(history, "occurrence", "all")
        magnitude_gate = gate_class.fit(history, "magnitude", "all")
        occurrence_weight = occurrence_gate.predict(current)
        magnitude_weight = magnitude_gate.predict(current)
        weights.append((float(occurrence_weight.mean()), float(magnitude_weight.mean())))
        candidates = {
            "self": (current.self_probability.to_numpy(), current.self_quantity.to_numpy()),
            "peer": (current.peer_probability.to_numpy(), current.peer_quantity.to_numpy()),
            "fixed_cs_hcl": (current.peer_probability.to_numpy(), current.self_quantity.to_numpy()),
            "half": (
                blend_probability(current.self_probability, current.peer_probability, np.full(len(current), .5)),
                blend_quantity(current.self_quantity, current.peer_quantity, np.full(len(current), .5)),
            ),
            "CATCH": (
                blend_probability(current.self_probability, current.peer_probability, occurrence_weight),
                blend_quantity(current.self_quantity, current.peer_quantity, magnitude_weight),
            ),
        }
        for name, (probability, quantity) in candidates.items():
            collected[name].append(evaluate(current, probability, quantity))
    rows = []
    for name, metrics in collected.items():
        row: dict[str, object] = {
            "seed": seed,
            "occurrence_peer": occurrence_peer,
            "magnitude_peer": magnitude_peer,
            "model": name,
        }
        for metric in metrics[0]:
            row[metric] = float(np.mean([item[metric] for item in metrics]))
        if name == "CATCH":
            row["occurrence_weight"] = float(np.mean([item[0] for item in weights]))
            row["magnitude_weight"] = float(np.mean([item[1] for item in weights]))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--gate-objective", choices=("preference", "direct"), default="direct")
    parser.add_argument("--output", type=Path, default=Path("paper/generated/synthetic_catch_results.csv"))
    args = parser.parse_args()
    rows = []
    for occurrence_peer in ("bad", "good"):
        for magnitude_peer in ("bad", "good"):
            for replicate in range(args.replicates):
                rows.extend(run_replicate(
                    2028000 + replicate, occurrence_peer, magnitude_peer, args.gate_objective
                ))
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    summary = result.groupby(["occurrence_peer", "magnitude_peer", "model"], as_index=False).agg(
        wmape=("wmape", "mean"),
        occurrence_logloss=("occurrence_logloss", "mean"),
        positive_log_mae=("positive_log_mae", "mean"),
        occurrence_weight=("occurrence_weight", "mean"),
        magnitude_weight=("magnitude_weight", "mean"),
    )
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    catch = summary.loc[summary.model.eq("CATCH")]
    mechanism = {
        "occurrence_weight_good": float(catch.loc[catch.occurrence_peer.eq("good"), "occurrence_weight"].mean()),
        "occurrence_weight_bad": float(catch.loc[catch.occurrence_peer.eq("bad"), "occurrence_weight"].mean()),
        "magnitude_weight_good": float(catch.loc[catch.magnitude_peer.eq("good"), "magnitude_weight"].mean()),
        "magnitude_weight_bad": float(catch.loc[catch.magnitude_peer.eq("bad"), "magnitude_weight"].mean()),
    }
    print(json.dumps(mechanism, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
