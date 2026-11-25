from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "runs"
OUTPUT = ROOT / "paper" / "generated"


def scorecard(run: str) -> pd.DataFrame:
    return pd.read_csv(RUNS / run / "scorecard.csv")


def pooled(run: str) -> tuple[float, float]:
    frames = [pd.read_csv(path) for path in sorted((RUNS / run / "anchor_contexts").glob("*.csv"))]
    frame = pd.concat(frames, ignore_index=True)
    actual = frame["actual_qty"].sum()
    return float((frame.prediction_qty - frame.actual_qty).abs().sum() / actual), float(frame.prediction_qty.sum() / actual)


def contexts(run: str) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(path) for path in sorted((RUNS / run / "anchor_contexts").glob("*.csv"))],
        ignore_index=True,
    )


def occurrence_row(label: str, run: str) -> dict[str, object]:
    frame = contexts(run)
    truth = frame.actual_qty.gt(0).astype(int)
    probability = frame.prob.clip(0, 1)
    bin_index = (probability * 10).astype(int).clip(upper=9)
    ece = sum(
        float(bin_index.eq(index).mean())
        * abs(float(truth.loc[bin_index.eq(index)].mean()) - float(probability.loc[bin_index.eq(index)].mean()))
        for index in range(10)
        if bin_index.eq(index).any()
    )
    return {
        "label": label,
        "run": run,
        "rows": len(frame),
        "prevalence": float(truth.mean()),
        "roc_auc": float(roc_auc_score(truth, probability)),
        "pr_auc": float(average_precision_score(truth, probability)),
        "brier_score": float(brier_score_loss(truth, probability)),
        "ece_10": ece,
    }


def asymmetric_cost_rows(grid: str, model: str, run: str) -> list[dict[str, object]]:
    """Evaluate shortage-sensitive normalized absolute error.

    Underage weight 1 is ordinary pooled WMAPE. Larger values represent settings
    where one unit of missed demand costs more than one unit of excess forecast.
    """
    frame = contexts(run)
    denominator = float(frame.actual_qty.sum())
    underage = float((frame.actual_qty - frame.prediction_qty).clip(lower=0).sum())
    overage = float((frame.prediction_qty - frame.actual_qty).clip(lower=0).sum())
    return [
        {
            "grid": grid,
            "model": model,
            "run": run,
            "underage_weight": weight,
            "overage_weight": 1,
            "normalized_asymmetric_cost": (weight * underage + overage) / denominator,
        }
        for weight in (1, 2, 3, 5)
    ]


def summary_row(label: str, run: str) -> dict[str, object]:
    card = scorecard(run)
    pooled_wmape, pooled_ratio = pooled(run)
    return {
        "label": label,
        "run": run,
        "origins": len(card),
        "mean_anchor_wmape": float(card.global_wmape.mean()),
        "pooled_wmape": pooled_wmape,
        "mean_anchor_ratio": float(card.global_ratio.mean()),
        "pooled_ratio": pooled_ratio,
        "mean_auc": float(card.auc.mean()),
        "mean_f1": float(card.f1.mean()),
        "false_zero_rate": float(card.false_zero_rate.mean()),
    }


def write_table(frame: pd.DataFrame, stem: str) -> None:
    frame.to_csv(OUTPUT / f"{stem}.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / f"{stem}.md").write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    main_runs = [
        ("v3 monthly", "optimized-v3-extended-20260902"),
        ("CS-HCL v4 monthly", "optimized-v4-extended-20260908"),
        ("v3 shifted", "optimized-v3-shifted-20260908"),
        ("CS-HCL v4 shifted", "optimized-v4-shifted-20260909"),
        ("nested v3 monthly, ratio floor 0.82", "nested-v3-monthly-floor082-20260909"),
        ("nested v4 monthly, ratio floor 0.82", "nested-v4-monthly-floor082-20260909"),
        ("nested v3 shifted, ratio floor 0.82", "nested-v3-shifted-floor082-20260909"),
        ("nested v4 shifted, ratio floor 0.82", "nested-v4-shifted-floor082-20260909"),
    ]
    main_table = pd.DataFrame(summary_row(label, run) for label, run in main_runs)
    write_table(main_table, "main_results")

    occurrence_runs = [
        ("v3 monthly", "optimized-v3-extended-20260902"),
        ("CS-HCL v4 monthly", "optimized-v4-extended-20260908"),
        ("v3 shifted", "optimized-v3-shifted-20260908"),
        ("CS-HCL v4 shifted", "optimized-v4-shifted-20260909"),
    ]
    occurrence_table = pd.DataFrame(occurrence_row(label, run) for label, run in occurrence_runs)
    write_table(occurrence_table, "occurrence_calibration")

    cost_runs = [
        ("Monthly", "v3", "optimized-v3-extended-20260902"),
        ("Monthly", "CS-HCL v4", "optimized-v4-extended-20260908"),
        ("Shifted", "v3", "optimized-v3-shifted-20260908"),
        ("Shifted", "CS-HCL v4", "optimized-v4-shifted-20260909"),
    ]
    cost_table = pd.DataFrame(
        row
        for grid, model, run in cost_runs
        for row in asymmetric_cost_rows(grid, model, run)
    )
    write_table(cost_table, "asymmetric_cost_sensitivity")

    ablation_runs = [
        ("v3", "optimized-v3-extended-20260902"),
        ("full cold-start expert", "cold-start-expert-v1-20260908"),
        ("cold magnitude only", "cold-start-quantity-v2-20260908"),
        ("high-tail expert", "blockbuster-expert-v1-20260908"),
        ("peer features in both components", "sibling-features-v1-20260908"),
        ("peer sums only in occurrence", "optimized-v4-sums-only-20260909"),
        ("target shares only in occurrence", "optimized-v4-shares-only-20260909"),
        ("sibling breadth only in occurrence", "optimized-v4-breadth-only-20260909"),
        ("complete CS-HCL v4", "optimized-v4-extended-20260908"),
    ]
    ablation_table = pd.DataFrame(summary_row(label, run) for label, run in ablation_runs)
    write_table(ablation_table, "ablation_results")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for axis, grid, base_run, candidate_run in [
        (axes[0], "Monthly origins", "optimized-v3-extended-20260902", "optimized-v4-extended-20260908"),
        (axes[1], "Shifted origins", "optimized-v3-shifted-20260908", "optimized-v4-shifted-20260909"),
    ]:
        base = scorecard(base_run)
        candidate = scorecard(candidate_run)
        delta = candidate.global_wmape.to_numpy() - base.global_wmape.to_numpy()
        colors = ["#2f7d32" if value < 0 else "#b23a33" for value in delta]
        axis.bar(range(1, len(delta) + 1), delta, color=colors)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(grid)
        axis.set_xlabel("Rolling origin index")
        axis.set_xticks(range(1, len(delta) + 1))
    axes[0].set_ylabel("WMAPE difference: CS-HCL minus v3")
    fig.tight_layout()
    fig.savefig(OUTPUT / "origin_wmape_deltas.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1), sharex=True, sharey=True)
    for axis, title, run_pairs in [
        (axes[0], "Monthly origins", [("v3", "optimized-v3-extended-20260902"), ("CS-HCL v4", "optimized-v4-extended-20260908")]),
        (axes[1], "Shifted origins", [("v3", "optimized-v3-shifted-20260908"), ("CS-HCL v4", "optimized-v4-shifted-20260909")]),
    ]:
        for label, run in run_pairs:
            frame = contexts(run)
            observed, predicted = calibration_curve(
                frame.actual_qty.gt(0).astype(int), frame.prob.clip(0, 1),
                n_bins=10, strategy="quantile",
            )
            axis.plot(predicted, observed, marker="o", linewidth=1.4, markersize=4, label=label)
        axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Mean predicted probability")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Observed occurrence rate")
    fig.tight_layout()
    fig.savefig(OUTPUT / "occurrence_reliability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    plot = ablation_table.copy()
    axis.scatter(plot.mean_anchor_ratio, plot.mean_anchor_wmape, color="#245a9b", s=45)
    label_offsets = {
        "v3": (8, -14),
        "peer sums only in occurrence": (8, 7),
        "sibling breadth only in occurrence": (8, -13),
        "complete CS-HCL v4": (8, 8),
        "cold magnitude only": (8, -12),
    }
    for row in plot.itertuples():
        offset = label_offsets.get(row.label, (5, 4))
        axis.annotate(row.label, (row.mean_anchor_ratio, row.mean_anchor_wmape), xytext=offset, textcoords="offset points", fontsize=7)
    axis.set_xlabel("Mean forecast-to-actual ratio")
    axis.set_ylabel("Mean anchor WMAPE")
    axis.set_title("Accuracy-bias trade-off across ablations")
    axis.set_ylim(plot.mean_anchor_wmape.min() - 0.0025, plot.mean_anchor_wmape.max() + 0.002)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "ablation_pareto.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.set_xlim(0, 10.5)
    axis.set_ylim(0, 5)
    axis.axis("off")

    def box(x: float, y: float, width: float, height: float, label: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.04",
            facecolor=color, edgecolor="#2b2b2b", linewidth=1.1,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9)

    def arrow(start: tuple[float, float], end: tuple[float, float], style: str = "-") -> None:
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5, "linestyle": style})

    box(0.3, 3.35, 2.3, 0.9, "SKU history, metadata,\ncalendar and raw hierarchy", "#dceafb")
    box(0.3, 1.25, 2.3, 0.9, "Peer-exclusive sibling\nactivity, share and breadth", "#e4f3df")
    box(3.45, 3.35, 2.25, 0.9, "Occurrence classifier\nP(Y > 0 | X)", "#fbe5c8")
    box(3.45, 1.25, 2.25, 0.9, "Positive magnitude\nconditional L1/L2 blend", "#f4dbe8")
    box(6.55, 2.3, 1.75, 0.9, "Signal-regime\nprobability gate", "#eee6fa")
    box(9.0, 2.3, 1.2, 0.9, "SKU H30\nforecast", "#d9efee")
    arrow((2.6, 3.8), (3.45, 3.8))
    arrow((2.6, 1.7), (3.45, 3.55))
    arrow((2.6, 3.55), (3.45, 1.85))
    arrow((5.7, 3.8), (6.55, 2.95))
    arrow((5.7, 1.7), (6.55, 2.55))
    arrow((8.3, 2.75), (9.0, 2.75))
    axis.text(3.05, 1.0, "excluded", ha="center", va="center", fontsize=8, color="#a33a32")
    axis.plot([2.65, 3.35], [1.45, 1.45], color="#a33a32", linewidth=2, linestyle="--")
    axis.set_title("Component-specific hierarchical cross-learning", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT / "architecture.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "main_runs": [run for _, run in main_runs],
        "ablation_runs": [run for _, run in ablation_runs],
        "outputs": [
            "main_results.csv", "main_results.md", "ablation_results.csv",
            "ablation_results.md", "origin_wmape_deltas.png", "ablation_pareto.png",
            "architecture.png", "occurrence_calibration.csv", "occurrence_calibration.md",
            "occurrence_reliability.png", "asymmetric_cost_sensitivity.csv",
            "asymmetric_cost_sensitivity.md",
        ],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
