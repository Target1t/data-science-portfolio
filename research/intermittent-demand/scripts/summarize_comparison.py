from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    base = pd.read_csv(args.base / "scorecard.csv")
    candidate = pd.read_csv(args.candidate / "scorecard.csv")
    keys = [
        "global_wmape", "global_ratio", "auc", "f1", "sku_p50",
        "within_20pct_rate", "within_30pct_rate", "within_50pct_rate",
        "catastrophic_under_rate", "catastrophic_over_rate", "false_zero_rate",
        "false_positive_rate_zero_true", "zero_true_pred_ge_3_rate",
        "rank_corr_positive_skus", "top20_true_volume_capture",
        "4_25_under_wape", "4_25_over_wape",
        "blockbuster_under_wape", "blockbuster_over_wape",
    ]
    for key in keys:
        before = float(base[key].mean())
        after = float(candidate[key].mean())
        print(f"{key:36s} {before:.6f} -> {after:.6f} ({after-before:+.6f})")
    aligned = min(len(base), len(candidate))
    wins = int((candidate.global_wmape.iloc[:aligned].to_numpy() < base.global_wmape.iloc[:aligned].to_numpy()).sum())
    print(f"anchor_wins: {wins}/{aligned}")
    print(f"wmape_std: {candidate.global_wmape.std(ddof=0):.6f}")
    print(f"wmape_worst: {candidate.global_wmape.max():.6f}")


if __name__ == "__main__":
    main()
