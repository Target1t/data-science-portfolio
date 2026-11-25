from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase9.statistics import group_losses, hierarchical_bootstrap_delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-subdir", default="anchor_contexts")
    parser.add_argument("--candidate-subdir", default="anchor_contexts")
    args = parser.parse_args()

    grouped = []
    for candidate_path in sorted(glob.glob(str(args.candidate / args.candidate_subdir / "*.csv"))):
        name = Path(candidate_path).name
        base_path = args.base / args.base_subdir / name
        if base_path.exists():
            grouped.append(group_losses(pd.read_csv(base_path), pd.read_csv(candidate_path)))
    result = hierarchical_bootstrap_delta(grouped, iterations=args.iterations, seed=args.seed)
    result["paired_anchors"] = len(grouped)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
