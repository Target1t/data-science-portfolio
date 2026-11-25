from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase9.backtest import run_backtest

parser = argparse.ArgumentParser(); parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1]); parser.add_argument("--config", type=Path, default=Path("configs/baseline_e3.json")); parser.add_argument("--run-id", required=True); parser.add_argument("--anchors", nargs="+")
args = parser.parse_args(); print(run_backtest(args.project_root, args.project_root / args.config, args.run_id, anchors=args.anchors))
