from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase9.inference import predict_final
parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True)
args = parser.parse_args(); print(predict_final(args.run_dir))
