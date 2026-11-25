from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase9.skc import build_skc_forecast
parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True)
args = parser.parse_args(); run=args.run_dir; client,audit=build_skc_forecast(pd.read_csv(run/'sku_forecast_h30.csv'), {0:1.1,1:.9,2:1.1}); client.to_csv(run/'skc_forecast_h30.csv',index=False,encoding='utf-8-sig'); audit.to_csv(run/'skc_forecast_h30_audit.csv',index=False,encoding='utf-8-sig'); (run/'skc_manifest.json').write_text(json.dumps({'route':'sku_derived_skc_band_calibrated','factors':{'0':1.1,'1':.9,'2':1.1}},ensure_ascii=False,indent=2),encoding='utf-8'); print(run/'skc_forecast_h30.csv')
