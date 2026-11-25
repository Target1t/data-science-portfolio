from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase9.backtest import run_backtest
from phase9.inference import train_final, predict_final
from phase9.skc import build_skc_forecast
import pandas as pd
parser=argparse.ArgumentParser(); parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1]); parser.add_argument('--config',type=Path,default=Path('configs/baseline_e3.json')); parser.add_argument('--run-id',required=True)
args=parser.parse_args(); root=args.project_root.resolve(); config=root/args.config; backtest=run_backtest(root,config,args.run_id+'-backtest'); run=train_final(root,config,args.run_id); sku=predict_final(run); client,audit=build_skc_forecast(pd.read_csv(sku),{0:1.1,1:.9,2:1.1}); client.to_csv(run/'skc_forecast_h30.csv',index=False,encoding='utf-8-sig'); audit.to_csv(run/'skc_forecast_h30_audit.csv',index=False,encoding='utf-8-sig'); print(run); print(backtest)
