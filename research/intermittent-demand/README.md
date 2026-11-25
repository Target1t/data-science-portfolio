# Intermittent Demand Forecasting for B2B Replenishment

A two-part forecasting model separates the chance of a replenishment order from its size. The experiments compare how products borrow information across a merchandise hierarchy and evaluate forecasts over successive time windows.

## Explore the project

| File | What it contains |
| --- | --- |
| [Research manuscript](paper/manuscript.pdf) | Component-specific hierarchical cross-learning for intermittent demand. |
| [Forecasting model](phase9/model.py) | Implementation of the two-part demand model. |
| [Temporal backtest](phase9/backtest.py) | Forecast origins, evaluation windows and scoring. |
| [Synthetic experiment](scripts/run_synthetic_catch.py) | Compares information sharing for occurrence and positive quantity. |
| [Synthetic results](paper/generated/synthetic_catch_results_summary.csv) | Forecast errors and component weights from the saved simulation. |

## Data and use

The synthetic experiment can be launched from this project directory with `python -m scripts.run_synthetic_catch`. The production backtest expects the original private order snapshot. The portfolio includes the source, configurations, tests and synthetic results.
