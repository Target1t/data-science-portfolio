# Coupon Redemption and Promotion Abuse

A Meituan competition team studied which coupons get redeemed and which usage patterns warrant examination. The project links feature engineering, supervised prediction and anomaly detection in a single competition report.

## Explore the project

| File | What it contains |
| --- | --- |
| [Team 0839 report (Chinese)](report/competition-report.pdf) | Competition submission on coupon allocation and abuse detection. |
| [Redemption modeling](analysis/redemption.ipynb) | Behavioral features, resampling and LightGBM. |
| [Anomaly detection](analysis/abuse-detection.ipynb) | Isolation Forest analysis of coupon-use behavior. |
| [Feature construction](analysis/feature-engineering.ipynb) | Order value, visit frequency and redemption history. |

## Data and use

The portfolio includes aggregate figures. Transaction logs, user-level scores and serialized models are excluded. Notebook cell sources are preserved and historical outputs have been cleared.
