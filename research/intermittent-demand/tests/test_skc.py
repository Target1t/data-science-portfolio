import numpy as np
import pandas as pd

from phase9.skc import build_skc_forecast


def test_skc_audit_matches_sum_times_calibration_factor() -> None:
    sku = pd.DataFrame({"sku_id": ["a", "b"], "style_id": ["s", "s"], "forecast_qty_h30": [2.0, 3.0], "product_name": ["x", "x"]})
    _, audit = build_skc_forecast(sku, {0: 1.1, 1: 0.9, 2: 1.1})

    assert np.allclose(audit["forecast_qty_h30"], audit["sku_sum_qty"] * audit["skc_calibration_factor"])
