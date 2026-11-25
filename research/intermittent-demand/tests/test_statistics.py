import pandas as pd

from phase9.statistics import group_losses, hierarchical_bootstrap_delta


def test_hierarchical_bootstrap_detects_uniformly_better_candidate() -> None:
    base = pd.DataFrame({
        "style_id": ["a", "b"],
        "sku_id": ["a1", "b1"],
        "actual_qty": [10.0, 20.0],
        "prediction_qty": [0.0, 0.0],
    })
    candidate = base.assign(prediction_qty=[10.0, 20.0])
    grouped = group_losses(base, candidate)

    result = hierarchical_bootstrap_delta([grouped, grouped], iterations=100, seed=1)

    assert result["ci_upper"] < 0.0
    assert result["probability_candidate_better"] == 1.0
