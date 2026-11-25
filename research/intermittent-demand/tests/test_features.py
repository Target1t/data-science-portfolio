import pandas as pd

from phase9.features import build_anchor_samples


CONFIG = {"lookback_days": 90, "forecast_days": 30, "purge_days": 30}


def test_anchor_features_exclude_anchor_and_future_rows() -> None:
    gold = pd.DataFrame(
        {
            "date": ["2026-01-12", "2026-04-11", "2026-04-12", "2026-04-13", "2026-05-01"],
            "buyer_id": ["b"] * 5,
            "sku_id": ["sku"] * 5,
            "style_id": ["style"] * 5,
            "product_name": ["product"] * 5,
            "qty_replenish": [1, 2, 99, 4, 5],
            "qty_future": [0, 3, 99, 0, 0],
            "qty_first_order": [10] * 5,
            "listing_date": ["2025-12-01"] * 5,
            "category": ["cat"] * 5,
            "sub_category": ["sub"] * 5,
            "season": ["spring"] * 5,
            "band": ["A"] * 5,
            "series": ["s"] * 5,
            "price_tag": [100] * 5,
            "size_id": ["M"] * 5,
            "color_id": ["black"] * 5,
        }
    )
    qfo = pd.DataFrame(
        {"sku_id": ["sku"], "qty_first_order": [10.0], "qfo_available_date": ["2026-01-01"]}
    )

    samples = build_anchor_samples(gold, qfo, pd.Timestamp("2026-04-12"), CONFIG)

    assert samples["feature_max_date"].lt(pd.Timestamp("2026-04-12")).all()
    assert samples["target_end"].eq(pd.Timestamp("2026-05-12")).all()
    assert samples.loc[0, "replenish_sum_90"] == 3.0
    # The label starts on the anchor, while the feature window ends before it.
    assert samples.loc[0, "actual_qty"] == 108.0


def test_qfo_quantity_is_hidden_until_available() -> None:
    gold = pd.DataFrame(
        {
            "date": ["2026-04-01"], "sku_id": ["sku"], "style_id": ["style"],
            "product_name": ["p"], "qty_replenish": [0], "qty_future": [0],
            "listing_date": ["2026-01-01"], "category": ["cat"], "sub_category": ["sub"],
            "season": ["spring"], "band": ["A"], "series": ["s"], "price_tag": [100],
            "size_id": ["M"], "color_id": ["black"],
        }
    )
    qfo = pd.DataFrame({"sku_id": ["sku"], "qty_first_order": [99.0], "qfo_available_date": ["2026-04-20"]})

    samples = build_anchor_samples(gold, qfo, pd.Timestamp("2026-04-12"), CONFIG)

    assert samples.loc[0, "qfo_available"] == 0.0
    assert samples.loc[0, "qfo_qty"] == 0.0


def test_hierarchical_features_share_sibling_sku_signal() -> None:
    rows = []
    for sku, qty in (("a", 2.0), ("b", 3.0)):
        rows.append({
            "date": "2026-04-01", "sku_id": sku, "style_id": "style", "product_name": "p",
            "qty_replenish": qty, "qty_future": 0, "listing_date": "2026-01-01",
            "category": "cat", "sub_category": "sub", "season": "spring", "band": "A",
            "series": "s", "price_tag": 100, "size_id": "M", "color_id": "black",
        })
    qfo = pd.DataFrame({"sku_id": ["a", "b"], "qty_first_order": [1, 1], "qfo_available_date": ["2026-01-01"] * 2})
    config = {**CONFIG, "hierarchical_features": True}

    samples = build_anchor_samples(pd.DataFrame(rows), qfo, pd.Timestamp("2026-04-12"), config)

    assert samples["skc_replenish_sum_30"].tolist() == [5.0, 5.0]
    assert samples["skc_active_skus_30"].tolist() == [2, 2]


def test_sibling_features_remove_target_sku_contribution() -> None:
    rows = []
    for sku, qty in (("a", 2.0), ("b", 3.0)):
        rows.append({
            "date": "2026-04-01", "sku_id": sku, "style_id": "style", "product_name": "p",
            "qty_replenish": qty, "qty_future": 0, "listing_date": "2026-01-01",
            "category": "cat", "sub_category": "sub", "season": "spring", "band": "A",
            "series": "s", "price_tag": 100, "size_id": "M", "color_id": "black",
        })
    qfo = pd.DataFrame({"sku_id": ["a", "b"], "qty_first_order": [1, 1], "qfo_available_date": ["2026-01-01"] * 2})
    config = {**CONFIG, "hierarchical_features": True, "sibling_features": True}

    samples = build_anchor_samples(pd.DataFrame(rows), qfo, pd.Timestamp("2026-04-12"), config)

    assert samples["sibling_replenish_sum_30"].tolist() == [3.0, 2.0]
    assert samples["sku_replenish_share_skc_30"].tolist() == [0.4, 0.6]
    assert samples["sibling_active_skus_30"].tolist() == [1.0, 1.0]


def test_multiscale_windows_create_nonoverlapping_time_bands() -> None:
    dates = ["2026-04-10", "2026-04-01", "2026-03-20", "2026-03-01"]
    gold = pd.DataFrame({
        "date": dates, "sku_id": ["sku"] * 4, "style_id": ["style"] * 4,
        "product_name": ["p"] * 4, "qty_replenish": [1, 2, 4, 8],
        "qty_future": [0, 0, 0, 0], "listing_date": ["2026-01-01"] * 4,
        "category": ["cat"] * 4, "sub_category": ["sub"] * 4,
        "season": ["spring"] * 4, "band": ["A"] * 4, "series": ["s"] * 4,
        "price_tag": [100] * 4, "size_id": ["M"] * 4, "color_id": ["black"] * 4,
    })
    qfo = pd.DataFrame({
        "sku_id": ["sku"], "qty_first_order": [1],
        "qfo_available_date": ["2026-01-01"],
    })
    config = {
        **CONFIG, "feature_windows": [7, 14, 30, 60, 90],
        "engineered_features": True,
    }

    samples = build_anchor_samples(gold, qfo, pd.Timestamp("2026-04-12"), config)

    assert samples.loc[0, "replenish_sum_14"] == 3
    assert samples.loc[0, "replenish_sum_60"] == 15
    assert samples.loc[0, "replenish_sum_band_15_30"] == 4
    assert samples.loc[0, "replenish_sum_band_31_60"] == 8
