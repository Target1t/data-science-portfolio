import numpy as np
import pandas as pd

from phase9.model import HurdleModel


def test_hard_gate_sets_low_probability_quantities_to_zero() -> None:
    predictions = HurdleModel.gate(
        prob=np.array([0.26, 0.27]), qty=np.array([5.0, 5.0]), threshold=0.27
    )

    assert predictions.tolist() == [0.0, 5.0]


def test_model_accepts_categorical_features_and_unseen_values() -> None:
    frame = pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0], "category": ["a", "a", "b", "b"], "target": [0.0, 1.0, 0.0, 2.0]})
    config = {"gate_threshold": 0.2, "learning_rate": 0.1, "n_estimators": 5, "num_leaves": 3, "min_child_samples": 1, "subsample": 1.0, "colsample_bytree": 1.0, "random_state": 1}

    model = HurdleModel.fit(frame, ["value", "category"], "target", config)
    prediction = model.predict(pd.DataFrame({"value": [1.0], "category": ["unseen"]}))

    assert {"prob", "conditional_qty", "prediction_qty"}.issubset(prediction.columns)
    assert prediction.loc[0, "prediction_qty"] >= 0.0


def test_probability_weighting_scales_conditional_quantity() -> None:
    model = HurdleModel(None, None, ["value"], 0.2, constant_probability=0.5, constant_quantity=8.0, probability_power=1.0)

    prediction = model.predict(pd.DataFrame({"value": [1.0]}))

    assert prediction.loc[0, "conditional_qty"] == 4.0
    assert prediction.loc[0, "prediction_qty"] == 4.0


def test_quadrant_thresholds_apply_by_historical_signal() -> None:
    model = HurdleModel(
        None, None, ["replenish_days_90", "future_days_90"], 0.5,
        constant_probability=0.3, constant_quantity=4.0,
        quadrant_thresholds={"repl0_fut0": 0.5, "repl0_fut1": 0.2},
    )
    frame = pd.DataFrame({"replenish_days_90": [0.0, 0.0], "future_days_90": [0.0, 1.0]})

    prediction = model.predict(frame)

    assert prediction["prediction_qty"].tolist() == [0.0, 4.0]


def test_cold_start_threshold_is_observable_hard_route() -> None:
    model = HurdleModel(
        None, None, ["listing_age_days"], 0.5,
        constant_probability=0.3, constant_quantity=4.0,
        cold_start_max_age_days=90, cold_start_threshold=0.2,
    )

    prediction = model.predict(pd.DataFrame({"listing_age_days": [30.0, 300.0]}))

    assert prediction["is_cold_start"].tolist() == [1, 0]
    assert prediction["prediction_qty"].tolist() == [4.0, 0.0]


def test_blockbuster_output_defaults_to_zero_without_expert() -> None:
    model = HurdleModel(None, None, ["value"], 0.2, constant_probability=0.5, constant_quantity=8.0)

    prediction = model.predict(pd.DataFrame({"value": [1.0]}))

    assert prediction.loc[0, "blockbuster_prob"] == 0.0


def test_sibling_features_can_be_classifier_only() -> None:
    frame = pd.DataFrame({
        "value": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "sibling_replenish_sum_30": [0.0, 1.0, 0.0, 2.0, 1.0, 3.0],
        "target": [0.0, 1.0, 0.0, 2.0, 0.0, 4.0],
    })
    config = {
        "gate_threshold": 0.2, "learning_rate": 0.1, "n_estimators": 5,
        "num_leaves": 3, "min_child_samples": 1, "subsample": 1.0,
        "colsample_bytree": 1.0, "random_state": 1,
        "sibling_features_classifier_only": True,
    }

    model = HurdleModel.fit(
        frame, ["value", "sibling_replenish_sum_30"], "target", config
    )
    prediction = model.predict(frame.iloc[:1])

    assert model.regression_feature_cols == ["value"]
    assert model.classifier.n_features_in_ == 2
    assert model.regressor.n_features_in_ == 1
    assert prediction.loc[0, "prediction_qty"] >= 0.0


def test_expected_high_tail_blend_never_reduces_base_quantity() -> None:
    result = HurdleModel.blend_high_tail(
        base=np.array([10.0, 10.0]),
        high=np.array([30.0, 5.0]),
        probability=np.array([0.2, 0.2]),
        floor=0.0,
        full=1.0,
        strength=1.0,
        expected_excess_only=True,
    )

    assert result.tolist() == [14.0, 10.0]


def test_quantile_objective_receives_configured_alpha() -> None:
    frame = pd.DataFrame({
        "value": range(8),
        "target": [0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 8.0],
    })
    config = {
        "gate_threshold": 0.2, "learning_rate": 0.1, "n_estimators": 5,
        "num_leaves": 3, "min_child_samples": 1, "subsample": 1.0,
        "colsample_bytree": 1.0, "random_state": 1,
        "regression_objective": "quantile", "regression_alpha": 0.6,
    }

    model = HurdleModel.fit(frame, ["value"], "target", config)

    assert model.regressor.get_params()["objective"] == "quantile"
    assert model.regressor.get_params()["alpha"] == 0.6
