import numpy as np
import pandas as pd

from phase9.catch import DirectLossGate, PreferenceGate, blend_probability, blend_quantity, preference_target
from scripts.run_catch_prequential import select_constant_weights


def paired_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "actual_qty": [0.0, 0.0, 2.0, 8.0],
        "self_probability": [0.1, 0.7, 0.3, 0.8],
        "peer_probability": [0.2, 0.2, 0.8, 0.7],
        "self_quantity": [1.0, 2.0, 1.5, 5.0],
        "peer_quantity": [1.2, 1.0, 2.2, 8.0],
        "replenish_days_90": [0, 1, 2, 4],
        "future_days_90": [0, 0, 1, 3],
        "listing_age_days": [20, 200, 80, 400],
        "category": ["a", "a", "b", "b"],
    })


def test_component_blends_recover_endpoints():
    probability = blend_probability(np.array([.2, .3]), np.array([.8, .7]), np.array([0, 1]))
    quantity = blend_quantity(np.array([2, 3]), np.array([8, 9]), np.array([0, 1]))
    np.testing.assert_allclose(probability, [.2, .7])
    np.testing.assert_allclose(quantity, [2, 9])


def test_preference_targets_are_component_specific():
    frame = paired_frame()
    assert preference_target(frame, "occurrence").tolist() == [False, True, True, False]
    assert preference_target(frame, "magnitude").tolist() == [False, True, True, True]


def test_preference_gate_returns_bounded_weights():
    frame = pd.concat([paired_frame()] * 20, ignore_index=True)
    gate = PreferenceGate.fit(frame, "occurrence")
    weights = gate.predict(frame)
    assert weights.shape == (len(frame),)
    assert np.all((weights >= 0) & (weights <= 1))


def test_direct_loss_gate_returns_bounded_weights():
    frame = pd.concat([paired_frame()] * 30, ignore_index=True)
    for component in ("occurrence", "magnitude"):
        gate = DirectLossGate.fit(frame, component)
        weights = gate.predict(frame)
        assert weights.shape == (len(frame),)
        assert np.all((weights >= 0) & (weights <= 1))


def test_nested_constant_selector_recovers_peer_endpoints():
    frame = paired_frame()
    frame["peer_probability"] = [0.01, 0.01, 0.99, 0.99]
    frame["peer_quantity"] = frame.actual_qty
    config = {"gate_threshold": .5, "quadrant_thresholds": {}}
    occurrence, magnitude, _, _ = select_constant_weights([frame], config, [0.0, 1.0])
    assert occurrence == 1.0
    assert magnitude == 1.0
