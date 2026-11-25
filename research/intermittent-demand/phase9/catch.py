"""Cross-fitted adaptive transfer utilities for component-wise hurdle models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.optimize import minimize


EPS = 1e-6


def blend_probability(self_probability: np.ndarray, peer_probability: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Convex interpolation in log-odds space."""
    self_p = np.clip(np.asarray(self_probability, dtype=float), EPS, 1 - EPS)
    peer_p = np.clip(np.asarray(peer_probability, dtype=float), EPS, 1 - EPS)
    alpha = np.clip(np.asarray(weight, dtype=float), 0.0, 1.0)
    self_logit = np.log(self_p / (1 - self_p))
    peer_logit = np.log(peer_p / (1 - peer_p))
    value = self_logit + alpha * (peer_logit - self_logit)
    return 1 / (1 + np.exp(-np.clip(value, -35, 35)))


def blend_quantity(self_quantity: np.ndarray, peer_quantity: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Convex interpolation on log1p quantity to preserve non-negativity."""
    self_q = np.maximum(np.asarray(self_quantity, dtype=float), 0.0)
    peer_q = np.maximum(np.asarray(peer_quantity, dtype=float), 0.0)
    alpha = np.clip(np.asarray(weight, dtype=float), 0.0, 1.0)
    return np.expm1(np.log1p(self_q) + alpha * (np.log1p(peer_q) - np.log1p(self_q)))


def transfer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build observable reliability features from paired out-of-sample forecasts."""
    result = pd.DataFrame(index=frame.index)
    for column in ("replenish_days_90", "future_days_90", "listing_age_days"):
        result[column] = pd.to_numeric(frame.get(column, 0.0), errors="coerce")
    result["self_probability"] = frame.self_probability
    result["peer_probability"] = frame.peer_probability
    result["probability_disagreement"] = (frame.peer_probability - frame.self_probability).abs()
    result["log_self_quantity"] = np.log1p(frame.self_quantity.clip(lower=0.0))
    result["log_peer_quantity"] = np.log1p(frame.peer_quantity.clip(lower=0.0))
    result["quantity_disagreement"] = (result.log_peer_quantity - result.log_self_quantity).abs()
    for column in frame.columns:
        if column.startswith("sibling_") or "_share_skc_" in column:
            result[column] = pd.to_numeric(frame[column], errors="coerce")
    result["category"] = frame.get("category", "__MISSING__").fillna("__MISSING__").astype(str)
    return result


def preference_target(frame: pd.DataFrame, component: str) -> pd.Series:
    """Return whether the peer model has smaller per-row proper/component loss."""
    actual = pd.to_numeric(frame.actual_qty, errors="coerce").fillna(0.0).clip(lower=0.0)
    if component == "occurrence":
        truth = actual.gt(0).astype(float).to_numpy()
        self_p = np.clip(frame.self_probability.to_numpy(float), EPS, 1 - EPS)
        peer_p = np.clip(frame.peer_probability.to_numpy(float), EPS, 1 - EPS)
        self_loss = -(truth * np.log(self_p) + (1 - truth) * np.log(1 - self_p))
        peer_loss = -(truth * np.log(peer_p) + (1 - truth) * np.log(1 - peer_p))
    elif component == "magnitude":
        self_loss = np.abs(np.log1p(actual) - np.log1p(frame.self_quantity.clip(lower=0.0)))
        peer_loss = np.abs(np.log1p(actual) - np.log1p(frame.peer_quantity.clip(lower=0.0)))
    else:
        raise ValueError(f"Unknown component: {component}")
    return pd.Series(np.asarray(peer_loss) < np.asarray(self_loss), index=frame.index, dtype=int)


@dataclass
class PreferenceGate:
    """Regularized gate that predicts when peer transfer beats self-only learning."""

    model: Pipeline | None
    constant: float

    @classmethod
    def fit(cls, frame: pd.DataFrame, component: str, feature_set: str = "all") -> "PreferenceGate":
        subset = frame if component == "occurrence" else frame.loc[frame.actual_qty.gt(0)]
        target = preference_target(subset, component)
        if target.empty or target.nunique() < 2:
            return cls(None, float(target.mean()) if len(target) else 0.0)
        features = transfer_features(subset)
        prediction_columns = {
            "self_probability", "peer_probability", "probability_disagreement",
            "log_self_quantity", "log_peer_quantity", "quantity_disagreement",
        }
        if feature_set == "prediction":
            selected = [column for column in features if column in prediction_columns]
        elif feature_set == "reliability":
            selected = [column for column in features if column not in prediction_columns]
        elif feature_set == "all":
            selected = list(features.columns)
        else:
            raise ValueError(f"Unknown gate feature set: {feature_set}")
        numeric = [column for column in selected if column != "category"]
        transformers = [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric)
        ]
        if "category" in selected:
            transformers.append(("category", OneHotEncoder(handle_unknown="ignore", min_frequency=25), ["category"]))
        transformer = ColumnTransformer(transformers)
        model = Pipeline([
            ("features", transformer),
            # The predicted probability is used directly as a mixture weight.
            # Class balancing would shift its base rate toward 0.5 and destroy
            # that probability interpretation, so the gate remains unweighted.
            ("gate", LogisticRegression(C=0.1, max_iter=500, random_state=2028)),
        ])
        model.fit(features.loc[:, selected], target)
        return cls(model, float(target.mean()))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(frame), self.constant)
        return self.model.predict_proba(transfer_features(frame))[:, 1]

    def coefficients(self) -> pd.DataFrame:
        """Return standardized logistic-gate coefficients for auditability."""
        if self.model is None:
            return pd.DataFrame(columns=["feature", "coefficient"])
        transformer = self.model.named_steps["features"]
        gate = self.model.named_steps["gate"]
        return pd.DataFrame({
            "feature": transformer.get_feature_names_out(),
            "coefficient": gate.coef_[0],
        }).sort_values("coefficient", key=lambda values: values.abs(), ascending=False)


@dataclass
class DirectLossGate:
    """Convex-mixture gate fitted directly to component predictive loss."""

    component: str
    feature_cols: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    coefficients_: np.ndarray

    @staticmethod
    def _columns(frame: pd.DataFrame, feature_set: str) -> list[str]:
        prediction_columns = {
            "self_probability", "peer_probability", "probability_disagreement",
            "log_self_quantity", "log_peer_quantity", "quantity_disagreement",
        }
        available = [column for column in transfer_features(frame).columns if column != "category"]
        if feature_set == "prediction":
            return [column for column in available if column in prediction_columns]
        if feature_set == "reliability":
            return [column for column in available if column not in prediction_columns]
        if feature_set == "all":
            return available
        raise ValueError(f"Unknown gate feature set: {feature_set}")

    @classmethod
    def fit(
        cls, frame: pd.DataFrame, component: str, feature_set: str = "all", ridge: float = .01
    ) -> "DirectLossGate":
        subset = frame if component == "occurrence" else frame.loc[frame.actual_qty.gt(0)].copy()
        columns = cls._columns(subset, feature_set)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        values = scaler.fit_transform(imputer.fit_transform(transfer_features(subset).loc[:, columns]))
        design = np.column_stack([np.ones(len(values)), values])
        actual = subset.actual_qty.to_numpy(float)

        if component == "occurrence":
            truth = (actual > 0).astype(float)
            self_p = np.clip(subset.self_probability.to_numpy(float), EPS, 1 - EPS)
            peer_p = np.clip(subset.peer_probability.to_numpy(float), EPS, 1 - EPS)
            self_value = np.log(self_p / (1 - self_p))
            delta = np.log(peer_p / (1 - peer_p)) - self_value
        elif component == "magnitude":
            truth = np.log1p(actual)
            self_value = np.log1p(subset.self_quantity.clip(lower=0).to_numpy(float))
            delta = np.log1p(subset.peer_quantity.clip(lower=0).to_numpy(float)) - self_value
        else:
            raise ValueError(f"Unknown component: {component}")

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            alpha = 1.0 / (1.0 + np.exp(-np.clip(design @ theta, -30, 30)))
            prediction = self_value + alpha * delta
            if component == "occurrence":
                probability = 1.0 / (1.0 + np.exp(-np.clip(prediction, -30, 30)))
                loss = -np.mean(truth * np.log(probability + EPS) + (1 - truth) * np.log(1 - probability + EPS))
                residual = probability - truth
            else:
                residual = prediction - truth
                loss = np.mean(residual ** 2)
                residual = 2 * residual
            penalty = ridge * np.sum(theta[1:] ** 2)
            gradient = design.T @ (residual * delta * alpha * (1 - alpha)) / len(design)
            gradient[1:] += 2 * ridge * theta[1:]
            return float(loss + penalty), gradient

        result = minimize(
            objective, np.zeros(design.shape[1]), method="L-BFGS-B", jac=True,
            options={"maxiter": 250, "ftol": 1e-10},
        )
        if not result.success:
            raise RuntimeError(f"Direct gate optimization failed: {result.message}")
        return cls(component, columns, imputer, scaler, result.x)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = self.scaler.transform(
            self.imputer.transform(transfer_features(frame).loc[:, self.feature_cols])
        )
        design = np.column_stack([np.ones(len(values)), values])
        return 1.0 / (1.0 + np.exp(-np.clip(design @ self.coefficients_, -30, 30)))

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": ["intercept", *self.feature_cols],
            "coefficient": self.coefficients_,
        }).sort_values("coefficient", key=lambda values: values.abs(), ascending=False)
