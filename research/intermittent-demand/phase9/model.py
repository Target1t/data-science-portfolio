"""Deterministic two-stage LightGBM hurdle model."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor


@dataclass
class HurdleModel:
    classifier: LGBMClassifier | None
    regressor: LGBMRegressor | None
    feature_cols: list[str]
    threshold: float
    constant_probability: float | None = None
    constant_quantity: float | None = None
    categorical_levels: dict[str, list[str]] | None = None
    target_transform: str = "none"
    quantity_scale: float = 1.0
    secondary_regressor: LGBMRegressor | None = None
    primary_weight: float = 1.0
    probability_power: float = 0.0
    secondary_min_primary_qty: float = 0.0
    quadrant_thresholds: dict[str, float] | None = None
    cold_start_classifier: LGBMClassifier | None = None
    cold_start_regressor: LGBMRegressor | None = None
    cold_start_max_age_days: float = 0.0
    cold_start_threshold: float | None = None
    cold_start_quantity_scale: float = 1.0
    blockbuster_classifier: LGBMClassifier | None = None
    blockbuster_regressor: LGBMRegressor | None = None
    blockbuster_target_qty: float = 25.0
    blockbuster_probability_floor: float = 1.0
    blockbuster_probability_full: float = 1.0
    blockbuster_blend_strength: float = 0.0
    blockbuster_skip_cold_start: bool = True
    blockbuster_expected_excess_only: bool = False
    regression_feature_cols: list[str] | None = None

    @staticmethod
    def gate(prob: np.ndarray, qty: np.ndarray, threshold: float) -> np.ndarray:
        return np.where(np.asarray(prob) >= threshold, np.maximum(np.asarray(qty), 0.0), 0.0)

    @staticmethod
    def blend_high_tail(
        base: np.ndarray,
        high: np.ndarray,
        probability: np.ndarray,
        floor: float,
        full: float,
        strength: float,
        expected_excess_only: bool = False,
    ) -> np.ndarray:
        """Probability-weight a latent high-tail expert without future-label routing."""

        full = max(float(full), float(floor) + 1e-9)
        weight = np.clip((np.asarray(probability) - floor) / (full - floor), 0.0, 1.0) * strength
        delta = np.asarray(high) - np.asarray(base)
        if expected_excess_only:
            delta = np.maximum(delta, 0.0)
        return np.asarray(base) + weight * delta

    @classmethod
    def fit(
        cls, frame: pd.DataFrame, feature_cols: list[str], target_col: str, config: dict[str, object]
    ) -> "HurdleModel":
        if frame.empty:
            raise ValueError("Cannot fit a hurdle model on an empty frame")
        target = pd.to_numeric(frame[target_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        x = frame.loc[:, feature_cols].copy()
        categorical_levels: dict[str, list[str]] = {}
        for column in feature_cols:
            if not pd.api.types.is_numeric_dtype(x[column]):
                values = x[column].fillna("__MISSING__").astype(str)
                levels = sorted(values.unique().tolist())
                categorical_levels[column] = levels
                x[column] = pd.Categorical(values, categories=levels)
        numeric_cols = [column for column in feature_cols if column not in categorical_levels]
        x[numeric_cols] = x[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        regression_feature_cols = list(feature_cols)
        if bool(config.get("sibling_features_classifier_only", False)):
            regression_feature_cols = [
                column for column in feature_cols
                if not column.startswith("sibling_") and "_share_skc_" not in column
            ]
        x_regression = x.loc[:, regression_feature_cols]
        regression_categoricals = [
            column for column in categorical_levels if column in regression_feature_cols
        ]
        positive = target.gt(0.0)
        threshold = float(config["gate_threshold"])
        sample_weight = None
        half_life = float(config.get("recency_half_life_days", 0.0))
        if half_life > 0 and "target_start" in frame:
            sample_dates = pd.to_datetime(frame["target_start"], errors="coerce")
            age_days = (sample_dates.max() - sample_dates).dt.days.fillna(0.0).clip(lower=0.0)
            sample_weight = np.power(0.5, age_days / half_life)
            sample_weight = sample_weight / sample_weight.mean()
        if positive.nunique() < 2:
            return cls(
                None, None, feature_cols, threshold,
                constant_probability=float(positive.iloc[0]),
                constant_quantity=float(target.mean()),
                categorical_levels=categorical_levels,
                quantity_scale=float(config.get("quantity_scale", 1.0)),
                primary_weight=float(config.get("primary_weight", 1.0)),
                probability_power=float(config.get("probability_power", 0.0)),
            )
        common = {
            "learning_rate": float(config["learning_rate"]),
            "n_estimators": int(config["n_estimators"]),
            "num_leaves": int(config["num_leaves"]),
            "min_child_samples": int(config["min_child_samples"]),
            "subsample": float(config["subsample"]),
            "colsample_bytree": float(config["colsample_bytree"]),
            "random_state": int(config["random_state"]),
            "n_jobs": 1,
            "verbosity": -1,
        }
        classifier = LGBMClassifier(**common)
        classifier.fit(
            x, positive.astype(int), sample_weight=sample_weight,
            categorical_feature=list(categorical_levels) or "auto",
        )
        target_transform = str(config.get("target_transform", "none"))
        regression_target = np.log1p(target.loc[positive]) if target_transform == "log1p" else target.loc[positive]
        regressor_params = dict(common)
        regressor_params["objective"] = str(config.get("regression_objective", "regression"))
        if regressor_params["objective"] == "tweedie":
            regressor_params["tweedie_variance_power"] = float(config.get("tweedie_variance_power", 1.3))
        if regressor_params["objective"] in {"quantile", "huber"}:
            regressor_params["alpha"] = float(config.get("regression_alpha", 0.9))
        regressor = LGBMRegressor(**regressor_params)
        positive_weight = sample_weight.loc[positive] if sample_weight is not None else None
        regressor.fit(
            x_regression.loc[positive], regression_target, sample_weight=positive_weight,
            categorical_feature=regression_categoricals or "auto",
        )
        secondary_regressor = None
        secondary_objective = config.get("secondary_regression_objective")
        if secondary_objective:
            secondary_params = dict(common)
            secondary_params["objective"] = str(secondary_objective)
            if secondary_params["objective"] in {"quantile", "huber"}:
                secondary_params["alpha"] = float(config.get("secondary_regression_alpha", 0.9))
            secondary_regressor = LGBMRegressor(**secondary_params)
            secondary_regressor.fit(
                x_regression.loc[positive], target.loc[positive], sample_weight=positive_weight,
                categorical_feature=regression_categoricals or "auto",
            )

        # A true cold-start regime is observable at scoring time, so it can use
        # a hard-routed expert.  Do not confuse this with an inactive/ice SKU:
        # the latter may be old and merely intermittent.
        cold_start_classifier = None
        cold_start_regressor = None
        cold_max_age = float(config.get("cold_start_max_age_days", 0.0))
        if cold_max_age > 0 and "listing_age_days" in frame:
            cold = pd.to_numeric(frame["listing_age_days"], errors="coerce").fillna(np.inf).le(cold_max_age)
            cold_positive = cold & positive
            min_cold_rows = int(config.get("cold_start_min_rows", 200))
            if (
                bool(config.get("cold_start_classifier_enabled", True))
                and int(cold.sum()) >= min_cold_rows
                and target.loc[cold].gt(0).nunique() == 2
            ):
                cold_start_classifier = LGBMClassifier(**common)
                cold_weight = sample_weight.loc[cold] if sample_weight is not None else None
                cold_start_classifier.fit(
                    x.loc[cold], positive.loc[cold].astype(int), sample_weight=cold_weight,
                    categorical_feature=list(categorical_levels) or "auto",
                )
            if int(cold_positive.sum()) >= max(20, min_cold_rows // 10):
                cold_params = dict(common)
                cold_params["objective"] = str(config.get("cold_start_regression_objective", "regression_l1"))
                cold_start_regressor = LGBMRegressor(**cold_params)
                cold_positive_weight = sample_weight.loc[cold_positive] if sample_weight is not None else None
                cold_start_regressor.fit(
                    x_regression.loc[cold_positive], target.loc[cold_positive], sample_weight=cold_positive_weight,
                    categorical_feature=regression_categoricals or "auto",
                )

        # Whether a SKU will be a blockbuster is unknown at scoring time.  A
        # classifier estimates that latent regime and softly blends a high-tail
        # quantity expert, avoiding any route based on the future label itself.
        blockbuster_classifier = None
        blockbuster_regressor = None
        blockbuster_target = float(config.get("blockbuster_target_qty", 25.0))
        blockbuster_strength = float(config.get("blockbuster_blend_strength", 0.0))
        blockbuster = target.gt(blockbuster_target)
        if blockbuster_strength > 0 and blockbuster.nunique() == 2:
            blockbuster_classifier = LGBMClassifier(**common)
            blockbuster_classifier.fit(
                x, blockbuster.astype(int), sample_weight=sample_weight,
                categorical_feature=list(categorical_levels) or "auto",
            )
            min_blockbusters = int(config.get("blockbuster_min_rows", 100))
            if int(blockbuster.sum()) >= min_blockbusters:
                blockbuster_params = dict(common)
                blockbuster_params["objective"] = str(config.get("blockbuster_regression_objective", "regression_l1"))
                blockbuster_regressor = LGBMRegressor(**blockbuster_params)
                blockbuster_weight = sample_weight.loc[blockbuster] if sample_weight is not None else None
                blockbuster_regressor.fit(
                    x_regression.loc[blockbuster], target.loc[blockbuster], sample_weight=blockbuster_weight,
                    categorical_feature=regression_categoricals or "auto",
                )
        return cls(
            classifier, regressor, feature_cols, threshold,
            categorical_levels=categorical_levels, target_transform=target_transform,
            quantity_scale=float(config.get("quantity_scale", 1.0)),
            secondary_regressor=secondary_regressor,
            primary_weight=float(config.get("primary_weight", 1.0)),
            probability_power=float(config.get("probability_power", 0.0)),
            secondary_min_primary_qty=float(config.get("secondary_min_primary_qty", 0.0)),
            quadrant_thresholds={str(k): float(v) for k, v in dict(config.get("quadrant_thresholds", {})).items()} or None,
            cold_start_classifier=cold_start_classifier,
            cold_start_regressor=cold_start_regressor,
            cold_start_max_age_days=cold_max_age,
            cold_start_threshold=float(config["cold_start_threshold"]) if "cold_start_threshold" in config else None,
            cold_start_quantity_scale=float(config.get("cold_start_quantity_scale", 1.0)),
            blockbuster_classifier=blockbuster_classifier,
            blockbuster_regressor=blockbuster_regressor,
            blockbuster_target_qty=blockbuster_target,
            blockbuster_probability_floor=float(config.get("blockbuster_probability_floor", 0.05)),
            blockbuster_probability_full=float(config.get("blockbuster_probability_full", 0.25)),
            blockbuster_blend_strength=blockbuster_strength,
            blockbuster_skip_cold_start=bool(config.get("blockbuster_skip_cold_start", True)),
            blockbuster_expected_excess_only=bool(config.get("blockbuster_expected_excess_only", False)),
            regression_feature_cols=regression_feature_cols,
        )

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame.loc[:, self.feature_cols].copy()
        categorical_levels = self.categorical_levels or {}
        for column, levels in categorical_levels.items():
            values = x[column].fillna("__MISSING__").astype(str)
            values = values.where(values.isin(levels))
            x[column] = pd.Categorical(values, categories=levels)
        numeric_cols = [column for column in self.feature_cols if column not in categorical_levels]
        x[numeric_cols] = x[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        regression_feature_cols = getattr(self, "regression_feature_cols", None) or self.feature_cols
        x_regression = x.loc[:, regression_feature_cols]
        if self.classifier is None or self.regressor is None:
            probability = np.full(len(x), self.constant_probability or 0.0)
            quantity = np.full(len(x), self.constant_quantity or 0.0)
        else:
            probability = self.classifier.predict_proba(x)[:, 1]
            quantity = self.regressor.predict(x_regression)
        if self.target_transform == "log1p":
            quantity = np.expm1(quantity)
        secondary = getattr(self, "secondary_regressor", None)
        if secondary is not None:
            secondary_quantity = np.maximum(secondary.predict(x_regression), 0.0)
            weight = float(getattr(self, "primary_weight", 1.0))
            blended_quantity = weight * quantity + (1.0 - weight) * secondary_quantity
            minimum = float(getattr(self, "secondary_min_primary_qty", 0.0))
            quantity = np.where(quantity >= minimum, blended_quantity, quantity)

        cold_mask = np.zeros(len(x), dtype=bool)
        cold_classifier = getattr(self, "cold_start_classifier", None)
        cold_regressor = getattr(self, "cold_start_regressor", None)
        cold_max_age = float(getattr(self, "cold_start_max_age_days", 0.0))
        if cold_max_age > 0 and "listing_age_days" in frame:
            cold_mask = pd.to_numeric(frame["listing_age_days"], errors="coerce").fillna(np.inf).to_numpy() <= cold_max_age
            if cold_classifier is not None:
                cold_probability = cold_classifier.predict_proba(x)[:, 1]
                probability = np.where(cold_mask, cold_probability, probability)
            if cold_regressor is not None:
                cold_quantity = np.maximum(cold_regressor.predict(x_regression), 0.0) * float(
                    getattr(self, "cold_start_quantity_scale", 1.0)
                )
                quantity = np.where(cold_mask, cold_quantity, quantity)

        blockbuster_probability = np.zeros(len(x), dtype=float)
        blockbuster_quantity = np.zeros(len(x), dtype=float)
        blockbuster_classifier = getattr(self, "blockbuster_classifier", None)
        blockbuster_regressor = getattr(self, "blockbuster_regressor", None)
        if blockbuster_classifier is not None and blockbuster_regressor is not None:
            blockbuster_probability = blockbuster_classifier.predict_proba(x)[:, 1]
            blockbuster_quantity = np.maximum(blockbuster_regressor.predict(x_regression), 0.0)
            floor = float(getattr(self, "blockbuster_probability_floor", 0.05))
            full = float(getattr(self, "blockbuster_probability_full", 0.25))
            strength = float(getattr(self, "blockbuster_blend_strength", 0.0))
            weight = np.clip((blockbuster_probability - floor) / (max(full, floor + 1e-9) - floor), 0.0, 1.0)
            weight *= strength
            if bool(getattr(self, "blockbuster_skip_cold_start", True)):
                weight = np.where(cold_mask, 0.0, weight)
            high_tail_probability = floor + weight * (max(full, floor + 1e-9) - floor) / max(strength, 1e-9)
            quantity = self.blend_high_tail(
                quantity,
                blockbuster_quantity,
                high_tail_probability,
                floor,
                full,
                strength,
                bool(getattr(self, "blockbuster_expected_excess_only", False)),
            )
        probability_power = float(getattr(self, "probability_power", 0.0))
        quantity = np.maximum(quantity, 0.0) * self.quantity_scale * np.power(probability, probability_power)
        thresholds: float | np.ndarray = self.threshold
        quadrant_thresholds = getattr(self, "quadrant_thresholds", None)
        if quadrant_thresholds and {"replenish_days_90", "future_days_90"}.issubset(frame.columns):
            replenish = pd.to_numeric(frame["replenish_days_90"], errors="coerce").fillna(0.0).to_numpy() > 0
            future = pd.to_numeric(frame["future_days_90"], errors="coerce").fillna(0.0).to_numpy() > 0
            thresholds = np.select(
                [replenish & future, replenish, future],
                [
                    quadrant_thresholds.get("repl1_fut1", self.threshold),
                    quadrant_thresholds.get("repl1_fut0", self.threshold),
                    quadrant_thresholds.get("repl0_fut1", self.threshold),
                ],
                default=quadrant_thresholds.get("repl0_fut0", self.threshold),
            )
        cold_threshold = getattr(self, "cold_start_threshold", None)
        if cold_threshold is not None and cold_mask.any():
            if np.isscalar(thresholds):
                thresholds = np.full(len(x), float(thresholds))
            thresholds = np.where(cold_mask, float(cold_threshold), thresholds)
        return pd.DataFrame({
            "prob": probability,
            "conditional_qty": quantity,
            "prediction_qty": self.gate(probability, quantity, thresholds),
            "blockbuster_prob": blockbuster_probability,
            "blockbuster_conditional_qty": blockbuster_quantity,
            "is_cold_start": cold_mask.astype(int),
        })

    def save(self, path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "HurdleModel":
        with path.open("rb") as handle:
            return pickle.load(handle)
