"""
Market Prior Model

Trains a strong baseline using market features and control features
to predict the target (shock_minus_pre). Outputs predicted mu_hat.

Core equation context:
    y = mu + r
    mu_hat = MarketPriorModel.predict(X_market_and_controls)
"""

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


class MarketPriorModel:
    """Strong market prior baseline using XGBoost on market + control features."""

    MARKET_FEATURES = [
        "pre_call_volatility",
        "returns",
        "volume",
    ]

    CONTROL_FEATURES = [
        "firm_size",
        "sector",
        "historical_volatility",
    ]

    DEFAULT_PARAMS = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 5,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
    }

    TUNING_GRID = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [200, 300, 500],
    }

    def __init__(
        self,
        params: Optional[dict] = None,
        feature_columns: Optional[list] = None,
        tune: bool = False,
    ):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.feature_columns = feature_columns or (
            self.MARKET_FEATURES + self.CONTROL_FEATURES
        )
        self.tune = tune
        self.model: Optional[XGBRegressor] = None
        self.best_params: Optional[dict] = None

    def _prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select and validate feature columns."""
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return X[self.feature_columns].copy()

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MarketPriorModel":
        """Fit the market prior model.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain all market + control feature columns.
        y : array-like
            Target values (shock_minus_pre).

        Returns
        -------
        self
        """
        X_clean = self._prepare_X(X)
        y = np.asarray(y, dtype=np.float64)

        if self.tune:
            base = XGBRegressor(**self.params)
            search = GridSearchCV(
                base,
                self.TUNING_GRID,
                scoring="neg_mean_squared_error",
                cv=3,
                refit=True,
                n_jobs=-1,
            )
            search.fit(X_clean, y)
            self.model = search.best_estimator_
            self.best_params = search.best_params_
        else:
            self.model = XGBRegressor(**self.params)
            self.model.fit(X_clean, y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict mu_hat for given features.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain all market + control feature columns.

        Returns
        -------
        np.ndarray
            Predicted mu_hat values.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_clean = self._prepare_X(X)
        return self.model.predict(X_clean)

    def save(self, path: str) -> None:
        """Save model to disk.

        Parameters
        ----------
        path : str
            File path (e.g. outputs/models/market_prior_xgb.pkl).
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Nothing to save.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "model": self.model,
            "params": self.params,
            "feature_columns": self.feature_columns,
            "best_params": self.best_params,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> "MarketPriorModel":
        """Load model from disk.

        Parameters
        ----------
        path : str
            File path to saved model.

        Returns
        -------
        self
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.model = state["model"]
        self.params = state["params"]
        self.feature_columns = state["feature_columns"]
        self.best_params = state.get("best_params")
        return self
