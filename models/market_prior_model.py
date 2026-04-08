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
from itertools import product
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - environment dependent
    XGBRegressor = None


class MarketPriorModel:
    """Strong market prior baseline using XGBoost on clean pre-call features.

    The main experiment treats PREC as a post-call correction problem:
    - `mu` must use only decision-time-safe pre-call market and control inputs
    - ECC-derived features belong to the correction (`z`) layer instead
    - within-call / post-call outcomes are excluded from the prior
    """

    MARKET_FEATURES = [
        "pre_60m_rv",
        "pre_60m_vw_rv",
        "pre_60m_volume_sum",
    ]

    CONTROL_FEATURES = [
        "scheduled_hour_et",
        "revenue_surprise_pct",
        "ebitda_surprise_pct",
        "eps_gaap_surprise_pct",
        "analyst_eps_norm_num_est",
        "analyst_eps_norm_std",
        "analyst_revenue_num_est",
        "analyst_revenue_std",
        "analyst_net_income_num_est",
        "analyst_net_income_std",
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
        self.model: Optional[Any] = None
        self.best_params: Optional[dict] = None
        self.best_val_mse: Optional[float] = None

    @staticmethod
    def _require_xgboost() -> None:
        """Raise a clear error if xgboost is unavailable."""
        if XGBRegressor is None:
            raise ImportError(
                "xgboost is required for MarketPriorModel. "
                "Install it with `pip install xgboost` or `conda install -c conda-forge xgboost`."
            )

    def _prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select and validate feature columns."""
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return X[self.feature_columns].copy()

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "MarketPriorModel":
        """Fit the market prior model.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain all market + control feature columns.
        y : array-like
            Target values (shock_minus_pre).
        X_val : pd.DataFrame, optional
            Validation features used for model selection when tune=True.
        y_val : array-like, optional
            Validation targets used for model selection when tune=True.

        Returns
        -------
        self
        """
        self._require_xgboost()
        X_clean = self._prepare_X(X)
        y = np.asarray(y, dtype=np.float64)

        if self.tune:
            if X_val is None or y_val is None:
                raise ValueError("Validation data is required when tune=True.")

            X_val_clean = self._prepare_X(X_val)
            y_val = np.asarray(y_val, dtype=np.float64)

            best_model = None
            best_params = None
            best_val_mse = None
            grid_names = list(self.TUNING_GRID.keys())

            for grid_values in product(*(self.TUNING_GRID[name] for name in grid_names)):
                candidate_params = dict(zip(grid_names, grid_values))
                full_params = {**self.params, **candidate_params}
                candidate_model = XGBRegressor(**full_params)
                candidate_model.fit(X_clean, y)
                candidate_pred = candidate_model.predict(X_val_clean)
                candidate_mse = float(np.mean((y_val - candidate_pred) ** 2))

                if best_val_mse is None or candidate_mse < best_val_mse:
                    best_model = candidate_model
                    best_params = candidate_params
                    best_val_mse = candidate_mse

            self.model = best_model
            self.best_params = best_params
            self.best_val_mse = best_val_mse
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
            "best_val_mse": self.best_val_mse,
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
        self.best_val_mse = state.get("best_val_mse")
        return self
