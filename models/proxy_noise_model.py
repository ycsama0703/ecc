"""
Proxy Noise Model

Maps proxy features to estimated ECC error variance sigma^2.

Core equation context:
    u       = (r_tilde - z)^2          (squared ECC residual error)
    sigma2  = ProxyNoiseModel.predict(X_proxy)

Input proxy features (per data_contract.md):
    - transcript_coverage
    - alignment_score
    - audio_completeness

Notes (per code_structure.md 4.3):
    - should support monotone mapping
    - may use isotonic regression or monotone boosting
"""

import os
import pickle
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ProxyNoiseModel:
    """Map proxy features to estimated error variance sigma^2.

    Supports three mapping modes:
    - "isotonic":  First reduces proxies to a single score via Ridge,
                   then applies IsotonicRegression to enforce monotonicity.
                   This is the recommended default for the reliability story.
    - "monotone_boost": GradientBoosting with monotone_constraints so that
                        higher proxy quality maps to lower noise.
    - "ridge":     Plain Ridge regression (unconstrained, for ablation A6).
    """

    PROXY_FEATURES = [
        "transcript_coverage",
        "alignment_score",
        "audio_completeness",
    ]

    DEFAULT_PARAMS = {
        "mode": "isotonic",
        "ridge_alpha": 1.0,
        "gb_n_estimators": 200,
        "gb_max_depth": 3,
        "gb_learning_rate": 0.05,
        "gb_subsample": 0.8,
        "random_state": 42,
    }

    def __init__(
        self,
        params: Optional[dict] = None,
        feature_columns: Optional[list] = None,
        mode: Optional[Literal["isotonic", "monotone_boost", "ridge"]] = None,
    ):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        if mode is not None:
            self.params["mode"] = mode
        self.feature_columns = feature_columns or list(self.PROXY_FEATURES)
        self._model = None
        self._scaler: Optional[StandardScaler] = None
        self._ridge_projector: Optional[Ridge] = None
        self._isotonic: Optional[IsotonicRegression] = None

    def _prepare_X(self, X: pd.DataFrame) -> np.ndarray:
        """Select and validate proxy columns."""
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Missing proxy feature columns: {missing}")
        return X[self.feature_columns].values.astype(np.float64)

    def _fit_isotonic(self, X: np.ndarray, u: np.ndarray) -> None:
        """Fit isotonic pipeline: scale -> ridge score -> isotonic."""
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Project multi-dimensional proxies to a 1-D reliability score
        self._ridge_projector = Ridge(alpha=self.params["ridge_alpha"])
        self._ridge_projector.fit(X_scaled, u)
        scores = self._ridge_projector.predict(X_scaled)

        # Isotonic regression: enforce that predicted sigma2 is monotone
        # increasing with the projected score (higher score = higher noise)
        self._isotonic = IsotonicRegression(
            y_min=0.0,
            out_of_bounds="clip",
            increasing=True,
        )
        self._isotonic.fit(scores, u)

    def _predict_isotonic(self, X: np.ndarray) -> np.ndarray:
        """Predict via isotonic pipeline."""
        X_scaled = self._scaler.transform(X)
        scores = self._ridge_projector.predict(X_scaled)
        return self._isotonic.predict(scores)

    def _fit_monotone_boost(self, X: np.ndarray, u: np.ndarray) -> None:
        """Fit GradientBoosting with monotone constraints.

        Convention: higher proxy values = higher quality = LOWER noise.
        Therefore monotone_constraints = -1 for each proxy feature.
        """
        n_features = X.shape[1]
        # All proxy features are quality indicators: more coverage / higher
        # alignment / more audio completeness => less noise, so constrain
        # each to be monotone decreasing w.r.t. sigma2.
        monotone_constraints = tuple([-1] * n_features)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = GradientBoostingRegressor(
            n_estimators=self.params["gb_n_estimators"],
            max_depth=self.params["gb_max_depth"],
            learning_rate=self.params["gb_learning_rate"],
            subsample=self.params["gb_subsample"],
            random_state=self.params["random_state"],
            loss="squared_error",
            monotonic_cst=monotone_constraints,
        )
        self._model.fit(X_scaled, u)

    def _predict_monotone_boost(self, X: np.ndarray) -> np.ndarray:
        """Predict via monotone boosting."""
        X_scaled = self._scaler.transform(X)
        preds = self._model.predict(X_scaled)
        return np.maximum(preds, 0.0)

    def _fit_ridge(self, X: np.ndarray, u: np.ndarray) -> None:
        """Fit plain Ridge (unconstrained baseline for ablation)."""
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._model = Ridge(alpha=self.params["ridge_alpha"])
        self._model.fit(X_scaled, u)

    def _predict_ridge(self, X: np.ndarray) -> np.ndarray:
        """Predict via plain Ridge."""
        X_scaled = self._scaler.transform(X)
        preds = self._model.predict(X_scaled)
        return np.maximum(preds, 0.0)

    def fit(self, X_proxy: pd.DataFrame, u: np.ndarray) -> "ProxyNoiseModel":
        """Fit the proxy noise model.

        Parameters
        ----------
        X_proxy : pd.DataFrame
            Must contain proxy feature columns.
        u : array-like
            Squared ECC error: u = (r_tilde - z)^2.

        Returns
        -------
        self
        """
        X = self._prepare_X(X_proxy)
        u = np.asarray(u, dtype=np.float64)

        mode = self.params["mode"]
        if mode == "isotonic":
            self._fit_isotonic(X, u)
        elif mode == "monotone_boost":
            self._fit_monotone_boost(X, u)
        elif mode == "ridge":
            self._fit_ridge(X, u)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return self

    def predict(self, X_proxy: pd.DataFrame) -> np.ndarray:
        """Predict estimated error variance sigma^2.

        Parameters
        ----------
        X_proxy : pd.DataFrame
            Must contain proxy feature columns.

        Returns
        -------
        np.ndarray
            Estimated sigma^2 values (non-negative).
        """
        if self._scaler is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._prepare_X(X_proxy)

        mode = self.params["mode"]
        if mode == "isotonic":
            return self._predict_isotonic(X)
        elif mode == "monotone_boost":
            return self._predict_monotone_boost(X)
        elif mode == "ridge":
            return self._predict_ridge(X)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def save(self, path: str) -> None:
        """Save model to disk.

        Parameters
        ----------
        path : str
            File path (e.g. outputs/models/proxy_noise_isotonic.pkl).
        """
        if self._scaler is None:
            raise RuntimeError("Model not fitted. Nothing to save.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "params": self.params,
            "feature_columns": self.feature_columns,
            "scaler": self._scaler,
            "model": self._model,
            "ridge_projector": self._ridge_projector,
            "isotonic": self._isotonic,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> "ProxyNoiseModel":
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
        self.params = state["params"]
        self.feature_columns = state["feature_columns"]
        self._scaler = state["scaler"]
        self._model = state["model"]
        self._ridge_projector = state["ridge_projector"]
        self._isotonic = state["isotonic"]
        return self
