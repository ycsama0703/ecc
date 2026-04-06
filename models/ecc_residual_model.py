"""
ECC Residual Model

Trains a residual predictor using ECC features to estimate z ≈ r.

Core equation context:
    r_tilde = y - mu_hat          (residual target)
    z       = ECCResidualModel.predict(X_ecc)   (predicted residual)

Input ECC features (per data_contract.md):
    - text_embedding
    - qa_embedding
    - audio_features
"""

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ECCResidualModel:
    """ECC-based residual predictor.

    Uses a Ridge regression pipeline with optional PCA dimensionality
    reduction, appropriate for high-dimensional embedding inputs where
    regularisation is essential to avoid overfitting noisy residual targets.
    """

    ECC_FEATURE_GROUPS = [
        "text_embedding",
        "qa_embedding",
        "audio_features",
    ]

    DEFAULT_PARAMS = {
        "pca_components": 50,
        "ridge_alpha": 1.0,
        "random_state": 42,
    }

    def __init__(
        self,
        params: Optional[dict] = None,
        feature_columns: Optional[list] = None,
    ):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.feature_columns = feature_columns
        self.pipeline: Optional[Pipeline] = None
        self._fitted_feature_columns: Optional[list] = None

    def _resolve_ecc_columns(self, X: pd.DataFrame) -> list[str]:
        """Resolve ECC feature columns from the DataFrame.

        If feature_columns was provided explicitly, use those.
        Otherwise, detect columns that start with the known ECC
        feature group prefixes (handles the common case where
        embeddings are stored as text_embedding_0, text_embedding_1, ...).
        """
        if self.feature_columns is not None:
            return self.feature_columns

        cols = []
        for group in self.ECC_FEATURE_GROUPS:
            matched = [c for c in X.columns if c.startswith(group)]
            if matched:
                cols.extend(sorted(matched))
            elif group in X.columns:
                cols.append(group)
        if not cols:
            raise ValueError(
                "No ECC feature columns found. Provide feature_columns explicitly "
                "or ensure columns start with one of: "
                + ", ".join(self.ECC_FEATURE_GROUPS)
            )
        return cols

    def _prepare_X(self, X: pd.DataFrame) -> np.ndarray:
        """Select ECC columns and return as numpy array."""
        cols = (
            self._fitted_feature_columns
            if self._fitted_feature_columns is not None
            else self._resolve_ecc_columns(X)
        )
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"Missing ECC feature columns: {missing}")
        return X[cols].values.astype(np.float64)

    def _build_pipeline(self, n_features: int) -> Pipeline:
        """Construct the sklearn pipeline."""
        pca_components = self.params["pca_components"]
        if pca_components is not None and pca_components < n_features:
            steps = [
                ("scaler", StandardScaler()),
                (
                    "pca",
                    PCA(
                        n_components=pca_components,
                        random_state=self.params["random_state"],
                    ),
                ),
                ("ridge", Ridge(alpha=self.params["ridge_alpha"])),
            ]
        else:
            steps = [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.params["ridge_alpha"])),
            ]
        return Pipeline(steps)

    def fit(self, X_ecc: pd.DataFrame, residual: np.ndarray) -> "ECCResidualModel":
        """Fit the ECC residual model.

        Parameters
        ----------
        X_ecc : pd.DataFrame
            Must contain ECC feature columns (text_embedding_*,
            qa_embedding_*, audio_features_*).
        residual : array-like
            Residual target r_tilde = y - mu_hat.

        Returns
        -------
        self
        """
        self._fitted_feature_columns = self._resolve_ecc_columns(X_ecc)
        X = self._prepare_X(X_ecc)
        residual = np.asarray(residual, dtype=np.float64)

        self.pipeline = self._build_pipeline(n_features=X.shape[1])
        self.pipeline.fit(X, residual)
        return self

    def predict(self, X_ecc: pd.DataFrame) -> np.ndarray:
        """Predict z (estimated residual) for given ECC features.

        Parameters
        ----------
        X_ecc : pd.DataFrame
            Must contain the same ECC feature columns used during fit.

        Returns
        -------
        np.ndarray
            Predicted z values.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._prepare_X(X_ecc)
        return self.pipeline.predict(X)

    def save(self, path: str) -> None:
        """Save model to disk.

        Parameters
        ----------
        path : str
            File path (e.g. outputs/models/ecc_residual_ridge.pkl).
        """
        if self.pipeline is None:
            raise RuntimeError("Model not fitted. Nothing to save.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "pipeline": self.pipeline,
            "params": self.params,
            "feature_columns": self._fitted_feature_columns,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> "ECCResidualModel":
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
        self.pipeline = state["pipeline"]
        self.params = state["params"]
        self._fitted_feature_columns = state["feature_columns"]
        self.feature_columns = state["feature_columns"]
        return self
