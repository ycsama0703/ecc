"""
Train Market Prior Model

Responsibilities (per code_structure.md 5.1):
- load train/val/test split
- fit market prior model
- save model
- save predictions for train/val/test

Output files:
- outputs/models/market_prior_xgb.pkl
- outputs/predictions/market_prior_train.csv
- outputs/predictions/market_prior_val.csv
- outputs/predictions/market_prior_test.csv
- outputs/logs/train_market_prior_<run_id>.json
"""

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.market_prior_model import MarketPriorModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_COLUMN = "shock_minus_pre"
EVENT_ID_COLUMN = "event_id"


def load_panel(panel_path: str) -> pd.DataFrame:
    """Load the processed event-level panel."""
    if panel_path.endswith(".parquet"):
        df = pd.read_parquet(panel_path)
    elif panel_path.endswith(".csv"):
        df = pd.read_csv(panel_path)
    else:
        raise ValueError(f"Unsupported file format: {panel_path}")
    logger.info("Loaded panel with %d rows, %d columns", len(df), len(df.columns))
    return df


def load_split(split_path: str) -> pd.DataFrame:
    """Load the split definition file with train_flag / val_flag / test_flag."""
    if split_path.endswith(".parquet"):
        df = pd.read_parquet(split_path)
    elif split_path.endswith(".csv"):
        df = pd.read_csv(split_path)
    else:
        raise ValueError(f"Unsupported file format: {split_path}")
    for col in ("train_flag", "val_flag", "test_flag"):
        if col not in df.columns:
            raise ValueError(f"Split file missing required column: {col}")
    logger.info("Loaded split with %d rows", len(df))
    return df


def split_data(
    panel: pd.DataFrame, split_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge panel with split flags and return train/val/test DataFrames."""
    if EVENT_ID_COLUMN in split_df.columns and EVENT_ID_COLUMN in panel.columns:
        merged = panel.merge(
            split_df[[EVENT_ID_COLUMN, "train_flag", "val_flag", "test_flag"]],
            on=EVENT_ID_COLUMN,
            how="inner",
        )
    else:
        if len(split_df) != len(panel):
            raise ValueError(
                "Split file has no event_id and row counts do not match panel."
            )
        merged = panel.copy()
        merged["train_flag"] = split_df["train_flag"].values
        merged["val_flag"] = split_df["val_flag"].values
        merged["test_flag"] = split_df["test_flag"].values

    train = merged[merged["train_flag"] == 1].copy()
    val = merged[merged["val_flag"] == 1].copy()
    test = merged[merged["test_flag"] == 1].copy()

    logger.info(
        "Split sizes  train=%d  val=%d  test=%d", len(train), len(val), len(test)
    )
    return train, val, test


def save_predictions(
    df: pd.DataFrame,
    mu_hat: np.ndarray,
    path: str,
) -> None:
    """Save event_id, true target, and mu_hat to CSV."""
    out = pd.DataFrame(
        {
            EVENT_ID_COLUMN: df[EVENT_ID_COLUMN].values,
            TARGET_COLUMN: df[TARGET_COLUMN].values,
            "mu_hat": mu_hat,
        }
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    logger.info("Saved predictions (%d rows) to %s", len(out), path)


def get_git_hash() -> str:
    """Return current git commit hash, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def save_run_log(
    log_dir: str,
    run_id: str,
    config: dict,
    model: MarketPriorModel,
    split_version: str,
    train_size: int,
    val_size: int,
    test_size: int,
) -> str:
    """Save run metadata to JSON log per logging requirements."""
    os.makedirs(log_dir, exist_ok=True)
    log_entry = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "model_name": "market_prior",
        "split_version": split_version,
        "random_seed": model.params.get("random_state"),
        "feature_columns": model.feature_columns,
        "model_hyperparameters": model.params,
        "best_params": model.best_params,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "config_snapshot": config,
    }
    log_path = os.path.join(log_dir, f"train_market_prior_{run_id}.json")
    with open(log_path, "w") as f:
        json.dump(log_entry, f, indent=2, default=str)
    logger.info("Saved run log to %s", log_path)
    return log_path


def train_market_prior(
    panel_path: str,
    split_path: str,
    output_dir: str = "outputs",
    split_version: str = "v1",
    run_id: str = "run01",
    tune: bool = False,
    params: dict | None = None,
) -> MarketPriorModel:
    """End-to-end training of the market prior model.

    Parameters
    ----------
    panel_path : str
        Path to the processed event-level panel.
    split_path : str
        Path to the split definition file.
    output_dir : str
        Root output directory.
    split_version : str
        Split version tag for naming.
    run_id : str
        Run identifier for naming.
    tune : bool
        Whether to run hyperparameter tuning.
    params : dict, optional
        Override default model hyperparameters.

    Returns
    -------
    MarketPriorModel
        The fitted model.
    """
    config = {
        "panel_path": panel_path,
        "split_path": split_path,
        "output_dir": output_dir,
        "split_version": split_version,
        "run_id": run_id,
        "tune": tune,
        "params": params,
    }
    logger.info("Starting market prior training  run_id=%s", run_id)

    # --- Load data ---
    panel = load_panel(panel_path)
    split_df = load_split(split_path)
    train_df, val_df, test_df = split_data(panel, split_df)

    # --- Fit model ---
    model = MarketPriorModel(params=params, tune=tune)
    model.fit(train_df, train_df[TARGET_COLUMN].values)
    logger.info("Model fitted on %d training samples", len(train_df))

    # --- Predict on all splits ---
    mu_hat_train = model.predict(train_df)
    mu_hat_val = model.predict(val_df)
    mu_hat_test = model.predict(test_df)

    # --- Save model ---
    model_path = os.path.join(
        output_dir, "models", f"market_prior_xgb_{split_version}_{run_id}.pkl"
    )
    model.save(model_path)

    # --- Save predictions ---
    pred_dir = os.path.join(output_dir, "predictions")
    save_predictions(
        train_df,
        mu_hat_train,
        os.path.join(pred_dir, f"market_prior_train_{split_version}_{run_id}.csv"),
    )
    save_predictions(
        val_df,
        mu_hat_val,
        os.path.join(pred_dir, f"market_prior_val_{split_version}_{run_id}.csv"),
    )
    save_predictions(
        test_df,
        mu_hat_test,
        os.path.join(pred_dir, f"market_prior_test_{split_version}_{run_id}.csv"),
    )

    # --- Save run log ---
    save_run_log(
        log_dir=os.path.join(output_dir, "logs"),
        run_id=run_id,
        config=config,
        model=model,
        split_version=split_version,
        train_size=len(train_df),
        val_size=len(val_df),
        test_size=len(test_df),
    )

    logger.info("Market prior training complete  run_id=%s", run_id)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the market prior model.")
    parser.add_argument(
        "--panel",
        type=str,
        required=True,
        help="Path to processed event-level panel (parquet or csv).",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Path to split definition file (parquet or csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Root output directory.",
    )
    parser.add_argument(
        "--split-version",
        type=str,
        default="v1",
        help="Split version tag for output naming.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="run01",
        help="Run identifier for output naming.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning via GridSearchCV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed.",
    )
    args = parser.parse_args()

    params = {}
    if args.seed is not None:
        params["random_state"] = args.seed

    train_market_prior(
        panel_path=args.panel,
        split_path=args.split,
        output_dir=args.output_dir,
        split_version=args.split_version,
        run_id=args.run_id,
        tune=args.tune,
        params=params if params else None,
    )


if __name__ == "__main__":
    main()
