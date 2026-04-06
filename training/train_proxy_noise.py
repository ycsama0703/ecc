"""
Train Proxy Noise Model

Responsibilities (per code_structure.md 5.3):
- read residual target and z from the ECC residual stage
- compute u = (r_tilde - z)^2
- fit proxy-to-noise model
- save sigma2 predictions for train/val/test

Output files:
- outputs/models/proxy_noise_<mode>_<split_version>_<run_id>.pkl
- outputs/predictions/proxy_sigma2_train_<split_version>_<run_id>.csv
- outputs/predictions/proxy_sigma2_val_<split_version>_<run_id>.csv
- outputs/predictions/proxy_sigma2_test_<split_version>_<run_id>.csv
- outputs/logs/train_proxy_noise_<run_id>.json
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

from models.proxy_noise_model import ProxyNoiseModel

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


def load_ecc_residual_preds(pred_path: str) -> pd.DataFrame:
    """Load ECC residual stage predictions.

    Expected columns: event_id, shock_minus_pre, mu_hat, r_tilde, z_hat
    """
    df = pd.read_csv(pred_path)
    for col in (EVENT_ID_COLUMN, "r_tilde", "z_hat"):
        if col not in df.columns:
            raise ValueError(
                f"ECC residual prediction file missing column: {col}"
            )
    logger.info(
        "Loaded ECC residual predictions (%d rows) from %s", len(df), pred_path
    )
    return df


def attach_ecc_and_compute_u(
    df: pd.DataFrame, ecc_pred_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge ECC residual predictions onto a split DataFrame and compute u.

    u = (r_tilde - z_hat)^2  (squared ECC prediction error)
    """
    # Carry forward all upstream columns from ECC stage
    merge_cols = [EVENT_ID_COLUMN, "mu_hat", "r_tilde", "z_hat"]
    merge_cols = [c for c in merge_cols if c in ecc_pred_df.columns]

    merged = df.merge(
        ecc_pred_df[merge_cols],
        on=EVENT_ID_COLUMN,
        how="inner",
    )
    merged["u"] = (merged["r_tilde"] - merged["z_hat"]) ** 2

    if len(merged) < len(df):
        logger.warning(
            "ECC pred merge dropped %d rows (inner join)", len(df) - len(merged)
        )
    return merged


def save_predictions(
    df: pd.DataFrame,
    sigma2: np.ndarray,
    path: str,
) -> None:
    """Save event_id, upstream values, u, and sigma2 to CSV."""
    out = pd.DataFrame(
        {
            EVENT_ID_COLUMN: df[EVENT_ID_COLUMN].values,
            TARGET_COLUMN: df[TARGET_COLUMN].values,
            "mu_hat": df["mu_hat"].values,
            "r_tilde": df["r_tilde"].values,
            "z_hat": df["z_hat"].values,
            "u": df["u"].values,
            "sigma2": sigma2,
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
    model: ProxyNoiseModel,
    split_version: str,
    train_size: int,
    val_size: int,
    test_size: int,
    u_train_stats: dict,
) -> str:
    """Save run metadata to JSON log per logging requirements."""
    os.makedirs(log_dir, exist_ok=True)
    log_entry = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "model_name": "proxy_noise",
        "mode": model.params.get("mode"),
        "split_version": split_version,
        "random_seed": model.params.get("random_state"),
        "feature_columns": model.feature_columns,
        "model_hyperparameters": model.params,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "u_train_stats": u_train_stats,
        "config_snapshot": config,
    }
    log_path = os.path.join(log_dir, f"train_proxy_noise_{run_id}.json")
    with open(log_path, "w") as f:
        json.dump(log_entry, f, indent=2, default=str)
    logger.info("Saved run log to %s", log_path)
    return log_path


def train_proxy_noise(
    panel_path: str,
    split_path: str,
    ecc_pred_train_path: str,
    ecc_pred_val_path: str,
    ecc_pred_test_path: str,
    output_dir: str = "outputs",
    split_version: str = "v1",
    run_id: str = "run01",
    mode: str = "isotonic",
    params: dict | None = None,
) -> ProxyNoiseModel:
    """End-to-end training of the proxy noise model.

    Parameters
    ----------
    panel_path : str
        Path to the processed event-level panel (needed for proxy features).
    split_path : str
        Path to the split definition file.
    ecc_pred_train_path : str
        Path to ECC residual predictions for training split.
    ecc_pred_val_path : str
        Path to ECC residual predictions for validation split.
    ecc_pred_test_path : str
        Path to ECC residual predictions for test split.
    output_dir : str
        Root output directory.
    split_version : str
        Split version tag for naming.
    run_id : str
        Run identifier for naming.
    mode : str
        Proxy noise model mode: "isotonic", "monotone_boost", or "ridge".
    params : dict, optional
        Override default ProxyNoiseModel hyperparameters.

    Returns
    -------
    ProxyNoiseModel
        The fitted model.
    """
    config = {
        "panel_path": panel_path,
        "split_path": split_path,
        "ecc_pred_train_path": ecc_pred_train_path,
        "ecc_pred_val_path": ecc_pred_val_path,
        "ecc_pred_test_path": ecc_pred_test_path,
        "output_dir": output_dir,
        "split_version": split_version,
        "run_id": run_id,
        "mode": mode,
        "params": params,
    }
    logger.info("Starting proxy noise training  run_id=%s  mode=%s", run_id, mode)

    # --- Load data ---
    panel = load_panel(panel_path)
    split_df = load_split(split_path)
    train_df, val_df, test_df = split_data(panel, split_df)

    # --- Load ECC residual predictions and compute u ---
    ecc_train_df = load_ecc_residual_preds(ecc_pred_train_path)
    ecc_val_df = load_ecc_residual_preds(ecc_pred_val_path)
    ecc_test_df = load_ecc_residual_preds(ecc_pred_test_path)

    train_df = attach_ecc_and_compute_u(train_df, ecc_train_df)
    val_df = attach_ecc_and_compute_u(val_df, ecc_val_df)
    test_df = attach_ecc_and_compute_u(test_df, ecc_test_df)

    u_train = train_df["u"].values
    logger.info(
        "u stats (train)  mean=%.6f  median=%.6f  std=%.6f  min=%.6f  max=%.6f",
        np.mean(u_train),
        np.median(u_train),
        np.std(u_train),
        np.min(u_train),
        np.max(u_train),
    )

    # --- Fit model ---
    model = ProxyNoiseModel(params=params, mode=mode)
    model.fit(train_df, u_train)
    logger.info("Model fitted on %d training samples", len(train_df))

    # --- Predict sigma2 on all splits ---
    sigma2_train = model.predict(train_df)
    sigma2_val = model.predict(val_df)
    sigma2_test = model.predict(test_df)

    logger.info(
        "sigma2 stats (train)  mean=%.6f  median=%.6f",
        np.mean(sigma2_train),
        np.median(sigma2_train),
    )

    # --- Save model ---
    model_path = os.path.join(
        output_dir,
        "models",
        f"proxy_noise_{mode}_{split_version}_{run_id}.pkl",
    )
    model.save(model_path)

    # --- Save predictions ---
    pred_dir = os.path.join(output_dir, "predictions")
    save_predictions(
        train_df,
        sigma2_train,
        os.path.join(pred_dir, f"proxy_sigma2_train_{split_version}_{run_id}.csv"),
    )
    save_predictions(
        val_df,
        sigma2_val,
        os.path.join(pred_dir, f"proxy_sigma2_val_{split_version}_{run_id}.csv"),
    )
    save_predictions(
        test_df,
        sigma2_test,
        os.path.join(pred_dir, f"proxy_sigma2_test_{split_version}_{run_id}.csv"),
    )

    # --- Save run log ---
    u_train_stats = {
        "mean": float(np.mean(u_train)),
        "median": float(np.median(u_train)),
        "std": float(np.std(u_train)),
        "min": float(np.min(u_train)),
        "max": float(np.max(u_train)),
    }
    save_run_log(
        log_dir=os.path.join(output_dir, "logs"),
        run_id=run_id,
        config=config,
        model=model,
        split_version=split_version,
        train_size=len(train_df),
        val_size=len(val_df),
        test_size=len(test_df),
        u_train_stats=u_train_stats,
    )

    logger.info("Proxy noise training complete  run_id=%s", run_id)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the proxy noise model.")
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
        "--ecc-pred-train",
        type=str,
        required=True,
        help="Path to ECC residual train predictions CSV.",
    )
    parser.add_argument(
        "--ecc-pred-val",
        type=str,
        required=True,
        help="Path to ECC residual val predictions CSV.",
    )
    parser.add_argument(
        "--ecc-pred-test",
        type=str,
        required=True,
        help="Path to ECC residual test predictions CSV.",
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
        "--mode",
        type=str,
        default="isotonic",
        choices=["isotonic", "monotone_boost", "ridge"],
        help="Proxy noise model mode.",
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

    train_proxy_noise(
        panel_path=args.panel,
        split_path=args.split,
        ecc_pred_train_path=args.ecc_pred_train,
        ecc_pred_val_path=args.ecc_pred_val,
        ecc_pred_test_path=args.ecc_pred_test,
        output_dir=args.output_dir,
        split_version=args.split_version,
        run_id=args.run_id,
        mode=args.mode,
        params=params if params else None,
    )


if __name__ == "__main__":
    main()
