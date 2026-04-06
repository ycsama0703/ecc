"""
Fit Full Pipeline

Orchestrates the complete training pipeline (per code_structure.md 5.4
and Section 3 Required Core Pipeline):

    1.  Load processed event-level panel
    2.  Build train/val/test split
    3.  Train market prior model
    4.  Compute residual target  r_tilde = y - mu_hat
    5.  Train ECC residual model
    6.  Compute ECC residual error  u = (r_tilde - z)^2
    7.  Train proxy noise model
    8.  Estimate signal variance  tau^2
    9.  Compute shrinkage gate  alpha
    10. Produce final prediction  y_hat
    11. Evaluate
    12. Save artifacts

Output files:
- outputs/predictions/final_main_test.csv
- outputs/metrics/main_metrics.json
- (plus all intermediate artifacts from each stage)
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
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.abstention import apply_abstention, select_threshold
from models.minimax_gate import compute_gate, estimate_tau2
from training.train_ecc_residual import train_ecc_residual
from training.train_market_prior import train_market_prior
from training.train_proxy_noise import train_proxy_noise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_COLUMN = "shock_minus_pre"
EVENT_ID_COLUMN = "event_id"


# ---------------------------------------------------------------------------
# Evaluation helpers (per evaluation_protocol.md)
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute primary regression metrics on a given set.

    Metrics (per evaluation_protocol.md):
        - MSE
        - MAE
        - R^2 (out-of-sample, baseline = mean prediction)
        - Spearman correlation
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    residuals = y_true - y_pred
    mse = float(np.mean(residuals ** 2))
    mae = float(np.mean(np.abs(residuals)))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    spearman_corr, spearman_p = scipy_stats.spearmanr(y_true, y_pred)

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "spearman": float(spearman_corr),
        "spearman_p": float(spearman_p),
        "n": int(len(y_true)),
    }


def compute_selective_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    accept: np.ndarray,
) -> dict:
    """Compute selective prediction metrics (per evaluation_protocol.md).

    Metrics:
        - coverage
        - accepted-set MSE
    """
    n = len(y_true)
    n_accepted = int(np.sum(accept))
    coverage = n_accepted / n if n > 0 else 0.0

    if n_accepted > 0:
        accepted_mse = float(np.mean((y_true[accept] - y_pred[accept]) ** 2))
    else:
        accepted_mse = float("nan")

    return {
        "coverage": coverage,
        "n_accepted": n_accepted,
        "n_total": n,
        "accepted_mse": accepted_mse,
    }


# ---------------------------------------------------------------------------
# Data loading (duplicated minimally from stage modules for self-contained use)
# ---------------------------------------------------------------------------

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
    """Load the split definition file."""
    if split_path.endswith(".parquet"):
        df = pd.read_parquet(split_path)
    elif split_path.endswith(".csv"):
        df = pd.read_csv(split_path)
    else:
        raise ValueError(f"Unsupported file format: {split_path}")
    for col in ("train_flag", "val_flag", "test_flag"):
        if col not in df.columns:
            raise ValueError(f"Split file missing required column: {col}")
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
    return train, val, test


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


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def pred_csv(output_dir: str, name: str, sv: str, rid: str) -> str:
    """Build a prediction CSV path consistent with stage naming.

    Example: pred_csv("outputs", "market_prior_train", "v1", "run01")
             -> "outputs/predictions/market_prior_train_v1_run01.csv"
    """
    return os.path.join(
        output_dir, "predictions", f"{name}_{sv}_{rid}.csv"
    )


def fit_full_pipeline(
    panel_path: str,
    split_path: str,
    output_dir: str = "outputs",
    split_version: str = "v1",
    run_id: str = "run01",
    tune_market_prior: bool = False,
    market_prior_params: dict | None = None,
    ecc_residual_params: dict | None = None,
    proxy_noise_mode: str = "isotonic",
    proxy_noise_params: dict | None = None,
    use_abstention: bool = True,
    abstention_metric: str = "mse",
) -> dict:
    """Run the full pipeline end-to-end.

    Parameters
    ----------
    panel_path : str
        Path to the processed event-level panel.
    split_path : str
        Path to the split definition file.
    output_dir : str
        Root output directory.
    split_version : str
        Split version tag.
    run_id : str
        Run identifier.
    tune_market_prior : bool
        Whether to tune the market prior model.
    market_prior_params : dict, optional
        Override market prior hyperparameters.
    ecc_residual_params : dict, optional
        Override ECC residual model hyperparameters.
    proxy_noise_mode : str
        Proxy noise model mode.
    proxy_noise_params : dict, optional
        Override proxy noise model hyperparameters.
    use_abstention : bool
        Whether to apply abstention (E7) or just minimax gate (E6).
    abstention_metric : str
        Metric for threshold selection ("mse" or "mae").

    Returns
    -------
    dict
        Dictionary with all computed metrics and pipeline metadata.
    """
    sv, rid = split_version, run_id

    # ===================================================================
    # Step 1-2: Load panel and split (done inside each stage trainer,
    #           but we also load here for steps 8-12)
    # ===================================================================
    logger.info("=" * 60)
    logger.info("FULL PIPELINE START  run_id=%s", rid)
    logger.info("=" * 60)

    # ===================================================================
    # Step 3: Train market prior model
    # ===================================================================
    logger.info("Step 3: Training market prior model")
    train_market_prior(
        panel_path=panel_path,
        split_path=split_path,
        output_dir=output_dir,
        split_version=sv,
        run_id=rid,
        tune=tune_market_prior,
        params=market_prior_params,
    )

    # ===================================================================
    # Step 4-5: Compute residual target & train ECC residual model
    # ===================================================================
    logger.info("Steps 4-5: Training ECC residual model")
    train_ecc_residual(
        panel_path=panel_path,
        split_path=split_path,
        mu_hat_train_path=pred_csv(output_dir, "market_prior_train", sv, rid),
        mu_hat_val_path=pred_csv(output_dir, "market_prior_val", sv, rid),
        mu_hat_test_path=pred_csv(output_dir, "market_prior_test", sv, rid),
        output_dir=output_dir,
        split_version=sv,
        run_id=rid,
        params=ecc_residual_params,
    )

    # ===================================================================
    # Step 6-7: Compute ECC error & train proxy noise model
    # ===================================================================
    logger.info("Steps 6-7: Training proxy noise model")
    train_proxy_noise(
        panel_path=panel_path,
        split_path=split_path,
        ecc_pred_train_path=pred_csv(output_dir, "ecc_residual_train", sv, rid),
        ecc_pred_val_path=pred_csv(output_dir, "ecc_residual_val", sv, rid),
        ecc_pred_test_path=pred_csv(output_dir, "ecc_residual_test", sv, rid),
        output_dir=output_dir,
        split_version=sv,
        run_id=rid,
        mode=proxy_noise_mode,
        params=proxy_noise_params,
    )

    # ===================================================================
    # Steps 8-10: Estimate tau^2, compute gate, produce final predictions
    # ===================================================================
    logger.info("Steps 8-10: tau^2 estimation, minimax gate, final prediction")

    # Load proxy noise stage predictions (they carry all upstream columns)
    train_preds = pd.read_csv(
        os.path.join(output_dir, "predictions", f"proxy_sigma2_train_{sv}_{rid}.csv")
    )
    val_preds = pd.read_csv(
        os.path.join(output_dir, "predictions", f"proxy_sigma2_val_{sv}_{rid}.csv")
    )
    test_preds = pd.read_csv(
        os.path.join(output_dir, "predictions", f"proxy_sigma2_test_{sv}_{rid}.csv")
    )

    # Step 8: Estimate tau^2 from training data
    tau2 = estimate_tau2(
        r_tilde=train_preds["r_tilde"].values,
        sigma2=train_preds["sigma2"].values,
    )
    logger.info("Estimated tau^2 = %.6f", tau2)

    # Step 9: Compute per-observation shrinkage gate alpha
    alpha_train = compute_gate(tau2, train_preds["sigma2"].values)
    alpha_val = compute_gate(tau2, val_preds["sigma2"].values)
    alpha_test = compute_gate(tau2, test_preds["sigma2"].values)

    logger.info(
        "alpha stats (test)  mean=%.4f  min=%.4f  max=%.4f",
        np.mean(alpha_test),
        np.min(alpha_test),
        np.max(alpha_test),
    )

    # Step 10: Produce final prediction
    if use_abstention:
        # Select threshold on validation set
        threshold = select_threshold(
            sigma2_val=val_preds["sigma2"].values,
            y_val=val_preds[TARGET_COLUMN].values,
            mu_hat_val=val_preds["mu_hat"].values,
            z_hat_val=val_preds["z_hat"].values,
            alpha_val=alpha_val,
            metric=abstention_metric,
        )
        logger.info("Selected abstention threshold = %.6f", threshold)

        y_hat_test, accept_test = apply_abstention(
            mu_hat=test_preds["mu_hat"].values,
            z_hat=test_preds["z_hat"].values,
            alpha=alpha_test,
            sigma2=test_preds["sigma2"].values,
            threshold=threshold,
        )
        y_hat_val, accept_val = apply_abstention(
            mu_hat=val_preds["mu_hat"].values,
            z_hat=val_preds["z_hat"].values,
            alpha=alpha_val,
            sigma2=val_preds["sigma2"].values,
            threshold=threshold,
        )
    else:
        # E6: minimax gate only, no abstention
        threshold = None
        y_hat_test = test_preds["mu_hat"].values + alpha_test * test_preds["z_hat"].values
        y_hat_val = val_preds["mu_hat"].values + alpha_val * val_preds["z_hat"].values
        accept_test = np.ones(len(test_preds), dtype=bool)
        accept_val = np.ones(len(val_preds), dtype=bool)

    # ===================================================================
    # Step 11: Evaluate (per evaluation_protocol.md — test set only)
    # ===================================================================
    logger.info("Step 11: Evaluating on test set")

    y_true_test = test_preds[TARGET_COLUMN].values
    y_true_val = val_preds[TARGET_COLUMN].values

    test_metrics = compute_metrics(y_true_test, y_hat_test)
    val_metrics = compute_metrics(y_true_val, y_hat_val)

    logger.info(
        "TEST  MSE=%.6f  MAE=%.6f  R2=%.4f  Spearman=%.4f",
        test_metrics["mse"],
        test_metrics["mae"],
        test_metrics["r2"],
        test_metrics["spearman"],
    )

    selective_test = None
    selective_val = None
    if use_abstention:
        selective_test = compute_selective_metrics(y_true_test, y_hat_test, accept_test)
        selective_val = compute_selective_metrics(y_true_val, y_hat_val, accept_val)
        logger.info(
            "TEST selective  coverage=%.2f%%  accepted_MSE=%.6f",
            selective_test["coverage"] * 100,
            selective_test["accepted_mse"],
        )

    # ===================================================================
    # Step 12: Save artifacts
    # ===================================================================
    logger.info("Step 12: Saving artifacts")

    # --- Final test predictions ---
    final_test_df = pd.DataFrame(
        {
            EVENT_ID_COLUMN: test_preds[EVENT_ID_COLUMN].values,
            TARGET_COLUMN: y_true_test,
            "mu_hat": test_preds["mu_hat"].values,
            "z_hat": test_preds["z_hat"].values,
            "sigma2": test_preds["sigma2"].values,
            "alpha": alpha_test,
            "y_hat": y_hat_test,
            "accept": accept_test.astype(int),
        }
    )
    final_test_path = os.path.join(
        output_dir, "predictions", f"final_main_test_{sv}_{rid}.csv"
    )
    os.makedirs(os.path.dirname(final_test_path), exist_ok=True)
    final_test_df.to_csv(final_test_path, index=False)
    logger.info("Saved final test predictions to %s", final_test_path)

    # --- Main metrics JSON ---
    results = {
        "run_id": rid,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "split_version": sv,
        "use_abstention": use_abstention,
        "proxy_noise_mode": proxy_noise_mode,
        "tau2": tau2,
        "abstention_threshold": threshold,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "selective_test_metrics": selective_test,
        "selective_val_metrics": selective_val,
        "config": {
            "panel_path": panel_path,
            "split_path": split_path,
            "output_dir": output_dir,
            "split_version": sv,
            "run_id": rid,
            "tune_market_prior": tune_market_prior,
            "market_prior_params": market_prior_params,
            "ecc_residual_params": ecc_residual_params,
            "proxy_noise_mode": proxy_noise_mode,
            "proxy_noise_params": proxy_noise_params,
            "use_abstention": use_abstention,
            "abstention_metric": abstention_metric,
        },
    }

    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"main_metrics_{sv}_{rid}.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved main metrics to %s", metrics_path)

    logger.info("=" * 60)
    logger.info("FULL PIPELINE COMPLETE  run_id=%s", rid)
    logger.info("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the full pipeline end-to-end."
    )
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
        help="Split version tag.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="run01",
        help="Run identifier.",
    )
    parser.add_argument(
        "--tune-market-prior",
        action="store_true",
        help="Enable market prior hyperparameter tuning.",
    )
    parser.add_argument(
        "--proxy-noise-mode",
        type=str,
        default="isotonic",
        choices=["isotonic", "monotone_boost", "ridge"],
        help="Proxy noise model mode.",
    )
    parser.add_argument(
        "--no-abstention",
        action="store_true",
        help="Disable abstention (run E6 instead of E7).",
    )
    parser.add_argument(
        "--abstention-metric",
        type=str,
        default="mse",
        choices=["mse", "mae"],
        help="Metric for abstention threshold selection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for all stages.",
    )
    args = parser.parse_args()

    market_prior_params = None
    ecc_residual_params = None
    proxy_noise_params = None
    if args.seed is not None:
        market_prior_params = {"random_state": args.seed}
        ecc_residual_params = {"random_state": args.seed}
        proxy_noise_params = {"random_state": args.seed}

    fit_full_pipeline(
        panel_path=args.panel,
        split_path=args.split,
        output_dir=args.output_dir,
        split_version=args.split_version,
        run_id=args.run_id,
        tune_market_prior=args.tune_market_prior,
        market_prior_params=market_prior_params,
        ecc_residual_params=ecc_residual_params,
        proxy_noise_mode=args.proxy_noise_mode,
        proxy_noise_params=proxy_noise_params,
        use_abstention=not args.no_abstention,
        abstention_metric=args.abstention_metric,
    )


if __name__ == "__main__":
    main()
