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

from models.abstention import (
    apply_abstention_from_risk,
    build_risk_coverage_curve,
    compute_aurc,
    compute_closed_form_risk,
    select_risk_threshold,
    select_threshold_for_target_coverage,
)
from models.minimax_gate import (
    build_sigma_bar,
    compute_gate,
    compute_gate_upper,
    estimate_conservative_margin,
    estimate_tau2,
)
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
    aurc: float | None = None,
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
        accepted_metrics = compute_metrics(y_true[accept], y_pred[accept])
        accepted_mse = accepted_metrics["mse"]
        accepted_mae = accepted_metrics["mae"]
        accepted_r2 = accepted_metrics["r2"]
        accepted_spearman = accepted_metrics["spearman"]
        accepted_spearman_p = accepted_metrics["spearman_p"]
    else:
        accepted_mse = float("nan")
        accepted_mae = float("nan")
        accepted_r2 = float("nan")
        accepted_spearman = float("nan")
        accepted_spearman_p = float("nan")

    return {
        "coverage": coverage,
        "n_accepted": n_accepted,
        "n_total": n,
        "accepted_mse": accepted_mse,
        "accepted_mae": accepted_mae,
        "accepted_r2": accepted_r2,
        "accepted_spearman": accepted_spearman,
        "accepted_spearman_p": accepted_spearman_p,
        "aurc": aurc,
    }


def parse_target_coverages(target_coverages: str) -> list[float | None]:
    parsed: list[float | None] = []
    for raw_item in target_coverages.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item in {"unconstrained", "full", "auto"}:
            parsed.append(None)
        else:
            value = float(item)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"Invalid target coverage: {raw_item}")
            parsed.append(value)
    if not parsed:
        raise ValueError("At least one target coverage must be provided.")
    return parsed


def build_family_row(
    model: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    accept: np.ndarray,
    prior_pred: np.ndarray,
    aurc: float | None = None,
) -> dict:
    full_metrics = compute_metrics(y_true, y_pred)
    selective_metrics = compute_selective_metrics(y_true, y_pred, accept, aurc=aurc)
    prior_metrics = compute_metrics(y_true, prior_pred)
    return {
        "model": model,
        "split": split_name,
        "mse": full_metrics["mse"],
        "mae": full_metrics["mae"],
        "r2": full_metrics["r2"],
        "spearman": full_metrics["spearman"],
        "spearman_p": full_metrics["spearman_p"],
        "coverage": selective_metrics["coverage"],
        "accepted_mse": selective_metrics["accepted_mse"],
        "accepted_mae": selective_metrics["accepted_mae"],
        "accepted_r2": selective_metrics["accepted_r2"],
        "accepted_spearman": selective_metrics["accepted_spearman"],
        "aurc": aurc,
        "gain_over_prior_mse": prior_metrics["mse"] - full_metrics["mse"],
        "n": full_metrics["n"],
        "n_accepted": selective_metrics["n_accepted"],
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


def save_dataframe(df: pd.DataFrame, path: str) -> str:
    """Persist a DataFrame to CSV and return its path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


def fit_full_pipeline(
    panel_path: str,
    split_path: str,
    output_dir: str = "outputs",
    split_version: str = "v1",
    run_id: str = "run01",
    tune_market_prior: bool = False,
    market_prior_params: dict | None = None,
    ecc_residual_params: dict | None = None,
    ecc_feature_mode: str = "ecc_only",
    proxy_noise_mode: str = "isotonic",
    proxy_noise_params: dict | None = None,
    use_abstention: bool = True,
    abstention_metric: str = "mse",
    abstention_min_coverage: float = 0.30,
    tau2_method: str = "variance",
    sigma_bar_quantile: float = 0.90,
    sigma_bar_min_margin: float = 0.0,
    abstention_target_coverage: float | None = None,
    coverage_sweep_targets: str = "0.2,0.4,0.6,0.8,unconstrained",
    target_mode: str = "shock_minus_pre",
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
    ecc_feature_mode : str
        Feature bundle for the residual model.
    proxy_noise_mode : str
        Proxy noise model mode.
    proxy_noise_params : dict, optional
        Override proxy noise model hyperparameters.
    use_abstention : bool
        Whether to apply abstention (E7) or just minimax gate (E6).
    abstention_metric : str
        Metric for threshold selection ("mse" or "mae").
    abstention_min_coverage : float
        Minimum validation coverage required when selecting kappa.
    tau2_method : str
        Signal variance estimator: "variance" or "moments".
    sigma_bar_quantile : float
        Quantile used to build the conservative sigma_bar^2 margin.
    sigma_bar_min_margin : float
        Lower bound for the conservative sigma_bar^2 margin.
    abstention_target_coverage : float, optional
        If provided, select kappa to match a target validation coverage.
    coverage_sweep_targets : str
        Comma-separated target coverages for selective summary export.

    Returns
    -------
    dict
        Dictionary with all computed metrics and pipeline metadata.
    """
    sv, rid = split_version, run_id

    if ecc_feature_mode != "ecc_only":
        raise ValueError(
            "Main PREC experiments now require `ecc_feature_mode='ecc_only'`. "
            "Hybrid `ecc_plus_market_controls` is treated as a contaminated appendix-only run."
        )

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
        target_mode=target_mode,
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
        feature_mode=ecc_feature_mode,
        target_mode=target_mode,
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
        target_mode=target_mode,
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
        method=tau2_method,
    )
    logger.info("Estimated tau^2 = %.6f", tau2)

    # Step 8.5: Build conservative sigma_bar^2 via a validation quantile margin
    sigma_bar_delta = estimate_conservative_margin(
        u=val_preds["u"].values,
        sigma2_hat=val_preds["sigma2"].values,
        quantile=sigma_bar_quantile,
        min_margin=sigma_bar_min_margin,
    )
    sigma_bar_train = build_sigma_bar(train_preds["sigma2"].values, sigma_bar_delta)
    sigma_bar_val = build_sigma_bar(val_preds["sigma2"].values, sigma_bar_delta)
    sigma_bar_test = build_sigma_bar(test_preds["sigma2"].values, sigma_bar_delta)
    logger.info("Conservative sigma_bar^2 margin delta = %.6f", sigma_bar_delta)

    # Step 9: Compute observed and minimax shrinkage gates
    alpha_obs_train = compute_gate(tau2, train_preds["sigma2"].values)
    alpha_obs_val = compute_gate(tau2, val_preds["sigma2"].values)
    alpha_obs_test = compute_gate(tau2, test_preds["sigma2"].values)

    alpha_train = compute_gate_upper(tau2, sigma_bar_train)
    alpha_val = compute_gate_upper(tau2, sigma_bar_val)
    alpha_test = compute_gate_upper(tau2, sigma_bar_test)

    logger.info(
        "alpha_mm stats (test)  mean=%.4f  min=%.4f  max=%.4f",
        np.mean(alpha_test),
        np.min(alpha_test),
        np.max(alpha_test),
    )

    risk_train = compute_closed_form_risk(tau2, sigma_bar_train)
    risk_val = compute_closed_form_risk(tau2, sigma_bar_val)
    risk_test = compute_closed_form_risk(tau2, sigma_bar_test)

    # Step 10: Produce final prediction
    if use_abstention:
        # Select threshold on validation risk scores
        threshold, risk_curve = select_risk_threshold(
            risk_val=risk_val,
            y_val=val_preds[TARGET_COLUMN].values,
            mu_hat_val=val_preds["mu_hat"].values,
            z_hat_val=val_preds["z_hat"].values,
            alpha_val=alpha_val,
            metric=abstention_metric,
            min_coverage=abstention_min_coverage,
        )
        selected_threshold_mode = "min_metric_subject_to_min_coverage"
        if abstention_target_coverage is not None:
            threshold, selected_curve_row = select_threshold_for_target_coverage(
                curve=risk_curve,
                target_coverage=abstention_target_coverage,
            )
            selected_threshold_mode = "target_coverage"
            logger.info(
                "Selected risk threshold by target coverage %.2f -> realized val coverage %.4f",
                abstention_target_coverage,
                float(selected_curve_row["coverage"]),
            )
        logger.info("Selected risk threshold kappa = %.6f", threshold)

        test_risk_curve = build_risk_coverage_curve(
            risk=risk_test,
            y_true=test_preds[TARGET_COLUMN].values,
            mu_hat=test_preds["mu_hat"].values,
            z_hat=test_preds["z_hat"].values,
            alpha=alpha_test,
        )
        val_aurc = compute_aurc(risk_curve, metric=abstention_metric)
        test_aurc = compute_aurc(test_risk_curve, metric=abstention_metric)

        y_hat_train, accept_train = apply_abstention_from_risk(
            mu_hat=train_preds["mu_hat"].values,
            z_hat=train_preds["z_hat"].values,
            alpha=alpha_train,
            risk=risk_train,
            threshold=threshold,
        )
        y_hat_test, accept_test = apply_abstention_from_risk(
            mu_hat=test_preds["mu_hat"].values,
            z_hat=test_preds["z_hat"].values,
            alpha=alpha_test,
            risk=risk_test,
            threshold=threshold,
        )
        y_hat_val, accept_val = apply_abstention_from_risk(
            mu_hat=val_preds["mu_hat"].values,
            z_hat=val_preds["z_hat"].values,
            alpha=alpha_val,
            risk=risk_val,
            threshold=threshold,
        )
    else:
        # E6: minimax gate only, no abstention
        threshold = None
        risk_curve = []
        test_risk_curve = []
        val_aurc = None
        test_aurc = None
        y_hat_train = (
            train_preds["mu_hat"].values + alpha_train * train_preds["z_hat"].values
        )
        y_hat_test = test_preds["mu_hat"].values + alpha_test * test_preds["z_hat"].values
        y_hat_val = val_preds["mu_hat"].values + alpha_val * val_preds["z_hat"].values
        accept_train = np.ones(len(train_preds), dtype=bool)
        accept_test = np.ones(len(test_preds), dtype=bool)
        accept_val = np.ones(len(val_preds), dtype=bool)

    # Variant predictions for benchmark-family comparison
    y_hat_prior_train = train_preds["mu_hat"].values
    y_hat_prior_val = val_preds["mu_hat"].values
    y_hat_prior_test = test_preds["mu_hat"].values

    y_hat_plus_z_train = train_preds["mu_hat"].values + train_preds["z_hat"].values
    y_hat_plus_z_val = val_preds["mu_hat"].values + val_preds["z_hat"].values
    y_hat_plus_z_test = test_preds["mu_hat"].values + test_preds["z_hat"].values

    y_hat_plus_alpha_train = train_preds["mu_hat"].values + alpha_train * train_preds["z_hat"].values
    y_hat_plus_alpha_val = val_preds["mu_hat"].values + alpha_val * val_preds["z_hat"].values
    y_hat_plus_alpha_test = test_preds["mu_hat"].values + alpha_test * test_preds["z_hat"].values

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
        selective_test = compute_selective_metrics(
            y_true_test,
            y_hat_test,
            accept_test,
            aurc=test_aurc,
        )
        selective_val = compute_selective_metrics(
            y_true_val,
            y_hat_val,
            accept_val,
            aurc=val_aurc,
        )
        logger.info(
            "TEST selective  coverage=%.2f%%  accepted_MSE=%.6f  AURC=%.6f",
            selective_test["coverage"] * 100,
            selective_test["accepted_mse"],
            selective_test["aurc"],
        )

    family_rows = [
        build_family_row(
            model="prior_only",
            split_name="val",
            y_true=y_true_val,
            y_pred=y_hat_prior_val,
            accept=np.ones(len(y_true_val), dtype=bool),
            prior_pred=y_hat_prior_val,
            aurc=None,
        ),
        build_family_row(
            model="prior_only",
            split_name="test",
            y_true=y_true_test,
            y_pred=y_hat_prior_test,
            accept=np.ones(len(y_true_test), dtype=bool),
            prior_pred=y_hat_prior_test,
            aurc=None,
        ),
        build_family_row(
            model="prior_plus_z_no_gate",
            split_name="val",
            y_true=y_true_val,
            y_pred=y_hat_plus_z_val,
            accept=np.ones(len(y_true_val), dtype=bool),
            prior_pred=y_hat_prior_val,
            aurc=None,
        ),
        build_family_row(
            model="prior_plus_z_no_gate",
            split_name="test",
            y_true=y_true_test,
            y_pred=y_hat_plus_z_test,
            accept=np.ones(len(y_true_test), dtype=bool),
            prior_pred=y_hat_prior_test,
            aurc=None,
        ),
        build_family_row(
            model="prior_plus_alpha_z_gate_only",
            split_name="val",
            y_true=y_true_val,
            y_pred=y_hat_plus_alpha_val,
            accept=np.ones(len(y_true_val), dtype=bool),
            prior_pred=y_hat_prior_val,
            aurc=None,
        ),
        build_family_row(
            model="prior_plus_alpha_z_gate_only",
            split_name="test",
            y_true=y_true_test,
            y_pred=y_hat_plus_alpha_test,
            accept=np.ones(len(y_true_test), dtype=bool),
            prior_pred=y_hat_prior_test,
            aurc=None,
        ),
        build_family_row(
            model="prec_selective",
            split_name="val",
            y_true=y_true_val,
            y_pred=y_hat_val,
            accept=accept_val,
            prior_pred=y_hat_prior_val,
            aurc=val_aurc,
        ),
        build_family_row(
            model="prec_selective",
            split_name="test",
            y_true=y_true_test,
            y_pred=y_hat_test,
            accept=accept_test,
            prior_pred=y_hat_prior_test,
            aurc=test_aurc,
        ),
    ]

    target_coverages = parse_target_coverages(coverage_sweep_targets)
    selective_summary_rows = []
    for target in target_coverages:
        if target is None:
            target_label = "unconstrained"
            target_threshold = threshold
            val_curve_row = min(
                risk_curve,
                key=lambda row: abs(float(row["threshold"]) - float(target_threshold)),
            ) if risk_curve else None
        else:
            target_label = f"{target:.2f}"
            target_threshold, val_curve_row = select_threshold_for_target_coverage(
                curve=risk_curve,
                target_coverage=target,
            )

        sweep_test_y_hat, sweep_test_accept = apply_abstention_from_risk(
            mu_hat=test_preds["mu_hat"].values,
            z_hat=test_preds["z_hat"].values,
            alpha=alpha_test,
            risk=risk_test,
            threshold=target_threshold,
        )
        sweep_val_y_hat, sweep_val_accept = apply_abstention_from_risk(
            mu_hat=val_preds["mu_hat"].values,
            z_hat=val_preds["z_hat"].values,
            alpha=alpha_val,
            risk=risk_val,
            threshold=target_threshold,
        )
        sweep_test_metrics = compute_metrics(y_true_test, sweep_test_y_hat)
        sweep_test_selective = compute_selective_metrics(
            y_true_test,
            sweep_test_y_hat,
            sweep_test_accept,
        )
        sweep_val_metrics = compute_metrics(y_true_val, sweep_val_y_hat)
        sweep_val_selective = compute_selective_metrics(
            y_true_val,
            sweep_val_y_hat,
            sweep_val_accept,
        )
        selective_summary_rows.append(
            {
                "target_coverage": target_label,
                "threshold": float(target_threshold),
                "val_realized_coverage": None if val_curve_row is None else float(val_curve_row["coverage"]),
                "val_full_r2": sweep_val_metrics["r2"],
                "val_full_mse": sweep_val_metrics["mse"],
                "val_accepted_r2": sweep_val_selective["accepted_r2"],
                "val_accepted_mse": sweep_val_selective["accepted_mse"],
                "test_realized_coverage": sweep_test_selective["coverage"],
                "test_full_r2": sweep_test_metrics["r2"],
                "test_full_mse": sweep_test_metrics["mse"],
                "test_accepted_r2": sweep_test_selective["accepted_r2"],
                "test_accepted_mse": sweep_test_selective["accepted_mse"],
                "test_aurc": test_aurc,
                "test_gain_over_prior_mse": compute_metrics(y_true_test, y_hat_prior_test)["mse"] - sweep_test_metrics["mse"],
            }
        )

    # ===================================================================
    # Step 12: Save artifacts
    # ===================================================================
    logger.info("Step 12: Saving artifacts")

    pred_dir = os.path.join(output_dir, "predictions")

    # --- alpha outputs ---
    sigma_bar_train_path = save_dataframe(
        pd.DataFrame(
            {
                EVENT_ID_COLUMN: train_preds[EVENT_ID_COLUMN].values,
                "sigma2": train_preds["sigma2"].values,
                "sigma_bar_sq": sigma_bar_train,
            }
        ),
        os.path.join(pred_dir, f"sigma_bar_sq_train_{sv}_{rid}.csv"),
    )
    sigma_bar_val_path = save_dataframe(
        pd.DataFrame(
            {
                EVENT_ID_COLUMN: val_preds[EVENT_ID_COLUMN].values,
                "sigma2": val_preds["sigma2"].values,
                "sigma_bar_sq": sigma_bar_val,
            }
        ),
        os.path.join(pred_dir, f"sigma_bar_sq_val_{sv}_{rid}.csv"),
    )
    sigma_bar_test_path = save_dataframe(
        pd.DataFrame(
            {
                EVENT_ID_COLUMN: test_preds[EVENT_ID_COLUMN].values,
                "sigma2": test_preds["sigma2"].values,
                "sigma_bar_sq": sigma_bar_test,
            }
        ),
        os.path.join(pred_dir, f"sigma_bar_sq_test_{sv}_{rid}.csv"),
    )

    # --- alpha outputs ---
    alpha_train_path = save_dataframe(
        pd.DataFrame(
            {
                EVENT_ID_COLUMN: train_preds[EVENT_ID_COLUMN].values,
                "sigma2": train_preds["sigma2"].values,
                "sigma_bar_sq": sigma_bar_train,
                "alpha_obs": alpha_obs_train,
                "alpha_mm": alpha_train,
            }
        ),
        os.path.join(pred_dir, f"alpha_train_{sv}_{rid}.csv"),
    )
    alpha_val_path = save_dataframe(
        pd.DataFrame(
            {
                EVENT_ID_COLUMN: val_preds[EVENT_ID_COLUMN].values,
                "sigma2": val_preds["sigma2"].values,
                "sigma_bar_sq": sigma_bar_val,
                "alpha_obs": alpha_obs_val,
                "alpha_mm": alpha_val,
            }
        ),
        os.path.join(pred_dir, f"alpha_val_{sv}_{rid}.csv"),
    )
    alpha_test_path = save_dataframe(
        pd.DataFrame(
            {
                EVENT_ID_COLUMN: test_preds[EVENT_ID_COLUMN].values,
                "sigma2": test_preds["sigma2"].values,
                "sigma_bar_sq": sigma_bar_test,
                "alpha_obs": alpha_obs_test,
                "alpha_mm": alpha_test,
            }
        ),
        os.path.join(pred_dir, f"alpha_test_{sv}_{rid}.csv"),
    )

    # --- acceptance outputs ---
    accept_train_path = save_dataframe(
        pd.DataFrame(
            {
                EVENT_ID_COLUMN: train_preds[EVENT_ID_COLUMN].values,
                "risk": risk_train,
                "accept": accept_train.astype(int),
            }
        ),
        os.path.join(pred_dir, f"A_train_{sv}_{rid}.csv"),
    )
    accept_val_path = save_dataframe(
        pd.DataFrame(
            {
                EVENT_ID_COLUMN: val_preds[EVENT_ID_COLUMN].values,
                "risk": risk_val,
                "accept": accept_val.astype(int),
            }
        ),
        os.path.join(pred_dir, f"A_val_{sv}_{rid}.csv"),
    )
    accept_test_path = save_dataframe(
        pd.DataFrame(
            {
                EVENT_ID_COLUMN: test_preds[EVENT_ID_COLUMN].values,
                "risk": risk_test,
                "accept": accept_test.astype(int),
            }
        ),
        os.path.join(pred_dir, f"A_test_{sv}_{rid}.csv"),
    )

    # --- Final test predictions ---
    final_test_df = pd.DataFrame(
        {
            EVENT_ID_COLUMN: test_preds[EVENT_ID_COLUMN].values,
            TARGET_COLUMN: y_true_test,
            "mu_hat": test_preds["mu_hat"].values,
            "z_hat": test_preds["z_hat"].values,
            "sigma2": test_preds["sigma2"].values,
            "sigma_bar_sq": sigma_bar_test,
            "alpha_obs": alpha_obs_test,
            "alpha": alpha_test,
            "risk": risk_test,
            "y_hat_prior": y_hat_prior_test,
            "y_hat_plus_z": y_hat_plus_z_test,
            "y_hat_plus_alpha_z": y_hat_plus_alpha_test,
            "y_hat": y_hat_test,
            "accept": accept_test.astype(int),
            "split": "test",
        }
    )
    final_test_path = os.path.join(
        pred_dir, f"final_main_test_{sv}_{rid}.csv"
    )
    save_dataframe(final_test_df, final_test_path)
    logger.info("Saved final test predictions to %s", final_test_path)

    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    risk_curve_path = save_dataframe(
        pd.DataFrame(risk_curve),
        os.path.join(metrics_dir, f"risk_coverage_curve_val_{sv}_{rid}.csv"),
    )
    test_risk_curve_path = save_dataframe(
        pd.DataFrame(test_risk_curve),
        os.path.join(metrics_dir, f"risk_coverage_curve_test_{sv}_{rid}.csv"),
    )
    family_metrics_path = save_dataframe(
        pd.DataFrame(family_rows),
        os.path.join(metrics_dir, f"prior_correction_family_{sv}_{rid}.csv"),
    )
    selective_summary_path = save_dataframe(
        pd.DataFrame(selective_summary_rows),
        os.path.join(metrics_dir, f"selective_summary_by_target_coverage_{sv}_{rid}.csv"),
    )

    kappa_config = {
        "run_id": rid,
        "split_version": sv,
        "use_abstention": use_abstention,
        "metric": abstention_metric,
        "min_coverage": abstention_min_coverage,
        "tau2_method": tau2_method,
        "tau2": tau2,
        "sigma_bar_mode": "conservative_margin",
        "sigma_bar_quantile": sigma_bar_quantile,
        "sigma_bar_min_margin": sigma_bar_min_margin,
        "sigma_bar_delta": sigma_bar_delta,
        "selected_kappa": threshold,
        "selected_threshold_mode": selected_threshold_mode if use_abstention else "disabled",
        "abstention_target_coverage": abstention_target_coverage,
        "val_risk_curve_path": risk_curve_path,
        "test_risk_curve_path": test_risk_curve_path,
    }
    kappa_path = os.path.join(metrics_dir, f"kappa_config_{sv}_{rid}.json")
    with open(kappa_path, "w") as f:
        json.dump(kappa_config, f, indent=2, default=str)
    logger.info("Saved kappa config to %s", kappa_path)

    # --- Main metrics JSON ---
    results = {
        "run_id": rid,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "split_version": sv,
        "use_abstention": use_abstention,
        "proxy_noise_mode": proxy_noise_mode,
        "tau2_method": tau2_method,
        "tau2": tau2,
        "sigma_bar_mode": "conservative_margin",
        "sigma_bar_quantile": sigma_bar_quantile,
        "sigma_bar_min_margin": sigma_bar_min_margin,
        "sigma_bar_delta": sigma_bar_delta,
        "abstention_threshold": threshold,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "val_aurc": val_aurc,
        "test_aurc": test_aurc,
        "selective_test_metrics": selective_test,
        "selective_val_metrics": selective_val,
        "prior_correction_family_metrics_path": family_metrics_path,
        "selective_summary_by_target_coverage_path": selective_summary_path,
        "artifacts": {
            "sigma_bar_train": sigma_bar_train_path,
            "sigma_bar_val": sigma_bar_val_path,
            "sigma_bar_test": sigma_bar_test_path,
            "alpha_train": alpha_train_path,
            "alpha_val": alpha_val_path,
            "alpha_test": alpha_test_path,
            "A_train": accept_train_path,
            "A_val": accept_val_path,
            "A_test": accept_test_path,
            "final_test_predictions": final_test_path,
            "prior_correction_family_metrics": family_metrics_path,
            "selective_summary_by_target_coverage": selective_summary_path,
            "val_risk_curve": risk_curve_path,
            "test_risk_curve": test_risk_curve_path,
            "kappa_config": kappa_path,
        },
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
        "target_mode": target_mode,
        "abstention_metric": abstention_metric,
            "abstention_min_coverage": abstention_min_coverage,
            "abstention_target_coverage": abstention_target_coverage,
            "coverage_sweep_targets": coverage_sweep_targets,
            "tau2_method": tau2_method,
            "sigma_bar_quantile": sigma_bar_quantile,
            "sigma_bar_min_margin": sigma_bar_min_margin,
        },
    }

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
        "--ecc-feature-mode",
        type=str,
        default="ecc_only",
        choices=["ecc_only"],
        help="Feature bundle for the residual correction model. Mainline only allows ecc_only.",
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
        "--abstention-min-coverage",
        type=float,
        default=0.30,
        help="Minimum validation coverage required when selecting kappa.",
    )
    parser.add_argument(
        "--tau2-method",
        type=str,
        default="variance",
        choices=["variance", "moments"],
        help="Signal variance estimator.",
    )
    parser.add_argument(
        "--sigma-bar-quantile",
        type=float,
        default=0.90,
        help="Quantile used to build the conservative sigma_bar^2 margin.",
    )
    parser.add_argument(
        "--sigma-bar-min-margin",
        type=float,
        default=0.0,
        help="Lower bound for the conservative sigma_bar^2 margin.",
    )
    parser.add_argument(
        "--abstention-target-coverage",
        type=float,
        default=None,
        help="If set, choose kappa to match this target validation coverage.",
    )
    parser.add_argument(
        "--coverage-sweep-targets",
        type=str,
        default="0.2,0.4,0.6,0.8,unconstrained",
        help="Comma-separated target coverages for selective summary export.",
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        default="shock_minus_pre",
        choices=["shock_minus_pre", "log_rv_ratio"],
        help="Target definition used throughout the full pipeline.",
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
        ecc_feature_mode=args.ecc_feature_mode,
        proxy_noise_mode=args.proxy_noise_mode,
        proxy_noise_params=proxy_noise_params,
        use_abstention=not args.no_abstention,
        abstention_metric=args.abstention_metric,
        abstention_min_coverage=args.abstention_min_coverage,
        tau2_method=args.tau2_method,
        sigma_bar_quantile=args.sigma_bar_quantile,
        sigma_bar_min_margin=args.sigma_bar_min_margin,
        abstention_target_coverage=args.abstention_target_coverage,
        coverage_sweep_targets=args.coverage_sweep_targets,
        target_mode=args.target_mode,
    )


if __name__ == "__main__":
    main()
