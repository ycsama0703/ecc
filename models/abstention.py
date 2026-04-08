"""
Abstention Module

Applies abstention rules that decide whether the ECC correction should
be used or whether the prediction should fall back to the market prior.

Core equation context:
    if risk_i <= kappa:
        y_hat = mu_hat + alpha * z_hat      (accept: use ECC correction)
    else:
        y_hat = mu_hat                       (abstain: fall back to prior)

Output:
    - final y_hat
    - accept flag (True = ECC correction used, False = abstained)
"""

import numpy as np
from scipy import stats as scipy_stats


def compute_closed_form_risk(
    tau2: float | np.ndarray,
    sigma2: float | np.ndarray,
) -> float | np.ndarray:
    """Compute the closed-form conditional risk used for abstention.

    R = tau^2 * sigma^2 / (tau^2 + sigma^2)
    """
    tau2 = np.asarray(tau2, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)
    denom = tau2 + sigma2
    risk = np.where(denom > 0, tau2 * sigma2 / denom, 0.0)
    return float(risk) if risk.ndim == 0 else risk


def apply_abstention(
    mu_hat: np.ndarray,
    z_hat: np.ndarray,
    alpha: float | np.ndarray,
    sigma2: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the abstention rule to produce final predictions.

    Parameters
    ----------
    mu_hat : np.ndarray
        Market prior predictions.
    z_hat : np.ndarray
        ECC residual predictions.
    alpha : float or np.ndarray
        Shrinkage coefficient(s) from the minimax gate.
    sigma2 : np.ndarray
        Per-observation noise variance estimates from the proxy noise model.
    threshold : float
        Abstention threshold on sigma2. Observations with sigma2 > threshold
        fall back to mu_hat (the ECC correction is not trusted).

    Returns
    -------
    y_hat : np.ndarray
        Final predictions.
    accept : np.ndarray
        Boolean array. True where ECC correction is accepted,
        False where abstention occurred.
    """
    mu_hat = np.asarray(mu_hat, dtype=np.float64)
    z_hat = np.asarray(z_hat, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    accept = sigma2 <= threshold

    y_hat = np.where(
        accept,
        mu_hat + alpha * z_hat,
        mu_hat,
    )

    return y_hat, accept


def apply_abstention_from_risk(
    mu_hat: np.ndarray,
    z_hat: np.ndarray,
    alpha: float | np.ndarray,
    risk: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply abstention using risk scores instead of raw sigma^2."""
    mu_hat = np.asarray(mu_hat, dtype=np.float64)
    z_hat = np.asarray(z_hat, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    risk = np.asarray(risk, dtype=np.float64)

    accept = risk <= threshold
    y_hat = np.where(accept, mu_hat + alpha * z_hat, mu_hat)
    return y_hat, accept


def select_threshold(
    sigma2_val: np.ndarray,
    y_val: np.ndarray,
    mu_hat_val: np.ndarray,
    z_hat_val: np.ndarray,
    alpha_val: float | np.ndarray,
    candidates: np.ndarray | None = None,
    metric: str = "mse",
) -> float:
    """Select the abstention threshold on the validation set.

    Sweeps over candidate thresholds and picks the one that minimises
    the chosen metric on accepted observations, subject to a minimum
    coverage of 10% to avoid degenerate solutions.

    Parameters
    ----------
    sigma2_val : np.ndarray
        Validation noise variance estimates.
    y_val : np.ndarray
        True validation targets.
    mu_hat_val : np.ndarray
        Validation market prior predictions.
    z_hat_val : np.ndarray
        Validation ECC residual predictions.
    alpha_val : float or np.ndarray
        Validation shrinkage coefficient(s).
    candidates : np.ndarray, optional
        Candidate thresholds to evaluate. If None, uses percentiles
        of sigma2_val from 10th to 100th.
    metric : str
        Metric to minimise on accepted observations: "mse" or "mae".

    Returns
    -------
    float
        Selected threshold.
    """
    sigma2_val = np.asarray(sigma2_val, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)
    mu_hat_val = np.asarray(mu_hat_val, dtype=np.float64)
    z_hat_val = np.asarray(z_hat_val, dtype=np.float64)
    alpha_val = np.asarray(alpha_val, dtype=np.float64)

    if candidates is None:
        percentiles = np.arange(10, 101, 5, dtype=np.float64)
        candidates = np.percentile(sigma2_val, percentiles)
        # Always include the maximum so that 100% coverage is a candidate
        candidates = np.append(candidates, np.max(sigma2_val) + 1e-8)
        candidates = np.unique(candidates)

    n = len(y_val)
    min_coverage = 0.10

    best_threshold = float(np.max(sigma2_val) + 1e-8)
    best_score = np.inf

    for t in candidates:
        y_hat, accept = apply_abstention(
            mu_hat_val, z_hat_val, alpha_val, sigma2_val, float(t)
        )
        coverage = np.sum(accept) / n
        if coverage < min_coverage:
            continue

        errors = y_val[accept] - y_hat[accept]
        if metric == "mse":
            score = float(np.mean(errors ** 2))
        elif metric == "mae":
            score = float(np.mean(np.abs(errors)))
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score < best_score:
            best_score = score
            best_threshold = float(t)

    return best_threshold


def select_risk_threshold(
    risk_val: np.ndarray,
    y_val: np.ndarray,
    mu_hat_val: np.ndarray,
    z_hat_val: np.ndarray,
    alpha_val: float | np.ndarray,
    candidates: np.ndarray | None = None,
    metric: str = "mse",
    min_coverage: float = 0.30,
) -> tuple[float, list[dict]]:
    """Select the abstention threshold on validation risk scores.

    Returns both the selected threshold and the full validation curve so
    the caller can export risk-coverage diagnostics.
    """
    curve = build_risk_coverage_curve(
        risk=risk_val,
        y_true=y_val,
        mu_hat=mu_hat_val,
        z_hat=z_hat_val,
        alpha=alpha_val,
        candidates=candidates,
    )
    best_threshold = select_risk_threshold_from_curve(
        curve=curve,
        metric=metric,
        min_coverage=min_coverage,
    )
    return best_threshold, curve


def build_risk_coverage_curve(
    risk: np.ndarray,
    y_true: np.ndarray,
    mu_hat: np.ndarray,
    z_hat: np.ndarray,
    alpha: float | np.ndarray,
    candidates: np.ndarray | None = None,
) -> list[dict]:
    """Build a risk-coverage curve over candidate abstention thresholds."""
    risk = np.asarray(risk, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    mu_hat = np.asarray(mu_hat, dtype=np.float64)
    z_hat = np.asarray(z_hat, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)

    if candidates is None:
        percentiles = np.arange(0, 101, 5, dtype=np.float64)
        candidates = np.percentile(risk, percentiles)
        candidates = np.append(candidates, np.max(risk) + 1e-8)
        candidates = np.unique(candidates)

    n = len(y_true)
    curve = []

    for t in candidates:
        y_hat, accept = apply_abstention_from_risk(
            mu_hat,
            z_hat,
            alpha,
            risk,
            float(t),
        )
        coverage = float(np.sum(accept) / n) if n > 0 else 0.0
        accepted_errors = y_true[accept] - y_hat[accept]

        if np.any(accept):
            mse = float(np.mean(accepted_errors ** 2))
            mae = float(np.mean(np.abs(accepted_errors)))
            accepted_true = y_true[accept]
            accepted_pred = y_hat[accept]
            accepted_mean = float(np.mean(accepted_true))
            ss_res = float(np.sum((accepted_true - accepted_pred) ** 2))
            ss_tot = float(np.sum((accepted_true - accepted_mean) ** 2))
            accepted_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
            if len(accepted_true) >= 2:
                accepted_spearman, _ = scipy_stats.spearmanr(accepted_true, accepted_pred)
                accepted_spearman = float(accepted_spearman)
            else:
                accepted_spearman = float("nan")
        else:
            mse = float("nan")
            mae = float("nan")
            accepted_r2 = float("nan")
            accepted_spearman = float("nan")

        curve.append(
            {
                "threshold": float(t),
                "coverage": coverage,
                "accepted_mse": mse,
                "accepted_mae": mae,
                "accepted_r2": accepted_r2,
                "accepted_spearman": accepted_spearman,
                "n_accepted": int(np.sum(accept)),
            }
        )

    return curve


def select_risk_threshold_from_curve(
    curve: list[dict],
    metric: str = "mse",
    min_coverage: float = 0.30,
) -> float:
    """Select the best threshold from a precomputed risk-coverage curve."""
    if not curve:
        raise ValueError("curve must not be empty")

    metric_key = {
        "mse": "accepted_mse",
        "mae": "accepted_mae",
    }.get(metric)
    if metric_key is None:
        raise ValueError(f"Unknown metric: {metric}")

    best_threshold = float(curve[-1]["threshold"])
    best_score = np.inf

    for row in curve:
        coverage = float(row["coverage"])
        score = float(row[metric_key])
        if coverage < min_coverage or np.isnan(score):
            continue
        if score < best_score:
            best_score = score
            best_threshold = float(row["threshold"])

    return best_threshold


def compute_aurc(curve: list[dict], metric: str = "mse") -> float:
    """Approximate AURC by trapezoidal integration over coverage."""
    if not curve:
        return float("nan")

    metric_key = {
        "mse": "accepted_mse",
        "mae": "accepted_mae",
    }.get(metric)
    if metric_key is None:
        raise ValueError(f"Unknown metric: {metric}")

    valid = [
        row for row in curve
        if not np.isnan(float(row["coverage"])) and not np.isnan(float(row[metric_key]))
    ]
    if len(valid) < 2:
        return float("nan")

    valid = sorted(valid, key=lambda row: float(row["coverage"]))
    coverage = np.asarray([float(row["coverage"]) for row in valid], dtype=np.float64)
    risk = np.asarray([float(row[metric_key]) for row in valid], dtype=np.float64)
    return float(np.trapz(risk, coverage))


def select_threshold_for_target_coverage(
    curve: list[dict],
    target_coverage: float,
) -> tuple[float, dict]:
    """Select the threshold whose realized coverage is closest to target."""
    if not curve:
        raise ValueError("curve must not be empty")
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be in (0, 1].")

    best_row = min(
        curve,
        key=lambda row: (
            abs(float(row["coverage"]) - target_coverage),
            abs(float(row["coverage"]) - target_coverage) > 1e-12,
            -float(row["coverage"]),
        ),
    )
    return float(best_row["threshold"]), best_row
