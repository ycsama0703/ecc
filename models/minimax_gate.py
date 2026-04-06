"""
Minimax Gate

Computes the minimax shrinkage coefficient alpha that controls
how much the ECC residual correction z is trusted.

Core equation context:
    y_hat = mu_hat + alpha * z
    alpha = tau^2 / (tau^2 + sigma^2)

Where:
    tau^2   = signal variance (estimated from training residuals)
    sigma^2 = observation-level noise variance (from proxy noise model)

A higher alpha means the ECC signal is trusted more.
When sigma^2 is large relative to tau^2, alpha shrinks toward 0
and the prediction falls back to the market prior mu_hat.
"""

import numpy as np


def compute_gate(
    tau2: float | np.ndarray,
    sigma2: float | np.ndarray,
) -> float | np.ndarray:
    """Compute the minimax shrinkage coefficient.

    Parameters
    ----------
    tau2 : float or np.ndarray
        Signal variance. Scalar (global estimate) or per-observation array.
    sigma2 : float or np.ndarray
        Noise variance. Scalar or per-observation array from the proxy
        noise model.

    Returns
    -------
    float or np.ndarray
        Shrinkage coefficient alpha in [0, 1].
        alpha = tau2 / (tau2 + sigma2)
    """
    tau2 = np.asarray(tau2, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    denom = tau2 + sigma2
    # When both tau2 and sigma2 are zero, the signal carries no information;
    # default to full shrinkage (fall back to market prior).
    alpha = np.where(denom > 0, tau2 / denom, 0.0)
    return float(alpha) if alpha.ndim == 0 else alpha


def compute_gate_upper(
    tau2: float | np.ndarray,
    sigma2_upper: float | np.ndarray,
) -> float | np.ndarray:
    """Compute the robust (worst-case) minimax shrinkage coefficient.

    Uses the upper bound of sigma^2 to produce a conservative alpha.
    This is the minimax-optimal choice: it minimises the maximum risk
    over the uncertainty set for sigma^2.

    Parameters
    ----------
    tau2 : float or np.ndarray
        Signal variance.
    sigma2_upper : float or np.ndarray
        Upper bound of noise variance (e.g. confidence interval upper end
        or worst-case estimate from the proxy noise model).

    Returns
    -------
    float or np.ndarray
        Conservative shrinkage coefficient alpha in [0, 1].
    """
    return compute_gate(tau2, sigma2_upper)


def estimate_tau2(
    r_tilde: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """Estimate the signal variance tau^2 from training data.

    Uses the method-of-moments estimator:
        tau^2 = max(0, Var(r_tilde) - mean(sigma^2))

    This decomposes the observed residual variance into signal and noise
    components. If noise dominates, tau^2 is floored at zero.

    Parameters
    ----------
    r_tilde : np.ndarray
        Training residuals (y - mu_hat).
    sigma2 : np.ndarray
        Per-observation noise variance estimates from the proxy noise model
        on the training set.

    Returns
    -------
    float
        Estimated signal variance tau^2 (non-negative).
    """
    r_tilde = np.asarray(r_tilde, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    var_r = float(np.var(r_tilde, ddof=1))
    mean_sigma2 = float(np.mean(sigma2))

    tau2 = max(0.0, var_r - mean_sigma2)
    return tau2
