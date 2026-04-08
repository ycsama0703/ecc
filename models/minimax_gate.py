"""
Minimax Gate

Computes the minimax shrinkage coefficient alpha that controls
how much the ECC residual correction z is trusted.

Core equation context:
    y_hat      = mu_hat + alpha * z
    alpha_obs  = tau^2 / (tau^2 + sigma_hat^2)
    alpha_mm   = tau^2 / (tau^2 + sigma_bar^2)

Where:
    tau^2        = signal variance (estimated from training residuals)
    sigma_hat^2  = point estimate of observation-level noise variance
    sigma_bar^2  = conservative / worst-case observation-level noise variance

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


def estimate_conservative_margin(
    u: np.ndarray,
    sigma2_hat: np.ndarray,
    quantile: float = 0.90,
    min_margin: float = 0.0,
) -> float:
    """Estimate a conservative additive margin for sigma_hat^2.

    Implements plan option B:
        sigma_bar^2 = sigma_hat^2 + delta

    The margin is estimated from positive under-coverage of the point
    estimate against realised squared correction error:
        delta = Quantile(max(u - sigma_hat^2, 0), q)

    Parameters
    ----------
    u : np.ndarray
        Realised squared ECC errors.
    sigma2_hat : np.ndarray
        Point estimates of noise variance.
    quantile : float
        Quantile in [0, 1] used to obtain a conservative margin.
    min_margin : float
        Lower bound applied to the returned margin.

    Returns
    -------
    float
        Non-negative conservative margin delta.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")

    u = np.asarray(u, dtype=np.float64)
    sigma2_hat = np.asarray(sigma2_hat, dtype=np.float64)
    if u.shape != sigma2_hat.shape:
        raise ValueError("u and sigma2_hat must have the same shape")

    positive_gap = np.maximum(u - sigma2_hat, 0.0)
    delta = float(np.quantile(positive_gap, quantile)) if len(positive_gap) else 0.0
    return max(float(min_margin), delta)


def build_sigma_bar(
    sigma2_hat: float | np.ndarray,
    delta: float | np.ndarray,
) -> float | np.ndarray:
    """Build conservative noise estimates sigma_bar^2.

    Parameters
    ----------
    sigma2_hat : float or np.ndarray
        Point estimates of noise variance.
    delta : float or np.ndarray
        Conservative additive margin.

    Returns
    -------
    float or np.ndarray
        Conservative noise estimate sigma_bar^2 >= sigma_hat^2.
    """
    sigma2_hat = np.asarray(sigma2_hat, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    sigma2_bar = np.maximum(sigma2_hat, 0.0) + np.maximum(delta, 0.0)
    return float(sigma2_bar) if sigma2_bar.ndim == 0 else sigma2_bar


def estimate_tau2(
    r_tilde: np.ndarray,
    sigma2: np.ndarray | None = None,
    method: str = "moments",
) -> float:
    """Estimate the signal variance tau^2 from training data.

    Supported estimators:
        - "variance": tau^2 = Var(r_tilde)
        - "moments":  tau^2 = max(0, Var(r_tilde) - mean(sigma^2))

    The engineering plan uses the plain training residual variance by
    default. A method-of-moments variant is also retained for ablations
    and backward compatibility.

    Parameters
    ----------
    r_tilde : np.ndarray
        Training residuals (y - mu_hat).
    sigma2 : np.ndarray, optional
        Per-observation noise variance estimates. Required only when
        method="moments".
    method : str
        Tau^2 estimation method: "variance" or "moments".

    Returns
    -------
    float
        Estimated signal variance tau^2 (non-negative).
    """
    r_tilde = np.asarray(r_tilde, dtype=np.float64)
    if len(r_tilde) < 2:
        return 0.0

    var_r = float(np.var(r_tilde, ddof=1))
    if method == "variance":
        return max(0.0, var_r)

    if method != "moments":
        raise ValueError(f"Unknown tau^2 estimation method: {method}")

    if sigma2 is None:
        raise ValueError("sigma2 is required when method='moments'")
    sigma2 = np.asarray(sigma2, dtype=np.float64)
    mean_sigma2 = float(np.mean(sigma2))
    return max(0.0, var_r - mean_sigma2)
