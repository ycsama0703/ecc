"""
Abstention Module

Applies the abstention rule: when the estimated noise variance sigma^2
exceeds a threshold, the ECC correction is deemed unreliable and the
prediction falls back to the market prior mu_hat.

Core equation context:
    if sigma2 <= threshold:
        y_hat = mu_hat + alpha * z_hat      (accept: use ECC correction)
    else:
        y_hat = mu_hat                       (abstain: fall back to prior)

Output:
    - final y_hat
    - accept flag (True = ECC correction used, False = abstained)
"""

import numpy as np


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
