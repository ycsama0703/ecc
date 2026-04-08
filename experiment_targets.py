"""Shared target construction helpers for experiment variants."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMN = "shock_minus_pre"
SUPPORTED_TARGET_MODES = {"shock_minus_pre", "log_rv_ratio"}


def apply_target_mode(
    df: pd.DataFrame,
    target_mode: str = "shock_minus_pre",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Return a copy with the canonical target column rewritten if needed."""
    if target_mode not in SUPPORTED_TARGET_MODES:
        raise ValueError(
            f"Unsupported target_mode={target_mode!r}. "
            f"Expected one of {sorted(SUPPORTED_TARGET_MODES)}."
        )

    out = df.copy()
    if target_mode == "shock_minus_pre":
        return out

    required = {"RV_pre_60m", "RV_post_60m"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(
            f"Target mode {target_mode!r} requires columns: {sorted(missing)}"
        )

    rv_pre = pd.to_numeric(out["RV_pre_60m"], errors="coerce").astype(float)
    rv_post = pd.to_numeric(out["RV_post_60m"], errors="coerce").astype(float)
    out[TARGET_COLUMN] = np.log((rv_post + eps) / (rv_pre + eps))
    return out
