"""Prediction ranges via split-conformal calibration.

On recent held-out games, measure misses in units of each player's own volatility (std of their last 10 games),
then take the 10th/90th percentiles of those scaled misses. Range = prediction + percentile * player's volatility.
It covers ~80% of recent results by construction, and steady players get tighter ranges than streaky ones.
"""

import numpy as np
import pandas as pd

LOW_Q, HIGH_Q = 0.10, 0.90
# Minimum volatility so near-constant players (few games, or always 0) still get a sensible range.
SPREAD_FLOOR = {"pts": 2.0, "reb": 1.0, "ast": 1.0}
# For models trained before ranges were calibrated: +/-1.28 sd is a normal 80% range.
FALLBACK = {"lo": -1.28, "hi": 1.28, "fill": 4.0}


def spread(df: pd.DataFrame, target: str, fill: float) -> np.ndarray:
    return df[f"{target}_std_l10"].fillna(fill).clip(lower=SPREAD_FLOOR[target]).to_numpy()


def fit(df: pd.DataFrame, target: str, pred: np.ndarray) -> dict:
    fill = float(df[f"{target}_std_l10"].median())
    s = spread(df, target, fill)
    z = (df[target].to_numpy() - pred) / s
    return {"lo": float(np.quantile(z, LOW_Q)), "hi": float(np.quantile(z, HIGH_Q)), "fill": fill}


def apply(df: pd.DataFrame, target: str, pred: np.ndarray, cal: dict) -> tuple[np.ndarray, np.ndarray]:
    s = spread(df, target, cal["fill"])
    return np.clip(pred + cal["lo"] * s, 0, None), pred + cal["hi"] * s


def coverage(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(((actual >= lo) & (actual <= hi)).mean())
