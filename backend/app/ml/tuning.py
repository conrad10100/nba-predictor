"""Champion/challenger self-tuning.

Each day one challenger config is trained alongside the current champion and both are scored on the most recent
weeks of games. The challenger takes over only if it is meaningfully better, so settings drift toward whatever
predicts this season best without flip-flopping on noise.
"""

from datetime import date

from app.db import SessionLocal
from app.ml import xgb
from app.models import ModelConfig

DEFAULT = {"model": "hgb", "half_life_days": 365, "learning_rate": 0.08, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 1.0}

# Variations around the default, one knob at a time; cycled through one per day.
CANDIDATES = [
    {"half_life_days": 120}, {"half_life_days": 240}, {"half_life_days": 730},
    {"max_leaf_nodes": 15}, {"max_leaf_nodes": 63},
    {"min_samples_leaf": 20}, {"min_samples_leaf": 100},
    {"learning_rate": 0.04}, {"l2_regularization": 5.0},
    {"model": "hgb"}, {"model": "xgboost"},  # algorithm itself competes too (XGBoost only where it's installed)
]
MIN_IMPROVEMENT = 0.005  # challenger must cut holdout MAE by 0.5%+


def champion(target: str) -> dict:
    with SessionLocal() as s:
        cfg = s.get(ModelConfig, target)
    params = {**DEFAULT, **cfg.params} if cfg else dict(DEFAULT)
    if params["model"] == "xgboost" and not xgb.AVAILABLE:
        params["model"] = "hgb"  # e.g. running locally on a Mac without libomp
    return params


def challenger(target: str, current: dict, on: date) -> dict:
    """Next candidate in the daily rotation that actually differs from the champion."""
    candidates = [c for c in CANDIDATES if xgb.AVAILABLE or c.get("model") != "xgboost"]
    start = on.toordinal() % len(candidates)
    for i in range(len(candidates)):
        cand = {**current, **candidates[(start + i) % len(candidates)]}
        if cand != current:
            return cand
    return current


def bias_correction(target: str) -> float:
    with SessionLocal() as s:
        cfg = s.get(ModelConfig, target)
        return cfg.bias_correction if cfg else 0.0


def save_champion(target: str, params: dict, note: str) -> None:
    with SessionLocal() as s:
        cfg = s.get(ModelConfig, target)
        if cfg is None:
            s.add(ModelConfig(target=target, params=params, bias_correction=0.0, note=note))
        else:
            cfg.params, cfg.note = params, note
        s.commit()
