"""Champion/challenger self-tuning.

Each day one challenger config is trained alongside the current champion and both are scored on the most recent
weeks of games. The challenger takes over only if it is meaningfully better, so settings drift toward whatever
predicts this season best without flip-flopping on noise.
"""

from datetime import date

from app.db import SessionLocal
from app.ml import xgb
from app.models import ModelConfig

DEFAULT = {"model": "hgb", "half_life_days": 365, "learning_rate": 0.08, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 1.0,
           "minutes_weight": 0,
           # Opponent defender features (who guards the player, how good tonight's defenders are). Off by default so the
           # existing model stays champion until the version with them wins on recent games.
           "defender_features": 0}

# Variations around the default, one knob at a time; cycled through one per day.
CANDIDATES = [
    {"minutes_weight": 1}, {"minutes_weight": 2},  # weight rotation players more (0 = every player counts the same)
    {"half_life_days": 120}, {"half_life_days": 240}, {"half_life_days": 730},
    {"max_leaf_nodes": 15}, {"max_leaf_nodes": 63},
    {"min_samples_leaf": 20}, {"min_samples_leaf": 100},
    {"learning_rate": 0.04}, {"l2_regularization": 5.0},
    {"model": "hgb"}, {"model": "xgboost"},  # algorithm itself competes too (XGBoost only where it's installed)
]
# Feature-set switches. These don't compete in the daily backtest: the champion with the switch flipped runs as a
# shadow model on live slates and takes over only by winning a week of real results (app.ml.shadow).
FEATURE_SWITCHES = ["defender_features"]
MIN_IMPROVEMENT = 0.005  # challenger must cut holdout MAE (on players with DraftKings lines) by 0.5%+


def locked(target: str) -> dict:
    """Settings fixed by hand (python -m app.ml.modelctl); self-tuning never changes these."""
    with SessionLocal() as s:
        cfg = s.get(ModelConfig, target)
        return dict(cfg.locked_params or {}) if cfg else {}


def champion(target: str) -> dict:
    with SessionLocal() as s:
        cfg = s.get(ModelConfig, target)
        stored = dict(cfg.params) if cfg else {}
        fixed = dict(cfg.locked_params or {}) if cfg else {}
    params = {**DEFAULT, **stored, **fixed}
    if params["model"] == "xgboost" and not xgb.AVAILABLE:
        params["model"] = "hgb"  # e.g. running locally on a Mac without libomp
    return params


def challenger(target: str, current: dict, on: date) -> dict:
    """Next candidate in the daily rotation that actually differs from the champion (never touching locked settings)."""
    fixed = locked(target)
    candidates = [c for c in CANDIDATES if (xgb.AVAILABLE or c.get("model") != "xgboost") and not set(c) & set(fixed)]
    start = on.toordinal() % max(len(candidates), 1)
    for i in range(len(candidates)):
        cand = {**current, **candidates[(start + i) % len(candidates)]}
        if cand != current:
            return cand
    return current


def shadow_params(target: str, current: dict) -> dict | None:
    """The champion with the feature switch flipped, or None if the switch is locked by hand."""
    key = FEATURE_SWITCHES[0]
    if key in locked(target):
        return None
    return {**current, key: 0 if current.get(key) else 1}


def set_switch(target: str, key: str, value: int, note: str) -> None:
    """Flip a feature switch on the stored champion (the weekly shadow verdict)."""
    with SessionLocal() as s:
        cfg = s.get(ModelConfig, target)
        if cfg is None:
            s.add(ModelConfig(target=target, params={**DEFAULT, key: value}, bias_correction=0.0, note=note))
        else:
            cfg.params = {**(cfg.params or {}), key: value}
        s.commit()


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
