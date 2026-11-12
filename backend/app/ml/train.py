"""Train one gradient-boosted model per target (+ 10th/90th percentile models for a range).

python -m app.ml.train
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.clock import today_et
from app.config import TARGETS as ACTIVE_TARGETS
from app.db import upsert
from app.features.build import FEATURES, build_features, load_frames, training_rows
from app.ml import intervals, tuning
from app.ml.registry import dump
from app.ml.xgb import XGBModel
from app.models import ModelRun

HOLDOUT_DAYS = 21
MAX_ITER = 800


def _model(params: dict, max_iter: int = MAX_ITER, early_stop: bool = True, **loss):
    if params.get("model") == "xgboost":
        return XGBModel(params, max_iter, early_stop)
    hgb = {k: v for k, v in params.items() if k not in ("half_life_days", "model")}
    return HistGradientBoostingRegressor(
        **hgb, **loss, max_iter=max_iter, early_stopping=early_stop, n_iter_no_change=25, validation_fraction=0.1, random_state=0
    )


def _weights(dates: pd.Series, half_life_days: float) -> np.ndarray:
    """A game `half_life_days` old counts half as much as last night's."""
    age = (dates.max() - dates).dt.days.to_numpy()
    return 0.5 ** (age / half_life_days)


def usable_features(train: pd.DataFrame) -> list[str]:
    """FEATURES that have at least one value in `train`.

    A brand-new feature (e.g. sportsbook lines, only recorded going forward) is entirely missing from the
    training window at first, and HistGradientBoostingRegressor crashes on an all-missing column. Leave such
    features out until they have data; the model bundle records which features it was trained on.
    """
    return [f for f in FEATURES if train[f].notna().any()]


def _score(params: dict, train: pd.DataFrame, hold: pd.DataFrame, target: str, feats: list[str]) -> tuple[float, object]:
    m = _model(params).fit(train[feats], train[target], sample_weight=_weights(train["date"], params["half_life_days"]))
    return mean_absolute_error(hold[target], m.predict(hold[feats])), m


def train_target(rows: pd.DataFrame, target: str, version: str) -> dict:
    cutoff = rows["date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train, hold = rows[rows["date"] <= cutoff], rows[rows["date"] > cutoff]
    feats = usable_features(train)

    # 1) Champion vs today's challenger on the most recent weeks, trained only on what came before them.
    champ = tuning.champion(target)
    chall = tuning.challenger(target, champ, today_et())
    champ_mae, champ_model = _score(champ, train, hold, target, feats)
    chall_mae, chall_model = _score(chall, train, hold, target, feats) if chall != champ else (np.inf, None)
    if chall_mae < champ_mae * (1 - tuning.MIN_IMPROVEMENT):
        params, point, mae = chall, chall_model, chall_mae
        note = f"challenger won: MAE {chall_mae:.3f} < {champ_mae:.3f} with {_diff(champ, chall)}"
    else:
        params, point, mae = champ, champ_model, champ_mae
        note = f"champion kept: MAE {champ_mae:.3f} vs challenger {_diff(champ, chall)} {chall_mae:.3f}"
    tuning.save_champion(target, params, note)

    pred = point.predict(hold[feats])
    baseline_mae = mean_absolute_error(hold[target], hold[f"{target}_l10"])
    # Calibrate ranges on the first half of the holdout and check them on the second, so coverage is honest.
    first = (hold["date"] <= hold["date"].median()).to_numpy()
    check_cal = intervals.fit(hold[first], target, pred[first])
    lo, hi = intervals.apply(hold[~first], target, pred[~first], check_cal)
    coverage = intervals.coverage(hold.loc[~first, target].to_numpy(), lo, hi)
    cal = intervals.fit(hold, target, pred)
    sample = hold.sample(min(len(hold), 2000), random_state=0)
    imp = permutation_importance(point, sample[feats], sample[target], n_repeats=1, random_state=0, scoring="neg_mean_absolute_error")
    importance = dict(sorted(zip(feats, imp.importances_mean.round(4).tolist()), key=lambda kv: -kv[1]))

    # 2) Refit on everything with the tree count found above (more data -> ~10% more trees).
    w_all = _weights(rows["date"], params["half_life_days"])
    n_iter = int(point.n_iter_ * 1.1)
    X, y = rows[feats], rows[target]
    bundle = {
        "features": feats,
        "params": params,
        "interval": cal,
        "point": _model(params, n_iter, early_stop=False).fit(X, y, sample_weight=w_all),
    }
    run = {
        "version": version, "target": target, "trained_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "n_train": len(train), "n_holdout": len(hold), "mae": float(mae),
        "rmse": float(np.sqrt(mean_squared_error(hold[target], pred))), "baseline_mae": float(baseline_mae),
        "interval_coverage": coverage,
        "promoted": bool(mae < baseline_mae),  # never ship a model that loses to "just use their last-10 average"
        "params": params, "tuning_note": note, "feature_importance": importance, "artifact": dump(bundle),
    }
    upsert(ModelRun.__table__, [run])
    return run


def _diff(a: dict, b: dict) -> str:
    return ", ".join(f"{k}={b[k]}" for k in b if a.get(k) != b[k]) or "(same)"


def train_all() -> list[dict]:
    rows = training_rows(build_features(load_frames()))
    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    results = []
    for target in ACTIVE_TARGETS:
        run = train_target(rows, target, version)
        top = list(run["feature_importance"])[:5]
        print(f"{target}: MAE {run['mae']:.2f} vs L10 baseline {run['baseline_mae']:.2f} "
              f"| 80% range coverage {run['interval_coverage']:.0%} | promoted={run['promoted']}\n"
              f"  tuning: {run['tuning_note']}\n  top features: {top}", flush=True)
        results.append(run)
    return results


if __name__ == "__main__":
    train_all()
