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
from app.db import SessionLocal, upsert
from app.features.build import DEFENDER_FEATURES, FEATURES, build_features, load_frames, training_rows
from app.ml import intervals, tuning
from app.ml.registry import dump
from app.ml.xgb import XGBModel
from app.models import ModelRun, PlayerProp, ShadowModel

HOLDOUT_DAYS = 21
MAX_ITER = 800
NOT_MODEL_PARAMS = ("half_life_days", "model", "minutes_weight", "defender_features")  # training settings, not estimator arguments
MINUTES_REF = 30.0  # a player averaging this many minutes counts fully under minutes weighting
ROTATION_MIN = 20.0  # fallback definition of "players with lines" before lines were stored
MIN_LINE_ROWS = 200  # need this many holdout rows with DraftKings lines to judge on them


def _model(params: dict, max_iter: int = MAX_ITER, early_stop: bool = True, **loss):
    if params.get("model") == "xgboost":
        return XGBModel(params, max_iter, early_stop)
    hgb = {k: v for k, v in params.items() if k not in NOT_MODEL_PARAMS}
    return HistGradientBoostingRegressor(
        **hgb, **loss, max_iter=max_iter, early_stopping=early_stop, n_iter_no_change=25, validation_fraction=0.1, random_state=0
    )


def _weights(rows: pd.DataFrame, params: dict) -> np.ndarray:
    """Training weight per game.

    Recency: a game `half_life_days` old counts half as much as last night's.
    Minutes (when the `minutes_weight` setting is on): rotation players count more, so the model spends its effort on
    the players DraftKings posts lines for. Uses the player's pre-game last-10 minutes average, never the minutes
    he went on to play, which would tie the weight to the outcome and bias predictions upward.
    """
    age = (rows["date"].max() - rows["date"]).dt.days.to_numpy()
    w = 0.5 ** (age / params["half_life_days"])
    k = params.get("minutes_weight", 0)
    if k:
        share = (rows["min_l10"].fillna(MINUTES_REF) / MINUTES_REF).clip(0.1, 1.5).to_numpy()
        w = w * share ** k
    return w


def _focus(hold: pd.DataFrame) -> tuple[np.ndarray, str]:
    """The holdout rows the model is judged on: players DraftKings posted a line for, since that's where the bets
    are (bench players are easy to predict and flatter an all-player average). Falls back to rotation players
    while there are too few lines in the holdout window."""
    with SessionLocal() as s:
        keys = set(s.execute(PlayerProp.__table__.select().with_only_columns(PlayerProp.player_id, PlayerProp.game_id)).all())
    mask = np.array([(p, g) in keys for p, g in zip(hold["player_id"], hold["game_id"])], dtype=bool)
    if mask.sum() >= MIN_LINE_ROWS:
        return mask, "players with DraftKings lines"
    return (hold["min_l10"] >= ROTATION_MIN).to_numpy(), f"rotation players ({ROTATION_MIN:.0f}+ min)"


def usable_features(train: pd.DataFrame, params: dict | None = None) -> list[str]:
    """FEATURES that have at least one value in `train` (leaving out the defender features unless `params` turns them on).

    A brand-new feature (e.g. sportsbook lines, only recorded going forward) is entirely missing from the
    training window at first, and HistGradientBoostingRegressor crashes on an all-missing column. Leave such
    features out until they have data; the model bundle records which features it was trained on.
    """
    skip = set() if params and params.get("defender_features") else set(DEFENDER_FEATURES)
    return [f for f in FEATURES if f not in skip and f in train and train[f].notna().any()]


def _score(params: dict, train: pd.DataFrame, hold: pd.DataFrame, target: str,
           focus: np.ndarray) -> tuple[float, float, object]:
    """(MAE on the focus players, MAE on everyone, fitted model)."""
    feats = usable_features(train, params)
    m = _model(params).fit(train[feats], train[target], sample_weight=_weights(train, params))
    pred = m.predict(hold[feats])
    return mean_absolute_error(hold[target][focus], pred[focus]), mean_absolute_error(hold[target], pred), m


def train_target(rows: pd.DataFrame, target: str, version: str) -> dict:
    cutoff = rows["date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train, hold = rows[rows["date"] <= cutoff], rows[rows["date"] > cutoff]

    # 1) Champion vs today's hyperparameter challenger on the most recent weeks, trained only on what came before
    #    them, judged on the players the bets are on. A challenger must beat the champion by MIN_IMPROVEMENT to take
    #    over. Feature sets (e.g. defender ratings) don't compete here: they're shadow-tested on live slates instead.
    focus, focus_label = _focus(hold)
    champ = tuning.champion(target)
    challengers = [c for c in [tuning.challenger(target, champ, today_et())] if c != champ]
    champ_focus, champ_mae, champ_model = _score(champ, train, hold, target, focus)
    judged = f"MAE on {focus_label} (n={int(focus.sum())})"
    params, point, mae, best_focus = champ, champ_model, champ_mae, champ_focus
    results = []
    for chall in challengers:
        c_focus, c_mae, c_model = _score(chall, train, hold, target, focus)
        results.append(f"{_diff(champ, chall)} {c_focus:.3f}")
        if c_focus < champ_focus * (1 - tuning.MIN_IMPROVEMENT) and c_focus < best_focus:
            params, point, mae, best_focus = chall, c_model, c_mae, c_focus
    if params is champ:
        note = f"champion kept: {judged} {champ_focus:.3f} vs challengers " + "; ".join(results)
    else:
        note = f"challenger won: {judged} {best_focus:.3f} < {champ_focus:.3f} with {_diff(champ, params)} (all: " + "; ".join(results) + ")"
    note = note[:512]
    feats = usable_features(train, params)
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
    w_all = _weights(rows, params)
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
    _train_shadow(rows, train, target, version, params, cal)
    return run


def _train_shadow(rows: pd.DataFrame, train: pd.DataFrame, target: str, version: str, params: dict, cal) -> None:
    """Train today's shadow model (the champion with the feature switch flipped) with the same recipe, for the
    week-long live comparison in app.ml.shadow. A failure here never affects the live model."""
    shadow = tuning.shadow_params(target, params)
    if shadow is None:  # switch locked by hand: no shadow test
        with SessionLocal() as s:
            s.query(ShadowModel).filter(ShadowModel.target == target).delete()
            s.commit()
        return
    try:
        feats = usable_features(train, shadow)
        probe = _model(shadow).fit(train[feats], train[target], sample_weight=_weights(train, shadow))
        model = _model(shadow, int(probe.n_iter_ * 1.1), early_stop=False).fit(
            rows[feats], rows[target], sample_weight=_weights(rows, shadow)
        )
        bundle = {"features": feats, "params": shadow, "interval": cal, "point": model}
        upsert(ShadowModel.__table__, [{"target": target, "version": f"{version}-shadow", "params": shadow,
                                        "trained_at": datetime.now(timezone.utc).replace(tzinfo=None), "artifact": dump(bundle)}])
    except Exception as e:  # noqa: BLE001
        print(f"{target}: shadow model failed: {e!r}", flush=True)


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
