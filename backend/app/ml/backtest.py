"""Backfill accuracy history for games played before live tracking began.

Walk-forward, one week at a time: for each week since the season opener, train only on games before that week,
predict every player-game in it, store the results as model_version "backtest-YYYYMMDD", and grade them with the
same daily evaluation the live system uses.

Caveat: a backtest knows who actually played (players who sat simply have no row, and missing-teammate features
use actual absences). Live predictions have to rely on the pre-game injury report, so backtest accuracy is a little
optimistic compared with live.

python -m app.ml.backtest [start YYYY-MM-DD] [end YYYY-MM-DD]
"""

import sys
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import delete

from app.clock import today_et
from app.config import CURRENT_SEASON, TARGETS
from app.db import engine, upsert
from app.features.build import FEATURES, build_features, load_frames, training_rows
from app.ml import intervals, tuning
from app.ml.evaluate import evaluate_date
from app.ml.predict import CONTEXT_COLS, _num
from app.ml.train import HOLDOUT_DAYS, _model, _weights
from app.models import Prediction

VERSION_PREFIX = "backtest-"


def _fit_week(history: pd.DataFrame, target: str) -> tuple[object, dict]:
    """Same recipe as the daily retrain: early-stopped fit, range calibration on the last 3 weeks, full refit."""
    params = tuning.champion(target)
    cutoff = history["date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train, hold = history[history["date"] <= cutoff], history[history["date"] > cutoff]
    probe = _model(params).fit(train[FEATURES], train[target], sample_weight=_weights(train["date"], params["half_life_days"]))
    cal = intervals.fit(hold, target, probe.predict(hold[FEATURES]))
    final = _model(params, int(probe.n_iter_ * 1.1), early_stop=False).fit(
        history[FEATURES], history[target], sample_weight=_weights(history["date"], params["half_life_days"])
    )
    return final, cal


def run(start: date | None = None, end: date | None = None) -> None:
    rows = training_rows(build_features(load_frames()))
    season = rows[rows["season"] == CURRENT_SEASON]
    start = start or season["date"].min().date()
    end = end or today_et() - timedelta(days=1)

    week = start
    while week <= end:
        week_end = min(week + timedelta(days=6), end)
        history = rows[rows["date"] < pd.Timestamp(week)]
        games = rows[(rows["date"] >= pd.Timestamp(week)) & (rows["date"] <= pd.Timestamp(week_end))]
        if games.empty:
            week += timedelta(days=7)
            continue
        version = f"{VERSION_PREFIX}{week:%Y%m%d}"
        records = []
        for target in TARGETS:
            model, cal = _fit_week(history, target)
            pred = model.predict(games[FEATURES]).clip(0)
            lo, hi = intervals.apply(games, target, pred, cal)
            for r, p, l, h in zip(games.itertuples(), pred, lo, hi):
                ctx = {c: _num(getattr(r, c)) for c in CONTEXT_COLS}
                records.append({
                    "player_id": int(r.player_id), "game_id": r.game_id, "target": target, "game_date": r.date.date(),
                    "model_version": version, "predicted": float(p), "p10": float(l), "p90": float(h),
                    "baseline": _num(getattr(r, f"{target}_l10")), "actual": float(getattr(r, target)),
                    "context": {**ctx, "injury_status": None, "raw": round(float(p), 2), "backtest": True},
                })
            print(f"{version} {target}: trained on {len(history):,} games, predicted {len(games):,}", flush=True)
        upsert(Prediction.__table__, records)
        for d in sorted(games["date"].dt.date.unique()):
            for e in evaluate_date(d):
                print(f"  {d} {e['target']}: n={e['n_players']} MAE {e['mae']:.2f} (L10 {e['baseline_mae']:.2f}) "
                      f"within ±{e['close_threshold']:g}: {e['within_close_pct']:.0%}", flush=True)
        week += timedelta(days=7)


def clear() -> int:
    """Remove all backtest predictions (their daily evaluations are recomputed on the next evaluate)."""
    with engine.begin() as conn:
        return conn.execute(delete(Prediction).where(Prediction.model_version.like(f"{VERSION_PREFIX}%"))).rowcount


if __name__ == "__main__":
    args = [date.fromisoformat(a) for a in sys.argv[1:3]]
    run(*args)
