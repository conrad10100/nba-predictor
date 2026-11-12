"""Score yesterday's predictions against what actually happened, and derive the live bias correction.

python -m app.ml.evaluate            evaluate every date that has predictions with results
python -m app.ml.evaluate 2026-11-12 evaluate one date
"""

import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.clock import today_et
from app.db import SessionLocal, engine, upsert
from app.features.build import TARGETS
from app.models import DailyEvaluation, ModelConfig, Prediction

# "Close" and "wide" miss thresholds per stat, e.g. within 3 / 5 points.
THRESHOLDS = {"pts": (3.0, 5.0), "reb": (2.0, 3.0), "ast": (2.0, 3.0)}
REEVALUATE_DAYS = 3  # stat corrections can land a day or two late
BIAS_WINDOW_DAYS = 14
BIAS_SHRINK_N = 300  # with n live predictions, trust the observed bias n / (n + 300)
BIAS_CAP = {"pts": 1.5, "reb": 0.75, "ast": 0.75}

SEGMENTS = {
    "b2b": lambda c: c.get("b2b") == 1,
    "rested": lambda c: c.get("b2b") == 0,
    "same_conference": lambda c: c.get("same_conference") == 1,
    "cross_conference": lambda c: c.get("same_conference") == 0,
    "home": lambda c: c.get("is_home") == 1,
    "away": lambda c: c.get("is_home") == 0,
    "starters": lambda c: (c.get("min_l10") or 0) >= 28,
    "bench": lambda c: (c.get("min_l10") or 0) < 28,
    "teammate_out": lambda c: (c.get("missing_teammate_pts") or 0) >= 10,
    "injury_listed": lambda c: bool(c.get("injury_status")),
}


def _load(on: date) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(select(Prediction).where(Prediction.game_date == on), conn)


def evaluate_date(on: date) -> list[dict]:
    preds = _load(on)
    rows = []
    for target in TARGETS:
        all_t = preds[preds["target"] == target]
        df = all_t[all_t["actual"].notna()]
        if df.empty:
            continue
        err = df["predicted"] - df["actual"]
        abs_err = err.abs()
        close, wide = THRESHOLDS[target]
        has_base = df["baseline"].notna()
        base_err = (df.loc[has_base, "baseline"] - df.loc[has_base, "actual"]).abs()
        contexts = df["context"].apply(lambda c: c or {})
        segments = {"did_not_play": {"n": int(len(all_t) - len(df))}}
        for name, test in SEGMENTS.items():
            mask = contexts.apply(test).to_numpy()
            if mask.any():
                segments[name] = {"n": int(mask.sum()), "mae": round(float(abs_err[mask].mean()), 3), "bias": round(float(err[mask].mean()), 3)}
        rows.append({
            "game_date": on, "target": target, "model_version": df["model_version"].mode().iat[0],
            "n_players": len(df), "mae": float(abs_err.mean()), "rmse": float(np.sqrt((err**2).mean())), "bias": float(err.mean()),
            "baseline_mae": float(base_err.mean()) if has_base.any() else None,
            "beat_baseline_pct": float((abs_err[has_base] < base_err).mean()) if has_base.any() else None,
            "close_threshold": close,
            "within_close_pct": float((abs_err <= close).mean()),
            "within_wide_pct": float((abs_err <= wide).mean()),
            "interval_coverage": float(((df["actual"] >= df["p10"]) & (df["actual"] <= df["p90"])).mean()),
            "segments": segments,
        })
    upsert(DailyEvaluation.__table__, rows)
    return rows


def pending_dates(today: date | None = None) -> list[date]:
    """Dates with graded predictions that haven't been evaluated yet, plus the last few days (stat corrections)."""
    today = today or today_et()
    with SessionLocal() as s:
        graded = set(s.scalars(select(Prediction.game_date).where(Prediction.actual.is_not(None), Prediction.game_date < today).distinct()))
        done = set(s.scalars(select(DailyEvaluation.game_date).distinct()))
    recent = {d for d in graded if d >= today - timedelta(days=REEVALUATE_DAYS)}
    return sorted((graded - done) | recent)


def evaluate_pending(today: date | None = None) -> list[dict]:
    return [row for d in pending_dates(today) for row in evaluate_date(d)]


def update_bias_corrections(today: date | None = None) -> dict[str, float]:
    """Correction = shrunk mean of (actual - raw model output) over the last two weeks of live games.

    Uses the raw (uncorrected) prediction stored in context, so the correction doesn't feed back on itself.
    """
    today = today or today_et()
    with engine.connect() as conn:
        df = pd.read_sql(
            select(Prediction.target, Prediction.predicted, Prediction.actual, Prediction.context)
            .where(Prediction.actual.is_not(None), Prediction.game_date >= today - timedelta(days=BIAS_WINDOW_DAYS)),
            conn,
        )
    out = {}
    with SessionLocal() as s:
        for target in TARGETS:
            t = df[df["target"] == target]
            if t.empty:
                continue
            raw = t.apply(lambda r: (r["context"] or {}).get("raw", r["predicted"]), axis=1)
            residual = float((t["actual"] - raw).mean())
            n = len(t)
            correction = float(np.clip(residual * n / (n + BIAS_SHRINK_N), -BIAS_CAP[target], BIAS_CAP[target]))
            cfg = s.get(ModelConfig, target)
            if cfg is None:
                continue  # created by the first training run
            cfg.bias_correction = round(correction, 3)
            cfg.note = f"bias from {n} live predictions over {BIAS_WINDOW_DAYS}d: residual {residual:+.2f}"
            out[target] = correction
        s.commit()
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1:
        rows = evaluate_date(date.fromisoformat(sys.argv[1]))
    else:
        rows = evaluate_pending()
    for r in rows:
        print(f"{r['game_date']} {r['target']}: n={r['n_players']} MAE {r['mae']:.2f} (baseline {r['baseline_mae'] or float('nan'):.2f}) "
              f"bias {r['bias']:+.2f} within ±{r['close_threshold']:g}: {r['within_close_pct']:.0%} range hit {r['interval_coverage']:.0%}")
