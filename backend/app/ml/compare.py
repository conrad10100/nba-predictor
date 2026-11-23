"""Before/after checks: walk forward through recent weeks, train two versions of the current champion settings
(only on games before each week), and compare the misses and the hit rate against the DraftKings line.

    python -m app.ml.compare [weeks=6]           defender features off vs on
    python -m app.ml.compare model [weeks=6]     HistGradientBoosting vs XGBoost (same settings otherwise)

Nothing is stored; it only prints. Ratings for past seasons load on the first daily run (or python -m app.ingest.defense).
"""

import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.config import TARGETS
from app.db import SessionLocal
from app.features.build import build_features, load_frames, training_rows
from app.ml import tuning, xgb
from app.ml.backtest import _fit_week
from app.models import PlayerProp

COMPARISONS = {
    "features": ("defender features", lambda champ: {"without": {**champ, "defender_features": 0},
                                                      "with": {**champ, "defender_features": 1}}),
    "model": ("algorithm", lambda champ: {"hgb": {**champ, "model": "hgb"}, "xgboost": {**champ, "model": "xgboost"}}),
}


def hit_rate(pred: np.ndarray, actual: np.ndarray, line: np.ndarray) -> tuple[float, int]:
    """Share of over/under calls (model above the line = over) that were right, leaving out pushes and
    predictions sitting exactly on the line. Returns (rate, number of graded calls)."""
    graded = ~np.isnan(line) & (actual != line) & (pred != line)
    if not graded.any():
        return float("nan"), 0
    right = (pred[graded] > line[graded]) == (actual[graded] > line[graded])
    return float(right.mean()), int(graded.sum())


def main(kind: str = "features", weeks: int = 6) -> None:
    label, make = COMPARISONS[kind]
    if kind == "model" and not xgb.AVAILABLE:
        sys.exit("XGBoost isn't importable here (on a Mac: brew install libomp)")
    rows = training_rows(build_features(load_frames()))
    with SessionLocal() as s:
        props = pd.DataFrame(s.execute(select(PlayerProp.player_id, PlayerProp.game_id, PlayerProp.target, PlayerProp.line)).all(),
                             columns=["player_id", "game_id", "target", "line"])
    end = rows["date"].max()
    starts = [end - pd.Timedelta(days=7 * (weeks - i)) + timedelta(days=1) for i in range(weeks)]
    for target in TARGETS:
        settings = make(tuning.champion(target))
        tlines = props[props["target"] == target].set_index(["player_id", "game_id"])["line"]
        preds = {k: [] for k in settings}
        actual, line = [], []
        for start in starts:
            history = rows[rows["date"] < start]
            week = rows[(rows["date"] >= start) & (rows["date"] < start + pd.Timedelta(days=7))]
            if week.empty:
                continue
            for name, params in settings.items():
                model, _, feats = _fit_week(history, target, params)
                preds[name].append(model.predict(week[feats]).clip(0))
            actual.append(week[target].to_numpy(dtype=float))
            idx = pd.MultiIndex.from_arrays([week["player_id"], week["game_id"]])
            line.append(tlines.reindex(idx).to_numpy(dtype=float))
            print(f"  {target} week of {start:%Y-%m-%d}: {len(week):,} player-games", flush=True)
        if not actual:
            continue
        y, ln = np.concatenate(actual), np.concatenate(line)
        has_line = ~np.isnan(ln)
        print(f"\n{target}: {label}, last {weeks} weeks")
        for name in settings:
            p = np.concatenate(preds[name])
            e = np.abs(p - y)
            dk = f"{e[has_line].mean():.4f} on {has_line.sum():,} with DraftKings lines" if has_line.any() else "no DraftKings lines"
            rate, n = hit_rate(p, y, ln)
            hits = f" | vs line {rate:.1%} of {n:,}" if n else ""
            print(f"  {name:>8}: MAE {e.mean():.4f} on all {len(e):,} | {dk}{hits}")
        if has_line.any():
            print(f"  DraftKings line itself: MAE {np.abs(ln[has_line] - y[has_line]).mean():.4f}")
        print(flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    kind = args.pop(0) if args and args[0] in COMPARISONS else "features"
    main(kind, int(args[0]) if args else 6)
