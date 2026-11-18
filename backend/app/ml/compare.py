"""Before/after check for the defender features: walk forward through recent weeks, train the current champion
settings with the defender features off and on (only on games before each week), and compare the misses.

Nothing is stored; it only prints. Run it once on-court history has been loaded (python -m app.ingest.on_court).

python -m app.ml.compare [weeks=6]
"""

import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.config import TARGETS
from app.db import SessionLocal
from app.features.build import build_features, load_frames, training_rows
from app.ml import tuning
from app.ml.backtest import _fit_week
from app.models import PlayerProp


def main(weeks: int = 6) -> None:
    rows = training_rows(build_features(load_frames()))
    with SessionLocal() as s:
        lines = set(s.execute(select(PlayerProp.player_id, PlayerProp.game_id).distinct()).all())
    end = rows["date"].max()
    starts = [end - pd.Timedelta(days=7 * (weeks - i)) + timedelta(days=1) for i in range(weeks)]
    for target in TARGETS:
        champ = tuning.champion(target)
        settings = {"without": {**champ, "defender_features": 0}, "with": {**champ, "defender_features": 1}}
        err = {k: [] for k in settings}
        has_line = []
        for start in starts:
            history = rows[rows["date"] < start]
            week = rows[(rows["date"] >= start) & (rows["date"] < start + pd.Timedelta(days=7))]
            if week.empty:
                continue
            for name, params in settings.items():
                model, _, feats = _fit_week(history, target, params)
                err[name].append(np.abs(model.predict(week[feats]).clip(0) - week[target].to_numpy()))
            has_line.append(np.array([(p, g) in lines for p, g in zip(week["player_id"], week["game_id"])]))
            print(f"  {target} week of {start:%Y-%m-%d}: {len(week):,} player-games", flush=True)
        if not has_line:
            continue
        mask = np.concatenate(has_line)
        print(f"\n{target}: average miss over the last {weeks} weeks")
        for name in settings:
            e = np.concatenate(err[name])
            dk = f"{e[mask].mean():.4f} on {mask.sum():,} with DraftKings lines" if mask.any() else "no DraftKings lines"
            print(f"  {name:>7} defender features: {e.mean():.4f} on all {len(e):,} | {dk}")
        print(flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
