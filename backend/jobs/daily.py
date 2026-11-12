"""Daily pipeline. Run by Railway cron.

python -m jobs.daily            full run: ingest -> actuals -> evaluate yesterday -> self-tune + retrain -> predict today
python -m jobs.daily --refresh  pre-tip-off: refresh injury report and re-predict today (no retrain)
"""

import argparse
import logging
import sys
from datetime import timedelta

from sqlalchemy import text

from app.clock import today_et
from app.config import CURRENT_SEASON
from app.db import engine
from app.features.build import load_frames
from app.ingest.load import load_injuries, load_players, load_schedule, load_season_logs
from app.init_db import init_db
from app.ml.evaluate import evaluate_pending, update_bias_corrections
from app.ml.predict import predict_slate
from app.ml.train import train_all

log = logging.getLogger("daily")

RELOAD_DAYS = 3  # re-pull a few days back to pick up late stat corrections

FILL_ACTUALS = text("""
UPDATE predictions SET actual = (
    SELECT CASE predictions.target WHEN 'pts' THEN s.pts WHEN 'reb' THEN s.reb ELSE s.ast END
    FROM player_game_stats s WHERE s.player_id = predictions.player_id AND s.game_id = predictions.game_id
)
WHERE actual IS NULL AND EXISTS (
    SELECT 1 FROM player_game_stats s WHERE s.player_id = predictions.player_id AND s.game_id = predictions.game_id
)
""")


def run(refresh_only: bool = False) -> None:
    today = today_et()
    init_db()
    if not refresh_only:
        n_team, n_player = load_season_logs(CURRENT_SEASON, since=today - timedelta(days=RELOAD_DAYS))
        log.info("box scores: %d team-games, %d player-games", n_team, n_player)
        load_players(CURRENT_SEASON)
        load_schedule(CURRENT_SEASON)
        with engine.begin() as conn:
            log.info("filled actuals on %d predictions", conn.execute(FILL_ACTUALS).rowcount)
        for e in evaluate_pending(today):
            log.info("eval %s %s: n=%d MAE %.2f bias %+.2f within %.0f: %.0f%%", e["game_date"], e["target"], e["n_players"],
                     e["mae"], e["bias"], e["close_threshold"], 100 * e["within_close_pct"])
        log.info("bias corrections: %s", update_bias_corrections(today))

    n_inj, unmatched = load_injuries(today)
    log.info("injuries: %d stored, %d unmatched %s", n_inj, len(unmatched), unmatched[:10])

    if not refresh_only:
        for r in train_all():
            log.info("%s: MAE %.2f vs baseline %.2f promoted=%s", r["target"], r["mae"], r["baseline_mae"], r["promoted"])

    slate = predict_slate(load_frames(), today, persist=True)
    log.info("predicted %d players across %d games for %s", len(slate["predictions"]), len(slate["games"]), today)


if __name__ == "__main__":
    # stdout, so Railway doesn't tag every INFO line as an error (it treats stderr as errors).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="injury refresh + re-predict only")
    run(parser.parse_args().refresh)
