"""Daily pipeline. Run by Railway cron.

python -m jobs.daily            full run: ingest -> actuals -> evaluate yesterday -> self-tune + retrain -> predict today
python -m jobs.daily --refresh  pre-tip-off: refresh injury report and re-predict today (no retrain)
"""

import argparse
import logging
import sys
from datetime import timedelta

from sqlalchemy import func, select, text

from app.clock import today_et
from app.config import CURRENT_SEASON
from app.db import engine
from app.features.build import load_frames
from app.ingest.load import (
    dates_missing_props, load_game_lines, load_injuries, load_player_props, load_players, load_schedule, load_season_logs,
)
from app.init_db import init_db
from app.ml.evaluate import evaluate_pending, update_bias_corrections
from app.ml.predict import predict_slate
from app.ml.train import train_all
from app.models import Game

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


def run(refresh_only: bool = False) -> bool:
    """Returns False if the box-score download failed (everything else still runs)."""
    today = today_et()
    init_db()
    # stats.nba.com sometimes times out or blocks cloud IPs. Don't lose the whole day to it: still store injuries,
    # lines and props and predict today's slate with the current models, then report the failure. The 5pm refresh
    # tries the download again if last night's games are still missing.
    ingest_ok = True
    if not refresh_only or _missing_last_night(today):
        ingest_ok = _ingest(today)

    if not refresh_only:
        with engine.begin() as conn:
            log.info("filled actuals on %d predictions", conn.execute(FILL_ACTUALS).rowcount)
        for e in evaluate_pending(today):
            log.info("eval %s %s: n=%d MAE %.2f bias %+.2f within %.0f: %.0f%%", e["game_date"], e["target"], e["n_players"],
                     e["mae"], e["bias"], e["close_threshold"], 100 * e["within_close_pct"])
        log.info("bias corrections: %s", update_bias_corrections(today))

    n_inj, unmatched = load_injuries(today)
    log.info("injuries: %d stored, %d unmatched %s", n_inj, len(unmatched), unmatched[:10])
    log.info("game lines: %d stored", load_game_lines([today, today + timedelta(days=1)]))
    # Today's lines before tip-off; on the morning run also yesterday's closing lines and any season dates still missing.
    # Props only feed the Performance page, so a failure here shouldn't stop the retrain and today's predictions.
    try:
        prop_dates = [today] if refresh_only else sorted({today - timedelta(days=1), today, *dates_missing_props(CURRENT_SEASON, today)})
        n_props, unmatched_props = load_player_props(prop_dates)
        log.info("player props: %d stored for %d dates, %d unmatched %s", n_props, len(prop_dates), len(unmatched_props), unmatched_props[:10])
    except Exception:
        log.exception("player props: failed, continuing without them")

    if not refresh_only and ingest_ok:  # no new box scores means nothing new to learn from
        for r in train_all():
            log.info("%s: MAE %.2f vs baseline %.2f promoted=%s", r["target"], r["mae"], r["baseline_mae"], r["promoted"])

    slate = predict_slate(load_frames(), today, persist=True)
    log.info("predicted %d players across %d games for %s", len(slate["predictions"]), len(slate["games"]), today)
    return ingest_ok


def _ingest(today) -> bool:
    try:
        n_team, n_player = load_season_logs(CURRENT_SEASON, since=today - timedelta(days=RELOAD_DAYS))
        log.info("box scores: %d team-games, %d player-games", n_team, n_player)
        load_players(CURRENT_SEASON)
        load_schedule(CURRENT_SEASON)
        return True
    except Exception:
        log.exception("box scores: download from stats.nba.com failed; continuing with the data already stored")
        return False


def _missing_last_night(today) -> bool:
    """True if yesterday had games that aren't loaded as final yet."""
    with engine.connect() as conn:
        n = conn.execute(select(func.count()).select_from(Game).where(Game.date == today - timedelta(days=1), Game.status != "final")).scalar()
    return bool(n)


if __name__ == "__main__":
    # stdout, so Railway doesn't tag every INFO line as an error (it treats stderr as errors).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="injury refresh + re-predict only")
    if not run(parser.parse_args().refresh):
        sys.exit(1)  # so Railway still shows the run as failed
