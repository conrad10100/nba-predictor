"""Daily pipeline. Run by Railway cron.

python -m jobs.daily            full run: ingest -> actuals -> evaluate yesterday -> self-tune + retrain -> predict today
python -m jobs.daily --refresh  injury update: refresh injuries + lines and re-predict today (no retrain)
python -m jobs.daily --refresh --before-tip 30
                                the same, but only if a game tips off within 30 minutes (else exit right away), so a
                                frequent cron gives every game a final pick taken shortly before its own tip-off
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select, text

from app.clock import today_et
from app.config import CURRENT_SEASON
from app.db import engine
from app.features.build import load_frames
from app.ingest.load import (
    dates_missing_props, load_espn_logs, load_game_lines, load_injuries, load_player_props, load_players, load_schedule,
    load_season_logs,
)
from app.ingest import defense
from app.init_db import init_db
from app.ml.evaluate import evaluate_pending, update_bias_corrections
from app.ml.predict import predict_slate
from app.ml.shadow import weekly_decision
from app.ml.snapshots import snapshot_pregame
from app.ml.train import train_all
from app.models import Game

log = logging.getLogger("daily")

RELOAD_DAYS = 3  # re-pull a few days back to pick up late stat corrections
GAP_FILL_DAYS = 14  # how far back to look for played games the stats.nba.com log left out

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
    # lines and props and predict today's slate with the current models, then report the failure. The evening refreshes
    # try the download again if last night's games are still missing.
    ingest_ok = True
    if not refresh_only or _missing_last_night(today):
        ingest_ok = _ingest(today)
        # Right away, so an evening refresh that picks up a late game grades its picks too (not void till morning).
        with engine.begin() as conn:
            log.info("filled actuals on %d predictions", conn.execute(FILL_ACTUALS).rowcount)

    if not refresh_only:
        # Defensive ratings (Basketball-Reference DBPM): today's snapshot, plus any past season not stored yet.
        # Only feeds the defender features, so a failure here never stops the run.
        try:
            for season, (n, unmatched) in defense.run(today).items():
                log.info("defender ratings %s: %d stored, %d unmatched %s", season, n, len(unmatched), unmatched[:10])
        except Exception:
            log.exception("defender ratings: failed, continuing without new ones")
        for e in evaluate_pending(today):
            log.info("eval %s %s: n=%d MAE %.2f bias %+.2f within %.0f: %.0f%%", e["game_date"], e["target"], e["n_players"],
                     e["mae"], e["bias"], e["close_threshold"], 100 * e["within_close_pct"])
        log.info("bias corrections: %s", update_bias_corrections(today))
        # Weekly shadow verdict (before retraining, so a switch applies to today's model).
        try:
            for note in weekly_decision(today):
                log.info("shadow test: %s", note)
        except Exception:
            log.exception("shadow test: weekly comparison failed")

    n_inj, unmatched = load_injuries(today)
    log.info("injuries: %d stored, %d unmatched %s", n_inj, len(unmatched), unmatched[:10])
    log.info("game lines: %d stored", load_game_lines([today, today + timedelta(days=1)]))
    # Today's lines before tip-off; on the morning run also yesterday's closing lines and any season dates still missing.
    # Props only feed the Performance page, so a failure here shouldn't stop the retrain and today's predictions.
    pregame_games: set[str] = set()
    try:
        prop_dates = [today] if refresh_only else sorted({today - timedelta(days=1), today, *dates_missing_props(CURRENT_SEASON, today)})
        n_props, unmatched_props, pregame_games = load_player_props(prop_dates)
        log.info("player props: %d stored for %d dates, %d unmatched %s", n_props, len(prop_dates), len(unmatched_props), unmatched_props[:10])
    except Exception:
        log.exception("player props: failed, continuing without them")

    if not refresh_only and ingest_ok:  # no new box scores means nothing new to learn from
        for r in train_all():
            log.info("%s: MAE %.2f vs baseline %.2f promoted=%s", r["target"], r["mae"], r["baseline_mae"], r["promoted"])

    slate = predict_slate(load_frames(), today, persist=True)
    log.info("predicted %d players across %d games for %s", len(slate["predictions"]), len(slate["games"]), today)

    # Freeze tonight's picks (line + prediction) for games that haven't started; the top 5 is ranked on these.
    try:
        phases = ("final",) if refresh_only else ("morning", "final")
        log.info("pick snapshots (%s): %d frozen for %d games not yet started", "/".join(phases),
                 snapshot_pregame(pregame_games, phases), len(pregame_games))
    except Exception:
        log.exception("pick snapshots: failed")
    return ingest_ok


def _ingest(today) -> bool:
    try:
        n_team, n_player = load_season_logs(CURRENT_SEASON, since=today - timedelta(days=RELOAD_DAYS))
        log.info("box scores: %d team-games, %d player-games", n_team, n_player)
        _fill_gaps_from_espn(today)
    except Exception:
        log.exception("box scores: stats.nba.com failed; falling back to ESPN box scores")
        # Same host as the box scores, so don't wait out the same timeouts for rosters and the schedule
        # (the schedule is already stored through the end of the season).
        try:
            dates = [today - timedelta(days=i) for i in range(RELOAD_DAYS, 0, -1)]
            n_team, n_player, unmatched = load_espn_logs(CURRENT_SEASON, dates)
            log.info("box scores (ESPN): %d team-games, %d player-games, %d unmatched %s", n_team, n_player, len(unmatched), unmatched[:10])
            return True
        except Exception:
            log.exception("box scores: ESPN fallback failed too; continuing with the data already stored")
            return False
    try:
        load_players(CURRENT_SEASON)
        load_schedule(CURRENT_SEASON)
    except Exception:
        log.exception("rosters/schedule: stats.nba.com failed; keeping what's stored")
    return True


def _fill_gaps_from_espn(today) -> None:
    """stats.nba.com's game log sometimes leaves out a finished game for days (MIL@MIA on 2026-11-18, DEN@NOP on
    2026-11-14), which leaves it "scheduled" and its picks pending. Load just those games from ESPN instead.
    ESPN only returns completed games, so a postponed one simply stays scheduled."""
    with engine.connect() as conn:
        missing = conn.execute(select(Game.id, Game.date).where(
            Game.season == CURRENT_SEASON, Game.status != "final",
            Game.date < today, Game.date >= today - timedelta(days=GAP_FILL_DAYS))).all()
    if not missing:
        return
    try:
        n_team, n_player, unmatched = load_espn_logs(CURRENT_SEASON, sorted({d for _, d in missing}), only={g for g, _ in missing})
        log.info("box scores (ESPN, %d games missing from stats.nba.com): %d team-games, %d player-games, %d unmatched %s",
                 len(missing), n_team, n_player, len(unmatched), unmatched[:10])
    except Exception:
        log.exception("box scores: ESPN gap fill failed; those games stay pending until a later run")


LATE_TIP_GRACE = timedelta(minutes=20)  # a game still "pre" this long after its scheduled tip counts as starting now


def _tipping_soon(tipoffs: list[dict], within: timedelta, now: datetime | None = None) -> list[dict]:
    """Games that haven't started and are scheduled to tip off within `within` (or are a little late)."""
    now = now or datetime.now(timezone.utc)
    return [g for g in tipoffs if g["state"] == "pre" and -LATE_TIP_GRACE <= g["start"] - now <= within]


def _lock_started(today, tipoffs: list[dict]) -> None:
    """Settle the top 5 for games that have tipped off since the last run (app.ml.top5). Runs before this run
    re-freezes any pick, so it sees the board as it stood at their tip-off. Never stops the refresh."""
    started = [g for g in tipoffs if g["state"] != "pre"]
    if not started:
        return
    try:
        from app.ml import top5
        from app.models import Team

        init_db()
        with engine.connect() as conn:
            abbr_to_id = dict(conn.execute(select(Team.abbr, Team.id)).all())
            by_matchup = {(h, a): gid for gid, h, a in conn.execute(
                select(Game.id, Game.home_team_id, Game.away_team_id).where(Game.date == today)).all()}
        ids = {by_matchup.get((abbr_to_id.get(g["home"]), abbr_to_id.get(g["away"]))) for g in started} - {None}
        n_games, n_picks = top5.lock_started(today, ids)
        if n_games:
            log.info("top 5 settled at tip-off: %d games just started, %d of their picks locked in", n_games, n_picks)
    except Exception:
        log.exception("top 5 lock: failed, continuing")


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
    parser.add_argument("--before-tip", type=int, metavar="MINUTES",
                        help="with --refresh: only run if a game tips off within this many minutes")
    args = parser.parse_args()
    if args.before_tip is not None and not args.refresh:
        parser.error("--before-tip needs --refresh")
    if args.refresh:
        from app.ingest import nba_client

        tipoffs = nba_client.espn_tipoffs(today_et())
        _lock_started(today_et(), tipoffs)
    if args.before_tip is not None:
        soon = _tipping_soon(tipoffs, timedelta(minutes=args.before_tip))
        if not soon:
            log.info("no game tips off in the next %d minutes; nothing to refresh", args.before_tip)
            sys.exit(0)
        log.info("tipping off soon: %s", ", ".join(f"{g['away']}@{g['home']} {g['start']:%H:%M}Z" for g in soon))
    if not run(args.refresh):
        sys.exit(1)  # so Railway still shows the run as failed
