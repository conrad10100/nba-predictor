"""Per-game defensive matchups (who guarded whom) from stats.nba.com's boxscorematchupsv3.

One request per game, so history is filled in gradually: each daily run loads the last few nights plus a batch of
older games, and `python -m app.ingest.matchups [max_games]` loads more by hand. Games already requested are
skipped, so any run can be stopped and resumed.
"""

import logging
import sys
import time

import pandas as pd
from sqlalchemy import select

from app.config import BACKFILL_SEASONS
from app.db import SessionLocal, upsert
from app.ingest import nba_client
from app.models import Game, MatchupFetch, PlayerGameStat, PlayerMatchup

log = logging.getLogger(__name__)

TRACKING_FIRST_SEASON = "2017-18"  # the NBA's matchup tracking starts here
DAILY_BACKFILL_GAMES = 150  # older games loaded per daily run (~1 request/second, a few minutes)
MAX_CONSECUTIVE_FAILURES = 3  # stats.nba.com blocking us: stop instead of waiting out every timeout


def pending_games(limit: int | None = None) -> list[str]:
    """Final games without a matchup request yet, newest first (recent games matter most for the features)."""
    seasons = [s for s in BACKFILL_SEASONS if s >= TRACKING_FIRST_SEASON]
    with SessionLocal() as s:
        q = (
            select(Game.id).outerjoin(MatchupFetch, MatchupFetch.game_id == Game.id)
            .where(Game.status == "final", Game.season.in_(seasons), MatchupFetch.game_id.is_(None))
            .order_by(Game.date.desc(), Game.id)
        )
        return list(s.scalars(q.limit(limit) if limit else q))


def _seconds(v) -> float:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0.0
    if isinstance(v, (int, float)):
        return float(v) * 60
    m, _, sec = str(v).partition(":")
    return float(m or 0) * 60 + float(sec or 0)


def parse(game_id: str, df: pd.DataFrame) -> list[dict]:
    rows = {}
    for r in df.itertuples():
        off_team = int(r.teamId)
        def_team = int(r.awayTeamId) if off_team == int(r.homeTeamId) else int(r.homeTeamId)
        key = (int(r.personIdOff), int(r.personIdDef))
        if key[0] == key[1] or not key[0] or not key[1]:
            continue
        rows[key] = {
            "game_id": game_id, "off_player_id": key[0], "def_player_id": key[1], "off_team_id": off_team, "def_team_id": def_team,
            "partial_poss": float(r.partialPossessions or 0), "seconds": _seconds(r.matchupMinutes),
            "pts": int(r.playerPoints or 0), "fgm": int(r.matchupFieldGoalsMade or 0), "fga": int(r.matchupFieldGoalsAttempted or 0),
            "fg3m": int(r.matchupThreePointersMade or 0), "fg3a": int(r.matchupThreePointersAttempted or 0),
            "fta": int(r.matchupFreeThrowsAttempted or 0), "ast": int(r.matchupAssists or 0), "tov": int(r.matchupTurnovers or 0),
        }
    return list(rows.values())


def _orient(rows: list[dict], team_of: dict[int, int]) -> list[dict]:
    """Make sure off/def are the right way round, checked against the box score's teams (a guard against the
    endpoint's layout changing): if most offensive players turn out to be on the 'defending' team, swap the roles."""
    known = [r for r in rows if r["off_player_id"] in team_of]
    flipped = sum(team_of[r["off_player_id"]] == r["def_team_id"] for r in known)
    if not known or flipped <= len(known) / 2:
        return rows
    log.warning("matchups: offense/defense reversed in the response, swapping")
    return [{**r, "off_player_id": r["def_player_id"], "def_player_id": r["off_player_id"],
             "off_team_id": r["def_team_id"], "def_team_id": r["off_team_id"]} for r in rows]


def load_matchups(game_ids: list[str], time_budget_s: float | None = None) -> tuple[int, int]:
    """Request matchups for the listed games. Returns (games loaded, matchup rows stored)."""
    start, games, n_rows, failures = time.monotonic(), 0, 0, 0
    for gid in game_ids:
        if time_budget_s is not None and time.monotonic() - start > time_budget_s:
            break
        try:
            df = nba_client.box_score_matchups(gid)
            rows = parse(gid, df) if df is not None and not df.empty else []
            if rows:
                with SessionLocal() as s:
                    team_of = dict(s.execute(select(PlayerGameStat.player_id, PlayerGameStat.team_id).where(PlayerGameStat.game_id == gid)).all())
                rows = _orient(rows, team_of)
        except (IndexError, KeyError, TypeError, ValueError):
            rows = []  # the game has no tracking data; remember that so it isn't requested again
        except Exception:
            failures += 1
            log.warning("matchups %s: request failed", gid, exc_info=failures == 1)
            if failures >= MAX_CONSECUTIVE_FAILURES:
                log.warning("matchups: %d failures in a row, stopping for this run", failures)
                break
            continue
        failures = 0
        upsert(PlayerMatchup.__table__, rows)
        upsert(MatchupFetch.__table__, [{"game_id": gid, "n_rows": len(rows)}])
        games, n_rows = games + 1, n_rows + len(rows)
    return games, n_rows


if __name__ == "__main__":
    from app.init_db import init_db

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    init_db()
    todo = pending_games(int(sys.argv[1]) if len(sys.argv) > 1 else None)
    print(f"{len(todo)} games to load", flush=True)
    done = 0
    for i in range(0, len(todo), 50):
        g, n = load_matchups(todo[i : i + 50])
        done += g
        print(f"  {done}/{len(todo)} games, {n} matchups in this batch", flush=True)
        if g == 0:
            break
