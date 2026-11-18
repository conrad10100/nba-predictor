"""Player defensive ratings (DBPM) from Basketball-Reference: one page per season.

The daily run stores today's snapshot of the current season; past seasons are stored once as final ratings. So
features never use a rating published after the game they predict (see app.features.defenders).

python -m app.ingest.defense      store any missing past seasons and today's current-season snapshot
"""

import logging
import sys
from collections import defaultdict
from datetime import date

from sqlalchemy import select

from app.clock import today_et
from app.config import BACKFILL_SEASONS, CURRENT_SEASON
from app.db import SessionLocal, upsert
from app.features import defenders
from app.features.defenders import previous_season
from app.ingest import nba_client
from app.models import DefenderRating, Player, Team

log = logging.getLogger(__name__)


def season_end(season: str) -> date:
    return defenders.season_end(season).date()


def load_season(season: str, as_of: date) -> tuple[int, list[str]]:
    """Store one season's ratings as published on `as_of`. Returns (rows stored, names that matched no player)."""
    table = nba_client.bbref_defense(season)
    with SessionLocal() as s:
        players = s.execute(select(Player.id, Player.name, Player.team_id)).all()
        abbr = {t.id: t.abbr for t in s.scalars(select(Team))}
    by_name: dict[str, list[tuple[int, str | None]]] = defaultdict(list)
    for pid, name, tid in players:
        by_name[nba_client.normalize_name(name)].append((pid, abbr.get(tid)))
    rows, unmatched = {}, []
    for r in table:
        cands = by_name.get(nba_client.normalize_name(r["name"]), [])
        pid = cands[0][0] if len(cands) == 1 else next((p for p, a in cands if a == r["team"]), None)
        if pid is None:
            unmatched.append(r["name"])
            continue
        rows[pid] = {"player_id": pid, "season": season, "as_of": as_of, "dbpm": r["dbpm"], "minutes": r["minutes"]}
    return upsert(DefenderRating.__table__, rows.values()), unmatched


def missing_final_seasons() -> list[str]:
    """Past seasons whose final ratings aren't stored yet, including the one before the first backfilled season
    (its ratings are the prior for that season's games)."""
    seasons = sorted({previous_season(min(BACKFILL_SEASONS)), *BACKFILL_SEASONS} - {CURRENT_SEASON})
    with SessionLocal() as s:
        stored = s.execute(select(DefenderRating.season, DefenderRating.as_of).distinct()).all()
    have = {season for season, as_of in stored if as_of == season_end(season)}
    return [x for x in seasons if x not in have and x < CURRENT_SEASON]


def run(today: date | None = None) -> dict:
    today = today or today_et()
    result = {}
    for season in missing_final_seasons():
        n, unmatched = load_season(season, season_end(season))
        result[season] = (n, unmatched)
    result[CURRENT_SEASON] = load_season(CURRENT_SEASON, today)
    return result


if __name__ == "__main__":
    from app.init_db import init_db

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    init_db()
    for season, (n, unmatched) in run().items():
        print(f"{season}: {n} players stored, {len(unmatched)} unmatched {unmatched[:10]}")
