"""One-time historical load: python -m app.ingest.backfill [season ...]"""

import sys

from app.config import BACKFILL_SEASONS, CURRENT_SEASON
from app.ingest.load import load_injuries, load_players, load_schedule, load_season_logs
from app.init_db import init_db


def main(seasons: list[str]) -> None:
    init_db()
    for season in seasons:
        n_team, n_player = load_season_logs(season)
        print(f"{season}: {n_team} team-games, {n_player} player-games")
        load_players(season)
    load_schedule(CURRENT_SEASON)
    n_inj, unmatched = load_injuries()
    print(f"injuries: {n_inj} stored, {len(unmatched)} unmatched")


if __name__ == "__main__":
    main(sys.argv[1:] or BACKFILL_SEASONS)
