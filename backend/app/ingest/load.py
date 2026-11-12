"""Transform raw NBA data into DB rows and upsert them. Shared by backfill and the daily job."""

from datetime import date, datetime

import pandas as pd
from sqlalchemy import select

from app.db import SessionLocal, upsert
from app.ingest import nba_client
from app.models import Game, Injury, Player, PlayerGameStat, TeamGameStat

REGULAR_SEASON_PREFIX = "002"


def _possessions(df: pd.DataFrame) -> pd.Series:
    return df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]


def load_schedule(season: str) -> int:
    """Insert scheduled (not yet played) games so upcoming slates, rest days and B2Bs are known."""
    sched = nba_client.season_schedule(season)
    sched = sched[sched["gameId"].str.startswith(REGULAR_SEASON_PREFIX)]
    with SessionLocal() as s:
        final_ids = set(s.scalars(select(Game.id).where(Game.season == season, Game.status == "final")))
    rows = [
        {
            "id": r.gameId,
            "date": datetime.strptime(r.gameDate, "%m/%d/%Y %H:%M:%S").date(),
            "season": season,
            "home_team_id": int(r.homeTeam_teamId),
            "away_team_id": int(r.awayTeam_teamId),
            "status": "scheduled",
        }
        for r in sched.itertuples()
        if r.gameId not in final_ids and r.homeTeam_teamId and r.awayTeam_teamId
    ]
    return upsert(Game.__table__, rows)


def load_season_logs(season: str, since: date | None = None) -> tuple[int, int]:
    """Load completed games + team/player box scores for a season (optionally only games on/after `since`)."""
    teams = nba_client.team_game_logs(season)
    players = nba_client.player_game_logs(season)
    if teams.empty:
        return 0, 0
    for df in (teams, players):
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
    if since:
        teams = teams[teams["GAME_DATE"] >= since]
        players = players[players["GAME_DATE"] >= since]

    teams["is_home"] = teams["MATCHUP"].str.contains("vs.")
    teams["poss"] = _possessions(teams)
    opp = teams[["GAME_ID", "TEAM_ID", "PTS", "REB", "AST", "poss"]].rename(
        columns={"TEAM_ID": "OPP_ID", "PTS": "OPP_PTS", "REB": "OPP_REB", "AST": "OPP_AST", "poss": "OPP_POSS"}
    )
    t = teams.merge(opp, on="GAME_ID")
    t = t[t["TEAM_ID"] != t["OPP_ID"]].copy()
    t["game_poss"] = (t["poss"] + t["OPP_POSS"]) / 2
    t["pace"] = 48 * t["game_poss"] / (t["MIN"] / 5)

    game_rows = {}
    for r in t.itertuples():
        home, away = (r.TEAM_ID, r.OPP_ID) if r.is_home else (r.OPP_ID, r.TEAM_ID)
        game_rows[r.GAME_ID] = {"id": r.GAME_ID, "date": r.GAME_DATE, "season": season,
                                "home_team_id": int(home), "away_team_id": int(away), "status": "final"}
    upsert(Game.__table__, game_rows.values())

    team_rows = [
        {
            "team_id": int(r.TEAM_ID), "game_id": r.GAME_ID, "opp_team_id": int(r.OPP_ID), "is_home": bool(r.is_home),
            "min": float(r.MIN), "pts": int(r.PTS), "reb": int(r.REB), "oreb": int(r.OREB), "dreb": int(r.DREB), "ast": int(r.AST),
            "tov": int(r.TOV), "fgm": int(r.FGM), "fga": int(r.FGA), "fg3m": int(r.FG3M), "fg3a": int(r.FG3A),
            "ftm": int(r.FTM), "fta": int(r.FTA),
            "possessions": float(r.game_poss), "pace": float(r.pace),
            "off_rtg": float(100 * r.PTS / r.game_poss), "def_rtg": float(100 * r.OPP_PTS / r.game_poss),
            "pts_allowed": int(r.OPP_PTS), "reb_allowed": int(r.OPP_REB), "ast_allowed": int(r.OPP_AST),
        }
        for r in t.itertuples()
    ]
    upsert(TeamGameStat.__table__, team_rows)

    opp_by_team_game = {(r["game_id"], r["team_id"]): r["opp_team_id"] for r in team_rows}
    players = players[players["GAME_ID"].isin(game_rows.keys())]
    player_rows = []
    for r in players.itertuples():
        player_rows.append({
            "player_id": int(r.PLAYER_ID), "game_id": r.GAME_ID, "team_id": int(r.TEAM_ID),
            "opp_team_id": opp_by_team_game[(r.GAME_ID, int(r.TEAM_ID))],
            "min": float(r.MIN or 0), "pts": int(r.PTS), "reb": int(r.REB), "oreb": int(r.OREB), "dreb": int(r.DREB),
            "ast": int(r.AST), "stl": int(r.STL), "blk": int(r.BLK), "tov": int(r.TOV), "pf": int(r.PF),
            "fgm": int(r.FGM), "fga": int(r.FGA), "fg3m": int(r.FG3M), "fg3a": int(r.FG3A),
            "ftm": int(r.FTM), "fta": int(r.FTA),
            "plus_minus": None if pd.isna(r.PLUS_MINUS) else float(r.PLUS_MINUS),
        })
    upsert(PlayerGameStat.__table__, player_rows)

    # Players seen in logs (latest team wins); positions come from load_players.
    latest = players.sort_values("GAME_DATE").groupby("PLAYER_ID").last()
    upsert(Player.__table__, [
        {"id": int(pid), "name": r.PLAYER_NAME, "team_id": int(r.TEAM_ID)} for pid, r in latest.iterrows()
    ])
    return len(team_rows), len(player_rows)


def load_players(season: str) -> int:
    idx = nba_client.player_index(season)
    rows = []
    for r in idx.itertuples():
        pos = r.POSITION or None
        rows.append({
            "id": int(r.PERSON_ID),
            "name": f"{r.PLAYER_FIRST_NAME} {r.PLAYER_LAST_NAME}".strip(),
            "team_id": int(r.TEAM_ID) if r.TEAM_ID else None,
            "position": pos,
            "pos_group": pos[0] if pos else None,
        })
    return upsert(Player.__table__, rows)


def load_injuries(on: date | None = None) -> tuple[int, list[str]]:
    """Snapshot today's ESPN injury report. Returns (rows stored, names that couldn't be matched)."""
    on = on or date.today()
    with SessionLocal() as s:
        by_name = {nba_client.normalize_name(p.name): p.id for p in s.scalars(select(Player))}
    rows, unmatched = [], []
    for inj in nba_client.espn_injuries():
        pid = by_name.get(nba_client.normalize_name(inj["name"]))
        if pid is None:
            unmatched.append(inj["name"])
            continue
        rows.append({"player_id": pid, "date": on, "status": inj["status"],
                     "detail": (inj["detail"] or "")[:512], "source": "espn"})
    return upsert(Injury.__table__, rows), unmatched
