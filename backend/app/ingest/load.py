"""Transform raw NBA data into DB rows and upsert them. Shared by backfill and the daily job."""

from datetime import date, datetime

import pandas as pd
from sqlalchemy import func, select

from app.db import SessionLocal, upsert
from app.ingest import nba_client
from app.models import Game, GameLine, Injury, Player, PlayerGameStat, PlayerProp, Team, TeamGameStat

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
    return _store_logs(season, teams, players)


def load_espn_logs(season: str, dates: list[date]) -> tuple[int, int, list[str]]:
    """Same as load_season_logs for the listed dates, but from ESPN box scores (for when stats.nba.com is unreachable).

    Games are matched to the stored schedule by date and teams, players to known players by name. A player we have
    never seen can't be given an NBA id this way and is skipped (returned in the unmatched list).
    Returns (team-games, player-games, unmatched names).
    """
    with SessionLocal() as s:
        abbr_to_id = {t.abbr: t.id for t in s.scalars(select(Team))}
        games = {(g.date, g.home_team_id, g.away_team_id): g.id for g in s.scalars(select(Game).where(Game.date.in_(dates)))}
        players_db = s.scalars(select(Player)).all()
    by_name: dict[str, list[Player]] = {}
    for p in players_db:
        by_name.setdefault(nba_client.normalize_name(p.name), []).append(p)
    team_rows, player_rows, unmatched = [], [], set()
    for d in dates:
        for box in nba_client.espn_box_scores(d):
            home, away = abbr_to_id.get(box["home"]), abbr_to_id.get(box["away"])
            game_id = games.get((d, home, away))
            if game_id is None:
                continue
            team_min = 240 + 25 * (box["periods"] - 4)
            totals = {home: {"MIN": team_min}, away: {"MIN": team_min}}
            for pl in box["players"]:
                team_id = abbr_to_id.get(pl["team"])
                candidates = by_name.get(nba_client.normalize_name(pl["name"]), [])
                match = next((c for c in candidates if c.team_id == team_id), candidates[0] if len(candidates) == 1 else None)
                stats = {k.upper(): v for k, v in pl.items() if k not in ("name", "team", "plus_minus")}
                for k, v in stats.items():
                    if k != "MIN":
                        totals[team_id][k] = totals[team_id].get(k, 0) + v
                if match is None:
                    unmatched.add(pl["name"])
                    continue
                player_rows.append({"GAME_ID": game_id, "GAME_DATE": d, "PLAYER_ID": match.id, "PLAYER_NAME": match.name,
                                    "TEAM_ID": team_id, "PLUS_MINUS": pl["plus_minus"], **stats})
            for abbr, tov in box["team_tov"].items():
                if abbr_to_id.get(abbr) in totals:
                    totals[abbr_to_id[abbr]]["TOV"] = tov
            for team_id, t in totals.items():
                team_rows.append({"GAME_ID": game_id, "GAME_DATE": d, "TEAM_ID": team_id, "MATCHUP": "vs." if team_id == home else "@", **t})
    if not team_rows:
        return 0, 0, sorted(unmatched)
    n_team, n_player = _store_logs(season, pd.DataFrame(team_rows), pd.DataFrame(player_rows))
    return n_team, n_player, sorted(unmatched)


def _store_logs(season: str, teams: pd.DataFrame, players: pd.DataFrame) -> tuple[int, int]:
    """Upsert games, team box scores and player box scores from frames shaped like the NBA's LeagueGameLog."""
    # The log includes games still in progress. A finished game has >= 240 team minutes (5 x 48) for both sides;
    # anything less is partial, so skip it until a later run sees it complete.
    complete = teams.groupby("GAME_ID")["MIN"].min() >= 240
    complete_ids = set(complete[complete].index)
    teams = teams[teams["GAME_ID"].isin(complete_ids)]
    players = players[players["GAME_ID"].isin(complete_ids)]

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


def load_game_lines(dates: list[date]) -> int:
    """Store the current sportsbook line for each listed date's games. Never blanks out a line once games start."""
    with SessionLocal() as s:
        abbr_to_id = {t.abbr: t.id for t in s.scalars(select(Team))}
        games = s.scalars(select(Game).where(Game.date.in_(dates))).all()
    by_matchup = {(g.date, g.home_team_id, g.away_team_id): g.id for g in games}
    rows = []
    for d in dates:
        for line in nba_client.espn_game_lines(d):
            game_id = by_matchup.get((d, abbr_to_id.get(line["home"]), abbr_to_id.get(line["away"])))
            if game_id:
                rows.append({"game_id": game_id, "home_spread": line["home_spread"], "total": line["total"], "provider": line["provider"]})
    return upsert(GameLine.__table__, rows)


def load_player_props(dates: list[date]) -> tuple[int, list[str]]:
    """Store DraftKings points/rebounds/assists lines for the listed dates' games.

    Games that haven't started overwrite any earlier snapshot (latest pre-game line wins). Finished games only fill
    in lines we never captured, so a pre-game snapshot is never replaced. Games in progress are skipped.
    Returns (rows written, ESPN names that couldn't be matched to a player).
    """
    with SessionLocal() as s:
        abbr_to_id = {t.abbr: t.id for t in s.scalars(select(Team))}
        games = s.scalars(select(Game).where(Game.date.in_(dates))).all()
        by_name = {nba_client.normalize_name(p.name): p.id for p in s.scalars(select(Player))}
        known = dict(s.execute(select(PlayerProp.espn_athlete_id, PlayerProp.player_id).distinct()).all())
    by_matchup = {(g.date, g.home_team_id, g.away_team_id): g.id for g in games}
    rosters: dict[str, dict[int, str]] = {}
    pre, post, unmatched = [], [], set()
    for d in dates:
        for prop in nba_client.espn_player_props(d):
            game_id = by_matchup.get((d, abbr_to_id.get(prop["home"]), abbr_to_id.get(prop["away"])))
            if game_id is None or prop["state"] == "in":
                continue
            aid = prop["espn_athlete_id"]
            if aid not in known:
                name = _espn_name(aid, prop["espn_team_ids"], rosters)
                known[aid] = by_name.get(nba_client.normalize_name(name)) if name else None
                if known[aid] is None:
                    unmatched.add(name or f"ESPN athlete {aid}")
            if known[aid] is None:
                continue
            row = {"player_id": known[aid], "game_id": game_id, "target": prop["target"], "line": prop["line"],
                   "open_line": prop["open_line"], "over_odds": prop["over_odds"], "under_odds": prop["under_odds"], "espn_athlete_id": aid, "provider": "DraftKings", "source_updated": prop["updated"]}
            (pre if prop["state"] == "pre" else post).append(row)
    n = upsert(PlayerProp.__table__, pre) + upsert(PlayerProp.__table__, post, update=False)
    return n, sorted(unmatched)


def _espn_name(athlete_id: int, team_ids: list[str], rosters: dict[str, dict[int, str]]) -> str | None:
    for team_id in team_ids:
        if team_id not in rosters:
            rosters[team_id] = nba_client.espn_team_roster(team_id)
        if athlete_id in rosters[team_id]:
            return rosters[team_id][athlete_id]
    return nba_client.espn_athlete_name(athlete_id)  # e.g. traded since the game


def dates_missing_props(season: str, through: date) -> list[date]:
    """Dates with finished games this season and no prop lines stored for any of them (for the season backfill)."""
    with SessionLocal() as s:
        rows = s.execute(
            select(Game.date).outerjoin(PlayerProp, PlayerProp.game_id == Game.id)
            .where(Game.season == season, Game.status == "final", Game.date <= through)
            .group_by(Game.date).having(func.count(PlayerProp.game_id) == 0).order_by(Game.date)
        ).scalars().all()
    return list(rows)


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
