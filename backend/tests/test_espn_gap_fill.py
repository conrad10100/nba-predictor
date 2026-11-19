from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.ingest import load
from app.models import Game, Player, Team
from jobs import daily

TODAY = date(2026, 11, 19)
MIA, MIL, CLE, MEM = 1610612748, 1610612749, 1610612739, 1610612763


@pytest.fixture
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(engine)
    with Session() as s:
        s.add_all([Team(id=i, abbr=a, name=a, conference="E", division="x") for i, a in ((MIA, "MIA"), (MIL, "MIL"), (CLE, "CLE"), (MEM, "MEM"))])
        s.add_all([
            Game(id="0022600258", date=date(2026, 11, 18), season="2026-27", home_team_id=CLE, away_team_id=MEM, status="final"),
            Game(id="0022600260", date=date(2026, 11, 18), season="2026-27", home_team_id=MIA, away_team_id=MIL, status="scheduled"),
            Game(id="0022600231", date=date(2026, 11, 14), season="2026-27", home_team_id=CLE, away_team_id=MIA, status="scheduled"),
            Game(id="0022600100", date=date(2026, 10, 1), season="2026-27", home_team_id=CLE, away_team_id=MIL, status="scheduled"),  # too old
            Game(id="0022600271", date=TODAY, season="2026-27", home_team_id=MIL, away_team_id=MEM, status="scheduled"),  # tonight
        ])
        s.add(Player(id=203507, name="Giannis Antetokounmpo", team_id=MIL))
        s.commit()
    monkeypatch.setattr(daily, "engine", engine)
    monkeypatch.setattr(daily, "CURRENT_SEASON", "2026-27")
    monkeypatch.setattr(load, "SessionLocal", Session)
    return engine


def test_played_games_missing_from_the_stats_log_are_filled_from_espn(db, monkeypatch):
    calls = []
    monkeypatch.setattr(daily, "load_espn_logs", lambda season, dates, only: calls.append((dates, only)) or (0, 0, []))
    daily._fill_gaps_from_espn(TODAY)
    assert calls == [([date(2026, 11, 14), date(2026, 11, 18)], {"0022600231", "0022600260"})]


def test_espn_load_touches_only_the_missing_games(db, monkeypatch):
    def box(home, away, players):
        return {"home": home, "away": away, "periods": 4, "team_tov": {}, "players": [
            {"name": n, "team": t, "min": 36.0, "pts": 30, "reb": 10, "oreb": 2, "dreb": 8, "ast": 5, "stl": 1, "blk": 1,
             "tov": 3, "pf": 2, "fgm": 11, "fga": 20, "fg3m": 1, "fg3a": 3, "ftm": 7, "fta": 9, "plus_minus": -7.0}
            for n, t in players]}
    monkeypatch.setattr(load.nba_client, "espn_box_scores", lambda d: [
        box("CLE", "MEM", [("Someone Else", "CLE")]), box("MIA", "MIL", [("Giannis Antetokounmpo", "MIL")])])
    stored = []
    monkeypatch.setattr(load, "_store_logs", lambda season, teams, players: stored.append((teams, players)) or (len(teams), len(players)))
    n_team, n_player, unmatched = load.load_espn_logs("2026-27", [date(2026, 11, 18)], only={"0022600260"})
    teams, players = stored[0]
    assert set(teams["GAME_ID"]) == {"0022600260"}  # the game stats.nba.com already gave us isn't overwritten
    assert list(players["PLAYER_ID"]) == [203507] and (n_team, n_player, unmatched) == (2, 1, [])
