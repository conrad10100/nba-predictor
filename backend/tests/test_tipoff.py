from datetime import datetime, timedelta, timezone

from app.ingest import nba_client
from jobs import daily

NOW = datetime(2026, 11, 19, 0, 10, tzinfo=timezone.utc)  # 7:10pm ET


def test_parse_espn_time():
    assert nba_client.parse_espn_time("2026-11-19T00:30Z") == datetime(2026, 11, 19, 0, 30, tzinfo=timezone.utc)


def test_only_games_about_to_tip_trigger_a_refresh():
    games = [
        {"home": "BOS", "away": "NYK", "state": "pre", "start": NOW + timedelta(minutes=20)},   # 7:30 tip: refresh
        {"home": "DEN", "away": "LAL", "state": "pre", "start": NOW + timedelta(minutes=110)},  # 9:00 tip: later
        {"home": "MIA", "away": "ORL", "state": "in", "start": NOW - timedelta(minutes=10)},    # already on
        {"home": "CHI", "away": "DET", "state": "pre", "start": NOW - timedelta(minutes=5)},    # tip running late
        {"home": "ATL", "away": "WAS", "state": "pre", "start": NOW - timedelta(hours=2)},     # stuck feed, ignore
    ]
    soon = daily._tipping_soon(games, timedelta(minutes=30), now=NOW)
    assert [g["home"] for g in soon] == ["BOS", "CHI"]
    assert daily._tipping_soon(games, timedelta(minutes=10), now=NOW) == [games[3]]
