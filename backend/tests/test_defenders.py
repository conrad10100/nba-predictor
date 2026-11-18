import pandas as pd
import pytest

from app.features.build import Frames, Upcoming, build_features
from app.features.defenders import DEFENDER_FEATURES

A, B = 1, 2
STAR, GOOD, ROOKIE = 100, 200, 300
DATES = pd.to_datetime(["2026-10-20", "2026-10-22", "2026-10-24", "2026-10-26"])


def _frames() -> Frames:
    games, ts, ps, mu = [], [], [], []
    for i, d in enumerate(DATES):
        gid = f"00226{i:05d}"
        games.append({"id": gid, "date": d, "season": "2026-27", "home_team_id": A, "away_team_id": B, "status": "final"})
        for team, other in ((A, B), (B, A)):
            ts.append({"team_id": team, "game_id": gid, "opp_team_id": other, "is_home": team == A, "min": 240.0,
                       "pts": 110, "reb": 44, "oreb": 10, "dreb": 34, "ast": 25, "tov": 14, "fgm": 40, "fga": 88,
                       "fg3m": 12, "fg3a": 35, "ftm": 18, "fta": 20, "possessions": 100.0, "pace": 100.0,
                       "off_rtg": 110.0, "def_rtg": 105.0, "pts_allowed": 105, "reb_allowed": 40, "ast_allowed": 22})
        box = {"game_id": gid, "reb": 5, "oreb": 1, "dreb": 4, "ast": 3, "stl": 1, "blk": 0, "tov": 2, "pf": 3, "fgm": 8,
               "fga": 18, "fg3m": 2, "fg3a": 6, "ftm": 4, "fta": 5, "plus_minus": 0.0}
        ps.append({**box, "player_id": STAR, "team_id": A, "opp_team_id": B, "min": 36.0, "pts": 25})
        ps.append({**box, "player_id": GOOD, "team_id": B, "opp_team_id": A, "min": 34.0, "pts": 10})
        ps.append({**box, "player_id": ROOKIE, "team_id": B, "opp_team_id": A, "min": 20.0, "pts": 6})
        # GOOD holds STAR to 4 points in 20 possessions; ROOKIE gives up 14 in 20.
        for d_id, pts in ((GOOD, 4), (ROOKIE, 14)):
            mu.append({"game_id": gid, "off_player_id": STAR, "def_player_id": d_id, "off_team_id": A, "def_team_id": B,
                       "partial_poss": 20.0, "pts": pts})
    teams = pd.DataFrame([
        {"id": A, "abbr": "AAA", "name": "A", "conference": "East", "division": "Atlantic"},
        {"id": B, "abbr": "BBB", "name": "B", "conference": "East", "division": "Central"},
    ])
    players = pd.DataFrame([
        {"id": STAR, "name": "Star", "team_id": A, "position": "G", "pos_group": "G"},
        {"id": GOOD, "name": "Good", "team_id": B, "position": "G", "pos_group": "G"},
        {"id": ROOKIE, "name": "Rookie", "team_id": B, "position": "G", "pos_group": "G"},
    ])
    return Frames(pd.DataFrame(ps), pd.DataFrame(ts), pd.DataFrame(games), players, teams, matchups=pd.DataFrame(mu))


def _star(f: pd.DataFrame) -> pd.DataFrame:
    return f[f["player_id"] == STAR].sort_values("date").reset_index(drop=True)


def test_no_defender_features_before_tracking_starts():
    s = _star(build_features(_frames()))
    assert s.loc[0, DEFENDER_FEATURES].isna().all()


def test_defender_history_excludes_current_game():
    s = _star(build_features(_frames()))
    assert s.loc[1, "likely_def_poss"] == pytest.approx(40)  # game 0 only
    assert s.loc[3, "likely_def_poss"] == pytest.approx(120)  # games 0-2, not game 3
    # Against these two STAR scores 18 per 40 possessions every game, exactly his own norm, so the ratio is 1.
    assert s.loc[3, "vs_likely_def_ratio"] == pytest.approx(1.0)


def _upcoming(fr: Frames, out: set[int]) -> pd.Series:
    game = pd.DataFrame([{"id": "0022699999", "date": pd.Timestamp("2026-10-28"), "season": "2026-27",
                          "home_team_id": A, "away_team_id": B, "status": "scheduled"}])
    roster = pd.DataFrame({"player_id": [STAR, GOOD, ROOKIE], "team_id": [A, B, B], "game_id": ["0022699999"] * 3})
    roster = roster[~roster["player_id"].isin(out)]
    f = build_features(fr, Upcoming(game, roster, out=out))
    return f[(f["game_id"] == "0022699999") & (f["player_id"] == STAR)].iloc[0]


def test_missing_good_defender_makes_matchup_easier():
    fr = _frames()
    full, without_good = _upcoming(fr, set()), _upcoming(fr, {GOOD})
    assert without_good["opp_def_rating_team"] > full["opp_def_rating_team"]
    assert without_good["opp_def_rating_vs_usual"] > full["opp_def_rating_vs_usual"]
    assert without_good["likely_def_rating"] > full["likely_def_rating"]  # only the rookie left to guard him
    assert without_good["likely_def_poss"] < full["likely_def_poss"]


def test_defender_features_only_used_when_switched_on():
    from app.ml.train import usable_features

    rows = _star(build_features(_frames()))
    assert not set(DEFENDER_FEATURES) & set(usable_features(rows, {"defender_features": 0}))
    assert set(DEFENDER_FEATURES) <= set(usable_features(rows, {"defender_features": 1}))


def test_good_defender_rates_better_than_rookie():
    fr = _frames()
    only_good, only_rookie = _upcoming(fr, {ROOKIE}), _upcoming(fr, {GOOD})
    assert only_good["opp_def_rating_team"] < 1.0 < only_rookie["opp_def_rating_team"]
    assert only_good["opp_def_rating_pos"] == pytest.approx(only_good["opp_def_rating_team"])


def test_matchup_parsing_and_orientation():
    from app.ingest.matchups import _orient, parse

    df = pd.DataFrame([{
        "teamId": A, "homeTeamId": A, "awayTeamId": B, "personIdOff": STAR, "personIdDef": GOOD, "matchupMinutes": "5:30",
        "partialPossessions": 12.4, "playerPoints": 6, "matchupFieldGoalsMade": 3, "matchupFieldGoalsAttempted": 7,
        "matchupThreePointersMade": 0, "matchupThreePointersAttempted": 2, "matchupFreeThrowsAttempted": 0,
        "matchupAssists": 1, "matchupTurnovers": 0,
    }])
    rows = parse("0022600001", df)
    assert rows[0]["def_team_id"] == B and rows[0]["seconds"] == 330 and rows[0]["pts"] == 6
    assert _orient(rows, {STAR: A, GOOD: B}) == rows
    swapped = _orient(rows, {STAR: B, GOOD: A})[0]
    assert (swapped["off_player_id"], swapped["def_player_id"], swapped["def_team_id"]) == (GOOD, STAR, A)
