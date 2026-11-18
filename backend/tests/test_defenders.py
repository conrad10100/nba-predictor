import pandas as pd
import pytest

from app.features.build import Frames, Upcoming, build_features
from app.features.defenders import DEFENDER_FEATURES

A, B = 1, 2
STAR, GOOD, ROOKIE = 100, 200, 300
DATES = pd.to_datetime(["2026-10-20", "2026-10-22", "2026-10-24", "2026-10-26"])


def _frames() -> Frames:
    games, ts, ps, oc = [], [], [], []
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
        # B allows points at 0.030/s with GOOD on and 0.057/s with GOOD off; with ROOKIE on it's the reverse.
        for pid, team, on, pts_on in ((STAR, A, 2400.0, 90), (GOOD, B, 2000.0, 60), (ROOKIE, B, 1000.0, 50)):
            oc.append({"game_id": gid, "player_id": pid, "team_id": team, "on_seconds": on, "opp_pts_on": pts_on,
                       "game_seconds": 2880.0, "opp_pts_game": 110})
    teams = pd.DataFrame([
        {"id": A, "abbr": "AAA", "name": "A", "conference": "East", "division": "Atlantic"},
        {"id": B, "abbr": "BBB", "name": "B", "conference": "East", "division": "Central"},
    ])
    players = pd.DataFrame([
        {"id": STAR, "name": "Star", "team_id": A, "position": "G", "pos_group": "G"},
        {"id": GOOD, "name": "Good", "team_id": B, "position": "G", "pos_group": "G"},
        {"id": ROOKIE, "name": "Rookie", "team_id": B, "position": "G", "pos_group": "G"},
    ])
    return Frames(pd.DataFrame(ps), pd.DataFrame(ts), pd.DataFrame(games), players, teams, on_court=pd.DataFrame(oc))


def _star(f: pd.DataFrame) -> pd.DataFrame:
    return f[f["player_id"] == STAR].sort_values("date").reset_index(drop=True)


def test_no_defender_features_before_tracking_starts():
    s = _star(build_features(_frames()))
    assert s.loc[0, DEFENDER_FEATURES].isna().all()


def test_defender_history_excludes_current_game():
    fr = _frames()
    base = _star(build_features(fr))
    # Rewrite the last game's on-court numbers: features for that game (built from earlier games) must not move.
    oc = fr.on_court.copy()
    last = oc["game_id"] == "0022600003"
    oc.loc[last & (oc["player_id"] == GOOD), "opp_pts_on"] = 110
    fr.on_court = oc
    changed = _star(build_features(fr))
    assert changed.loc[3, DEFENDER_FEATURES].tolist() == pytest.approx(base.loc[3, DEFENDER_FEATURES].tolist())
    assert base.loc[2, "opp_def_rating_team"] < 1.0  # GOOD plays more minutes than ROOKIE


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
    assert without_good["opp_pos_starter_def_rating"] > full["opp_pos_starter_def_rating"]  # the rookie steps in


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


def _athlete(aid, name, starter):
    return {"athlete": {"id": aid, "displayName": name, "shortName": name}, "starter": starter, "stats": ["1"]}


def _summary(plays):
    box = []
    for tid, abbr, prefix in (("1", "AAA", "H"), ("2", "BBB", "V")):
        athletes = [_athlete(f"{prefix}{i}", f"{prefix} Player{i}", i <= 5) for i in range(1, 7)]
        box.append({"team": {"id": tid, "abbreviation": abbr}, "statistics": [{"athletes": athletes}]})
    comps = [{"homeAway": "home", "team": {"id": "1"}}, {"homeAway": "away", "team": {"id": "2"}}]
    return {"boxscore": {"players": box}, "header": {"competitions": [{"competitors": comps}]}, "plays": plays}


def _play(period, clock, home, away, text="", team=None, participants=()):
    return {"period": {"number": period}, "clock": {"displayValue": clock}, "homeScore": home, "awayScore": away,
            "text": text, "type": {"text": "Substitution" if "enters" in text else "Shot"},
            "team": {"id": team} if team else {}, "participants": [{"athlete": {"id": p}} for p in participants]}


def test_on_court_from_play_by_play():
    from app.ingest.on_court import parse_summary

    plays = [
        _play(1, "11:00", 2, 0, "H Player1 makes layup", "1", ["H1"]),
        _play(1, "6:00", 2, 0, "H Player6 enters the game for H Player5", "1", ["H6", "H5"]),
        _play(1, "1:00", 2, 3, "V Player1 makes three", "2", ["V1"]),
        # Quarter-break change, not logged: H5 is back for H6 and shows up acting before any sub.
        _play(2, "11:30", 4, 3, "H Player5 makes jumper", "1", ["H5"]),
        _play(3, "12:00", 4, 3), _play(4, "12:00", 4, 3),
    ]
    out = parse_summary(_summary(plays))
    p = out["players"]
    assert out["game_seconds"] == 2880 and out["pts"] == {"1": 4, "2": 3}
    assert p["H5"]["on_seconds"] == pytest.approx(360 + 720 + 1440)  # half of Q1, then Q2-Q4
    assert p["H6"]["on_seconds"] == pytest.approx(360)
    assert p["H6"]["opp_pts_on"] == 3 and p["H5"]["opp_pts_on"] == 0
    assert p["V1"]["opp_pts_on"] == 4 and "V6" not in p
