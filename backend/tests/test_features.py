import numpy as np
import pandas as pd
import pytest

from app.features.build import Frames, Upcoming, build_features

A, B, C = 1, 2, 3  # A and B East, C West
STAR, ROLE = 100, 200


def _frames(n_games: int = 6) -> Frames:
    dates = pd.to_datetime(["2026-10-20", "2026-10-21", "2026-10-23", "2026-10-24", "2026-10-25", "2026-10-28"][:n_games])
    games, ts, ps = [], [], []
    for i, d in enumerate(dates):
        gid = f"00226{i:05d}"
        opp = B if i % 2 == 0 else C
        games.append({"id": gid, "date": d, "season": "2026-27", "home_team_id": A, "away_team_id": opp, "status": "final"})
        for team, other in ((A, opp), (opp, A)):
            ts.append({"team_id": team, "game_id": gid, "opp_team_id": other, "is_home": team == A, "min": 240.0,
                       "pts": 110, "reb": 44, "oreb": 10, "dreb": 34, "ast": 25, "tov": 14, "fgm": 40, "fga": 88,
                       "fg3m": 12, "fg3a": 35, "ftm": 18, "fta": 20,
                       "possessions": 100.0, "pace": 100.0, "off_rtg": 110.0, "def_rtg": 100.0 + i,
                       "pts_allowed": 100 + i, "reb_allowed": 40, "ast_allowed": 22})
        base = {"game_id": gid, "team_id": A, "opp_team_id": opp, "reb": 5, "oreb": 1, "dreb": 4, "ast": 3, "stl": 1,
                "blk": 0, "tov": 2, "pf": 3, "fgm": 8, "fga": 18, "fg3m": 2, "fg3a": 6, "ftm": 4, "fta": 5, "plus_minus": 0.0}
        ps.append({**base, "player_id": STAR, "min": 34.0, "pts": 10 * (i + 1)})
        if i < 4:  # role player plays the first 4 games then disappears
            ps.append({**base, "player_id": ROLE, "min": 25.0, "pts": 12})
    teams = pd.DataFrame([
        {"id": A, "abbr": "AAA", "name": "A", "conference": "East", "division": "Atlantic"},
        {"id": B, "abbr": "BBB", "name": "B", "conference": "East", "division": "Central"},
        {"id": C, "abbr": "CCC", "name": "C", "conference": "West", "division": "Pacific"},
    ])
    players = pd.DataFrame([{"id": STAR, "name": "Star", "team_id": A, "position": "G", "pos_group": "G"},
                            {"id": ROLE, "name": "Role", "team_id": A, "position": "F", "pos_group": "F"}])
    return Frames(pd.DataFrame(ps), pd.DataFrame(ts), pd.DataFrame(games), players, teams)


def _star(f: pd.DataFrame) -> pd.DataFrame:
    return f[f["player_id"] == STAR].sort_values("date").reset_index(drop=True)


def test_rolling_windows_exclude_current_game():
    s = _star(build_features(_frames()))
    assert np.isnan(s.loc[0, "pts_l5"])  # no history before the first game
    assert s.loc[1, "pts_l5"] == 10
    assert s.loc[3, "pts_l5"] == pytest.approx((10 + 20 + 30) / 3)
    assert s.loc[5, "pts_l5"] == pytest.approx((10 + 20 + 30 + 40 + 50) / 5)
    assert s.loc[5, "pts_season"] == pytest.approx(30)


def test_opponent_defense_uses_only_prior_games():
    s = _star(build_features(_frames()))
    # Game 4 is vs B; B's earlier games were 0 and 2 with def_rtg 100 and 102.
    assert s.loc[4, "opp_def_rtg_season"] == pytest.approx(101)


def test_rest_b2b_and_conference():
    s = _star(build_features(_frames()))
    assert list(s["b2b"]) == [0, 1, 0, 1, 1, 0]
    assert list(s["rest_days"].fillna(-1)) == [-1, 1, 2, 1, 1, 3]
    assert s.loc[4, "three_in_four"] == 1
    assert list(s["same_conference"]) == [1, 0, 1, 0, 1, 0]


def test_missing_rotation_teammate_detected():
    s = _star(build_features(_frames()))
    assert s.loc[3, "missing_teammates"] == 0
    assert s.loc[4, "missing_teammates"] == 1  # role player sat after playing 4 straight
    assert s.loc[4, "missing_teammate_pts"] == pytest.approx(12)


def test_upcoming_game_features_and_overrides():
    fr = _frames()
    game = pd.DataFrame([{"id": "0022699999", "date": pd.Timestamp("2026-10-29"), "season": "2026-27",
                          "home_team_id": C, "away_team_id": A, "status": "scheduled"}])
    roster = pd.DataFrame({"player_id": [STAR], "team_id": [A], "game_id": ["0022699999"]})
    f = build_features(fr, Upcoming(game, roster, out=set(), b2b_overrides={A: False}))
    row = f[f["game_id"] == "0022699999"].iloc[0]
    assert np.isnan(row["pts"])
    assert row["pts_l5"] == pytest.approx((20 + 30 + 40 + 50 + 60) / 5)
    assert row["b2b"] == 0 and row["is_home"] == 0

    forced = build_features(fr, Upcoming(game, roster, b2b_overrides={A: True}))
    assert forced[forced["game_id"] == "0022699999"].iloc[0]["b2b"] == 1


def test_shooting_efficiency_is_ratio_of_sums_from_prior_games():
    s = _star(build_features(_frames()))
    # Before game 2 the star scored 10 + 20 on 36 FGA and 10 FTA.
    assert s.loc[2, "ts_l5"] == pytest.approx(30 / (2 * (36 + 0.44 * 10)))
    assert s.loc[2, "efg_season"] == pytest.approx((16 + 0.5 * 4) / 36)
    assert s.loc[2, "pts_per36_l10"] == pytest.approx(36 * 30 / 68)
    assert np.isnan(s.loc[0, "ts_season"])


def test_opponent_ts_allowed():
    s = _star(build_features(_frames()))
    # Opponent B's defense before game 2: allowed 100 pts on 88 FGA / 20 FTA in game 0.
    assert s.loc[2, "opp_ts_allowed_season"] == pytest.approx(100 / (2 * (88 + 0.44 * 20)))


def test_head_to_head_history():
    s = _star(build_features(_frames()))
    # Games 0, 2, 4 are vs B. Before game 4: two prior meetings, star scored 10 then 30 vs B.
    assert s.loc[4, "games_vs_opp"] == 2
    assert s.loc[4, "pts_last_vs_opp"] == 30
    assert s.loc[4, "pts_vs_opp"] == pytest.approx(20)
    assert s.loc[4, "h2h_games"] == 2
    assert s.loc[4, "h2h_pts_allowed"] == pytest.approx((100 + 102) / 2)
    assert np.isnan(s.loc[0, "h2h_pace"])


def test_per36_rates_and_blowout_gap():
    s = _star(build_features(_frames()))
    assert s.loc[1, "pf_per36_season"] == pytest.approx(36 * 3 / 34)
    assert s.loc[1, "reb_per36_l5"] == pytest.approx(36 * 5 / 34)
    assert s.loc[2, "abs_net_rtg_diff"] == pytest.approx(abs(s.loc[2, "net_rtg_diff"]))


def test_rest_advantage():
    s = _star(build_features(_frames()))
    # Game 3 (Oct 24) vs C: A played the night before (1 day rest); C last played game 1 on Oct 21 (3 days).
    assert s.loc[3, "rest_days"] == 1
    assert s.loc[3, "rest_advantage"] == 1 - 3


def test_conformal_ranges_cover_80_percent_and_scale_with_volatility():
    from app.ml import intervals

    rng = np.random.default_rng(0)
    n = 4000
    vol = rng.choice([3.0, 9.0], n)  # steady vs streaky players
    df = pd.DataFrame({"pts_std_l10": vol, "pts": 20 + rng.normal(0, vol)})
    pred = np.full(n, 20.0)
    cal = intervals.fit(df.iloc[: n // 2], "pts", pred[: n // 2])
    lo, hi = intervals.apply(df.iloc[n // 2 :], "pts", pred[n // 2 :], cal)
    assert 0.77 < intervals.coverage(df["pts"].to_numpy()[n // 2 :], lo, hi) < 0.83
    width = hi - lo
    assert width[vol[n // 2 :] == 9.0].mean() > 2.5 * width[vol[n // 2 :] == 3.0].mean()
