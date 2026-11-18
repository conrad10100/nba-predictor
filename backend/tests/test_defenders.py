import pandas as pd
import pytest

from app.features.build import Frames, Upcoming, build_features
from app.features.defenders import DEFENDER_FEATURES

A, B = 1, 2
STAR, GOOD, ROOKIE = 100, 200, 300
DATES = pd.to_datetime(["2026-10-20", "2026-10-22", "2026-10-24", "2026-10-26"])


def _frames() -> Frames:
    games, ts, ps = [], [], []
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
    teams = pd.DataFrame([
        {"id": A, "abbr": "AAA", "name": "A", "conference": "East", "division": "Atlantic"},
        {"id": B, "abbr": "BBB", "name": "B", "conference": "East", "division": "Central"},
    ])
    players = pd.DataFrame([
        {"id": STAR, "name": "Star", "team_id": A, "position": "G", "pos_group": "G"},
        {"id": GOOD, "name": "Good", "team_id": B, "position": "G", "pos_group": "G"},
        {"id": ROOKIE, "name": "Rookie", "team_id": B, "position": "G", "pos_group": "G"},
    ])
    # Last season's final DBPM: GOOD is an elite defender, ROOKIE has none (counts as average). A snapshot of this
    # season is published on game 3's date, which must not reach game 3 itself.
    ratings = pd.DataFrame([
        {"player_id": GOOD, "season": "2025-26", "as_of": pd.Timestamp("2026-07-01"), "dbpm": 3.0, "minutes": 2500.0},
        {"player_id": STAR, "season": "2025-26", "as_of": pd.Timestamp("2026-07-01"), "dbpm": -1.0, "minutes": 2500.0},
        {"player_id": GOOD, "season": "2026-27", "as_of": DATES[3], "dbpm": -6.0, "minutes": 2000.0},
    ])
    return Frames(pd.DataFrame(ps), pd.DataFrame(ts), pd.DataFrame(games), players, teams, defender_ratings=ratings)


def _star(f: pd.DataFrame) -> pd.DataFrame:
    return f[f["player_id"] == STAR].sort_values("date").reset_index(drop=True)


def test_no_defender_features_without_ratings():
    fr = _frames()
    fr.defender_ratings = fr.defender_ratings.iloc[0:0]
    assert _star(build_features(fr))[DEFENDER_FEATURES].isna().all().all()


def test_rating_uses_only_what_was_published_before_the_game():
    s = _star(build_features(_frames()))
    # Through game 3 only last season's ratings exist: GOOD 3.0 shrunk by 2500/(2500+500) = 2.5, ROOKIE 0 (no data).
    assert s.loc[3, "opp_pos_starter_def_dbpm"] == pytest.approx(2.5)  # GOOD plays the most minutes
    assert s.loc[3, "opp_def_dbpm_team"] == pytest.approx((2.5 * 34 + 0 * 20) / 54)


def test_new_snapshot_applies_after_its_date():
    fr = _frames()
    up = _upcoming(fr, set())
    # Game after the snapshot: GOOD = (-6 * 2000 + 2.5 * 800) / 2800
    assert up["opp_pos_starter_def_dbpm"] == pytest.approx((-6 * 2000 + 2.5 * 800) / 2800)


def _upcoming(fr: Frames, out: set[int]) -> pd.Series:
    game = pd.DataFrame([{"id": "0022699999", "date": pd.Timestamp("2026-10-28"), "season": "2026-27",
                          "home_team_id": A, "away_team_id": B, "status": "scheduled"}])
    roster = pd.DataFrame({"player_id": [STAR, GOOD, ROOKIE], "team_id": [A, B, B], "game_id": ["0022699999"] * 3})
    roster = roster[~roster["player_id"].isin(out)]
    f = build_features(fr, Upcoming(game, roster, out=out))
    return f[(f["game_id"] == "0022699999") & (f["player_id"] == STAR)].iloc[0]


def test_missing_good_defender_makes_matchup_easier():
    fr = _frames()
    fr.defender_ratings = fr.defender_ratings.iloc[:2]  # last season's ratings only
    full, without_good = _upcoming(fr, set()), _upcoming(fr, {GOOD})
    assert without_good["opp_def_dbpm_team"] < full["opp_def_dbpm_team"]
    assert without_good["opp_def_dbpm_vs_usual"] < 0 < full["opp_def_dbpm_vs_usual"] + 1e-9
    assert without_good["opp_pos_starter_def_dbpm"] == 0 < full["opp_pos_starter_def_dbpm"]  # the rookie steps in


def test_defender_features_only_used_when_switched_on():
    from app.ml.train import usable_features

    rows = _star(build_features(_frames()))
    assert not set(DEFENDER_FEATURES) & set(usable_features(rows, {"defender_features": 0}))
    assert set(DEFENDER_FEATURES) <= set(usable_features(rows, {"defender_features": 1}))


def test_basketball_reference_table_parsing():
    from app.ingest.nba_client import parse_bbref_advanced

    def row(slug, name, team, mp, dbpm):
        return (f'<tr><th data-stat="ranker">1</th><td data-append-csv="{slug}" data-stat="name_display"><a href="#">{name}</a></td>'
                f'<td data-stat="team_name_abbr">{team}</td><td data-stat="mp">{mp}</td><td data-stat="dbpm">{dbpm}</td></tr>')
    html = ('<table id="advanced"><thead><tr><th data-stat="ranker">Rk</th></tr></thead><tbody>'
            + row("goodpl01", "Good Player", "BRK", "1500", "2.1") + row("trade01", "Traded Guy", "2TM", "900", "-0.4")
            + row("trade01", "Traded Guy", "PHO", "400", "1.0") + '<tr class="thead"><th data-stat="ranker">Rk</th></tr>'
            + row("lgavg", "League Average", "", "", "0.0") + "</tbody></table>")
    out = parse_bbref_advanced(html)
    assert out == [
        {"name": "Good Player", "slug": "goodpl01", "team": "BKN", "minutes": 1500.0, "dbpm": 2.1},
        {"name": "Traded Guy", "slug": "trade01", "team": "2TM", "minutes": 900.0, "dbpm": -0.4},
    ]
