"""Opponent defender features: how good tonight's defenders are, and who is likely to guard the player.

Built from the NBA's per-game matchup tracking (who guarded whom, in partial possessions). Everything is "as of
before this game's date", so training rows never see the game they predict.

Defender rating: points a defender allowed divided by the points those same scorers would normally score in that
many possessions (1.00 = average, 0.90 = holds scorers 10% below their norm). Shrunk toward 1.00, so a defender with
little tracked history counts as average rather than great or terrible.

For each player-game:
- opp_def_rating_team: tonight's opposing defenders, weighted by their recent minutes
- opp_def_rating_pos: the same, only defenders in the player's position group (G/F/C)
- opp_def_rating_vs_usual: tonight's team rating minus the opponent's usual one (positive = a good defender is out)
- likely_def_rating: defender ratings weighted by how much each of tonight's defenders has guarded this player before
- likely_def_poss: how many tracked possessions that history covers (how much to trust the two above)
- vs_likely_def_ratio: this player's own scoring against tonight's defenders, relative to their norm (shrunk to 1.00)

Rows before the first tracked game are left missing, so the model can tell "no data" from "average defenders".
"""

import numpy as np
import pandas as pd

DEFENDER_FEATURES = [
    "opp_def_rating_team", "opp_def_rating_pos", "opp_def_rating_vs_usual",
    "likely_def_rating", "likely_def_poss", "vs_likely_def_ratio",
]
MATCHUP_COLS = ["game_id", "off_player_id", "def_player_id", "off_team_id", "def_team_id", "partial_poss", "pts"]

K_OFF_POSS = 150.0  # scorer's own rate is shrunk toward league average by this many possessions
K_DEF_PTS = 60.0  # defender rating is shrunk toward 1.00 by this many expected points (~ several games of guarding)
K_PAIR_PTS = 20.0  # player-vs-these-defenders ratio is shrunk toward 1.00 by this many expected points
DEFAULT_MINUTES = 15.0  # weight for a defender with no recent minutes (e.g. first game back)
USUAL_WINDOW = 10


def empty_matchups() -> pd.DataFrame:
    return pd.DataFrame(columns=MATCHUP_COLS)


def merge_defender_features(pg: pd.DataFrame, matchups: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    if matchups is None or matchups.empty:
        return pg.assign(**{c: np.nan for c in DEFENDER_FEATURES})

    m = matchups[MATCHUP_COLS].merge(games[["id", "date"]].rename(columns={"id": "game_id"}), on="game_id")
    m = m[m["partial_poss"] > 0].copy()
    if m.empty:
        return pg.assign(**{c: np.nan for c in DEFENDER_FEATURES})
    m["date"] = pd.to_datetime(m["date"])
    coverage_start = m["date"].min()

    m = _expected_points(m)
    def_hist = _cumulative(m, ["def_player_id"], {"pts": "pts", "exp": "exp"})
    def_hist["rating"] = (def_hist["pts"] + K_DEF_PTS) / (def_hist["exp"] + K_DEF_PTS)
    pair_hist = _cumulative(m, ["off_player_id", "def_player_id"], {"partial_poss": "poss", "pts": "pts", "exp": "exp"})

    # Tonight's defenders: every player in the game on the other team (actual for past games, expected for upcoming).
    players = pg[["player_id", "team_id", "game_id", "date", "min_l10", "pos_group"]].copy()
    players["date"] = pd.to_datetime(players["date"])
    defenders = players.rename(columns={"player_id": "def_player_id", "team_id": "def_team_id", "pos_group": "def_pos"})
    defenders["w"] = defenders["min_l10"].fillna(DEFAULT_MINUTES).clip(lower=1.0)
    defenders = _asof(defenders, def_hist[["def_player_id", "date", "rating"]], ["def_player_id"])
    defenders["rating"] = defenders["rating"].fillna(1.0)  # no tracked history yet: treat as average

    team = _weighted(defenders, ["game_id", "def_team_id"], "rating", "w").rename("opp_def_rating_team").reset_index()
    team = team.merge(defenders[["game_id", "def_team_id", "date"]].drop_duplicates(["game_id", "def_team_id"]), on=["game_id", "def_team_id"])
    team = team.sort_values(["def_team_id", "date", "game_id"])
    usual = team.groupby("def_team_id")["opp_def_rating_team"].transform(lambda s: s.shift(1).rolling(USUAL_WINDOW, min_periods=1).mean())
    team["opp_def_rating_vs_usual"] = team["opp_def_rating_team"] - usual
    pos = _weighted(defenders, ["game_id", "def_team_id", "def_pos"], "rating", "w").rename("opp_def_rating_pos").reset_index()

    out = pg.merge(team.drop(columns="date").rename(columns={"def_team_id": "opp_team_id"}), on=["game_id", "opp_team_id"], how="left")
    out = out.merge(pos.rename(columns={"def_team_id": "opp_team_id", "def_pos": "pos_group"}), on=["game_id", "opp_team_id", "pos_group"], how="left")

    # Likely defenders: this player's past possessions against each of tonight's defenders.
    pairs = players[["player_id", "team_id", "game_id", "date"]].merge(
        defenders[["def_player_id", "def_team_id", "game_id", "rating"]], on="game_id"
    )
    pairs = pairs[pairs["team_id"] != pairs["def_team_id"]].rename(columns={"player_id": "off_player_id"})
    pairs = _asof(pairs, pair_hist, ["off_player_id", "def_player_id"])
    pairs = pairs[pairs["poss"].fillna(0) > 0]
    if pairs.empty:
        likely = pd.DataFrame(columns=["player_id", "game_id", "likely_def_rating", "likely_def_poss", "vs_likely_def_ratio"])
    else:
        pairs["rating_x_poss"] = pairs["rating"] * pairs["poss"]
        g = pairs.groupby(["off_player_id", "game_id"])[["poss", "pts", "exp", "rating_x_poss"]].sum()
        likely = pd.DataFrame({
            "likely_def_rating": g["rating_x_poss"] / g["poss"],
            "likely_def_poss": g["poss"],
            "vs_likely_def_ratio": (g["pts"] + K_PAIR_PTS) / (g["exp"] + K_PAIR_PTS),
        }).reset_index().rename(columns={"off_player_id": "player_id"})
    out = out.merge(likely, on=["player_id", "game_id"], how="left")

    # Before the first tracked game there is nothing to go on; after it, no shared history means none of them has guarded this player yet.
    before = pd.to_datetime(out["date"]) <= coverage_start
    out.loc[before, DEFENDER_FEATURES] = np.nan
    out.loc[~before, "likely_def_poss"] = out.loc[~before, "likely_def_poss"].fillna(0.0)
    return out


def _expected_points(m: pd.DataFrame) -> pd.DataFrame:
    """Points each matchup 'should' have produced: possessions x the scorer's own rate in games before this one."""
    league_rate = m["pts"].sum() / m["partial_poss"].sum()
    by_game = m.groupby(["off_player_id", "date"], as_index=False)[["pts", "partial_poss"]].sum().sort_values(["off_player_id", "date"])
    grp = by_game.groupby("off_player_id")
    prev_pts = grp["pts"].cumsum() - by_game["pts"]
    prev_poss = grp["partial_poss"].cumsum() - by_game["partial_poss"]
    by_game["off_rate"] = (prev_pts + K_OFF_POSS * league_rate) / (prev_poss + K_OFF_POSS)
    m = m.merge(by_game[["off_player_id", "date", "off_rate"]], on=["off_player_id", "date"], how="left")
    m["exp"] = m["partial_poss"] * m["off_rate"]
    return m


def _cumulative(m: pd.DataFrame, keys: list[str], cols: dict[str, str]) -> pd.DataFrame:
    """Running totals per key through each date (inclusive); looked up strictly before a game's date."""
    daily = m.groupby([*keys, "date"], as_index=False)[list(cols)].sum().rename(columns=cols)
    daily = daily.sort_values([*keys, "date"])
    vals = list(cols.values())
    daily[vals] = daily.groupby(keys)[vals].cumsum()
    return daily


def _asof(left: pd.DataFrame, right: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """For each left row, the latest right row with the same `by` keys dated strictly before the left row's date."""
    left = left.reset_index(drop=True)
    left["_row"] = np.arange(len(left))
    l_sorted = left.sort_values("date")
    r_sorted = right.sort_values("date")
    for k in by:  # merge_asof needs identical key dtypes
        r_sorted[k] = r_sorted[k].astype(l_sorted[k].dtype)
    merged = pd.merge_asof(l_sorted, r_sorted, on="date", by=by, allow_exact_matches=False, direction="backward")
    return merged.sort_values("_row").drop(columns="_row").reset_index(drop=True)


def _weighted(df: pd.DataFrame, keys: list[str], value: str, weight: str) -> pd.Series:
    tmp = df.assign(_vw=df[value] * df[weight])
    g = tmp.groupby(keys)[["_vw", weight]].sum()
    return g["_vw"] / g[weight]
