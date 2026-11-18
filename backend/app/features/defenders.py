"""Opponent defender features: how good tonight's opposing defenders are.

Built from on-court data (ESPN play-by-play, app.ingest.on_court). Each defender gets an on/off rating from games
strictly before the row's date: the rate their team allows points with them on the floor divided by the rate with them
off (0.95 = the team gives up 5% fewer points with them on). Both rates are shrunk toward the league average, so a
player with little floor time counts as neutral (1.00) rather than great or terrible.

For each player-game:
- opp_def_rating_team: tonight's opposing players, weighted by their recent minutes
- opp_def_rating_pos: the same, only opponents in the player's position group (G/F/C)
- opp_def_rating_vs_usual: tonight's team rating minus the opponent's usual one (positive = a good defender is out)
- opp_pos_starter_def_rating: the opponent at the player's position with the most recent minutes (the likeliest
  primary defender; actual matchup tracking isn't available from ESPN)

Rows before the first game with on-court data are left missing, so the model can tell "no data" from "average".
"""

import numpy as np
import pandas as pd

DEFENDER_FEATURES = ["opp_def_rating_team", "opp_def_rating_pos", "opp_def_rating_vs_usual", "opp_pos_starter_def_rating"]
ON_COURT_COLS = ["game_id", "player_id", "team_id", "on_seconds", "opp_pts_on", "game_seconds", "opp_pts_game"]

K_SECONDS = 7200.0  # each rate is shrunk toward league average by this much floor time (~4 full games for a starter)
DEFAULT_MINUTES = 15.0  # weight for a player with no recent minutes (e.g. first game back)
USUAL_WINDOW = 10


def empty_on_court() -> pd.DataFrame:
    return pd.DataFrame(columns=ON_COURT_COLS)


def merge_defender_features(pg: pd.DataFrame, on_court: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    oc = on_court[ON_COURT_COLS] if on_court is not None and not on_court.empty else empty_on_court()
    oc = oc.merge(games[["id", "date"]].rename(columns={"id": "game_id"}), on="game_id")
    oc = oc[oc["game_seconds"] > 0].copy()
    if oc.empty:
        return pg.assign(**{c: np.nan for c in DEFENDER_FEATURES})
    oc["date"] = pd.to_datetime(oc["date"])
    coverage_start = oc["date"].min()

    oc["off_seconds"] = (oc["game_seconds"] - oc["on_seconds"]).clip(lower=0)
    oc["opp_pts_off"] = (oc["opp_pts_game"] - oc["opp_pts_on"]).clip(lower=0)
    league_rate = oc["opp_pts_game"].sum() / oc["game_seconds"].sum()  # one game per team-row; ratio is unaffected
    cols = ["on_seconds", "opp_pts_on", "off_seconds", "opp_pts_off"]
    hist = oc.groupby(["player_id", "date"], as_index=False)[cols].sum().sort_values(["player_id", "date"])
    hist[cols] = hist.groupby("player_id")[cols].cumsum()
    on_rate = (hist["opp_pts_on"] + K_SECONDS * league_rate) / (hist["on_seconds"] + K_SECONDS)
    off_rate = (hist["opp_pts_off"] + K_SECONDS * league_rate) / (hist["off_seconds"] + K_SECONDS)
    hist["rating"] = on_rate / off_rate

    # Tonight's opponents: every player in the game on the other team (actual for past games, expected for upcoming).
    defenders = pg[["player_id", "team_id", "game_id", "date", "min_l10", "pos_group"]].rename(
        columns={"player_id": "def_player_id", "team_id": "def_team_id", "pos_group": "def_pos"}
    )
    defenders["date"] = pd.to_datetime(defenders["date"])
    defenders["w"] = defenders["min_l10"].fillna(DEFAULT_MINUTES).clip(lower=1.0)
    defenders = _asof(defenders, hist[["player_id", "date", "rating"]].rename(columns={"player_id": "def_player_id"}), ["def_player_id"])
    defenders["rating"] = defenders["rating"].fillna(1.0)  # no on-court history yet: treat as neutral

    team = _weighted(defenders, ["game_id", "def_team_id"], "rating", "w").rename("opp_def_rating_team").reset_index()
    team = team.merge(defenders[["game_id", "def_team_id", "date"]].drop_duplicates(["game_id", "def_team_id"]), on=["game_id", "def_team_id"])
    team = team.sort_values(["def_team_id", "date", "game_id"])
    usual = team.groupby("def_team_id")["opp_def_rating_team"].transform(lambda s: s.shift(1).rolling(USUAL_WINDOW, min_periods=1).mean())
    team["opp_def_rating_vs_usual"] = team["opp_def_rating_team"] - usual
    pos = _weighted(defenders, ["game_id", "def_team_id", "def_pos"], "rating", "w").rename("opp_def_rating_pos").reset_index()
    starter = (defenders.sort_values("w", ascending=False).drop_duplicates(["game_id", "def_team_id", "def_pos"])
               [["game_id", "def_team_id", "def_pos", "rating"]].rename(columns={"rating": "opp_pos_starter_def_rating"}))

    keys = {"def_team_id": "opp_team_id", "def_pos": "pos_group"}
    out = pg.merge(team.drop(columns="date").rename(columns=keys), on=["game_id", "opp_team_id"], how="left")
    out = out.merge(pos.rename(columns=keys), on=["game_id", "opp_team_id", "pos_group"], how="left")
    out = out.merge(starter.rename(columns=keys), on=["game_id", "opp_team_id", "pos_group"], how="left")
    out.loc[pd.to_datetime(out["date"]) <= coverage_start, DEFENDER_FEATURES] = np.nan
    return out


def _asof(left: pd.DataFrame, right: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """For each left row, the latest right row with the same `by` keys dated strictly before the left row's date."""
    left = left.reset_index(drop=True)
    left["_row"] = np.arange(len(left))
    l_sorted = left.sort_values("date")
    r_sorted = right.sort_values("date").copy()
    for k in by:  # merge_asof needs identical key dtypes
        r_sorted[k] = r_sorted[k].astype(l_sorted[k].dtype)
    merged = pd.merge_asof(l_sorted, r_sorted, on="date", by=by, allow_exact_matches=False, direction="backward")
    return merged.sort_values("_row").drop(columns="_row").reset_index(drop=True)


def _weighted(df: pd.DataFrame, keys: list[str], value: str, weight: str) -> pd.Series:
    tmp = df.assign(_vw=df[value] * df[weight])
    g = tmp.groupby(keys)[["_vw", weight]].sum()
    return g["_vw"] / g[weight]
