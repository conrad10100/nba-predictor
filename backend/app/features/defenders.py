"""Opponent defender features: how good tonight's opposing defenders are.

Ratings are Basketball-Reference's DBPM (defensive box plus/minus: points per 100 possessions better than an
average player; higher = better defender), as published before the game (app.ingest.defense):
- this season's most recent daily snapshot stored before the game, weighted by minutes played so far
- blended with last season's final DBPM as a prior (itself shrunk toward 0 for players with few minutes)
A player with neither counts as average (0).

For each player-game:
- opp_def_dbpm_team: tonight's opposing players, weighted by their recent minutes
- opp_def_dbpm_pos: the same, only opponents in the player's position group (G/F/C)
- opp_def_dbpm_vs_usual: tonight's team value minus the opponent's usual one (negative = a good defender is out)
- opp_pos_starter_def_dbpm: the opponent at the player's position with the most recent minutes (the likeliest
  primary defender; actual who-guards-whom tracking is only on stats.nba.com)

Rows for a season with no ratings published before the game are left missing, so the model can tell "no data"
from "average defenders".
"""

import numpy as np
import pandas as pd

DEFENDER_FEATURES = ["opp_def_dbpm_team", "opp_def_dbpm_pos", "opp_def_dbpm_vs_usual", "opp_pos_starter_def_dbpm"]
RATING_COLS = ["player_id", "season", "as_of", "dbpm", "minutes"]

PRIOR_MINUTES = 800.0  # last season's rating counts like this many minutes of this season's
PREV_SHRINK_MINUTES = 500.0  # last season's rating is shrunk toward 0 by this many minutes
DEFAULT_MINUTES = 15.0  # weight for a player with no recent minutes (e.g. first game back)
USUAL_WINDOW = 10


def empty_ratings() -> pd.DataFrame:
    return pd.DataFrame(columns=RATING_COLS)


def previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[2:]}"


def season_end(season: str) -> pd.Timestamp:
    """When a past season's ratings count as final (published): the July 1 after it."""
    return pd.Timestamp(int(season[:4]) + 1, 7, 1)


def player_ratings(players: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Rows with player_id, season, date -> the same rows plus `dbpm` as published before `date` (NaN: none yet)."""
    p = players.reset_index(drop=True).copy()
    p["date"] = pd.to_datetime(p["date"])
    r = ratings[RATING_COLS].copy()
    r["as_of"] = pd.to_datetime(r["as_of"])
    r["player_id"] = r["player_id"].astype(p["player_id"].dtype)
    final = r[r["as_of"] == r["season"].map(season_end)]

    # Last season's final rating, shrunk by how many minutes it's based on.
    p["prev_season"] = p["season"].map(previous_season)
    prev = final.rename(columns={"season": "prev_season", "dbpm": "prev_dbpm", "minutes": "prev_min"})
    p = p.merge(prev[["player_id", "prev_season", "prev_dbpm", "prev_min"]], on=["player_id", "prev_season"], how="left")
    prior = (p["prev_dbpm"] * p["prev_min"] / (p["prev_min"] + PREV_SHRINK_MINUTES)).fillna(0.0)

    # This season's latest rating published strictly before the game.
    p["_row"] = np.arange(len(p))
    snaps = r.rename(columns={"as_of": "date", "dbpm": "cur_dbpm", "minutes": "cur_min"}).sort_values("date")
    merged = pd.merge_asof(p.sort_values("date"), snaps[["player_id", "season", "date", "cur_dbpm", "cur_min"]],
                           on="date", by=["player_id", "season"], allow_exact_matches=False, direction="backward")
    merged = merged.sort_values("_row").reset_index(drop=True)
    cur_min = merged["cur_min"].fillna(0.0)
    p["dbpm"] = (merged["cur_dbpm"].fillna(0.0) * cur_min + prior * PRIOR_MINUTES) / (cur_min + PRIOR_MINUTES)

    # Anything published for this season by then? (last season's final table, or a snapshot of this one)
    has_prev = p["prev_season"].isin(set(final["season"]))
    first_snap = p["season"].map(r.groupby("season")["as_of"].min())
    has_snap = (p["date"] > first_snap).fillna(False)
    p.loc[~(has_prev | has_snap), "dbpm"] = np.nan
    return p.drop(columns=["_row", "prev_season", "prev_dbpm", "prev_min"])


def merge_defender_features(pg: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    if ratings is None or ratings.empty:
        return pg.assign(**{c: np.nan for c in DEFENDER_FEATURES})

    # Tonight's opponents: every player in the game on the other team (actual for past games, expected for upcoming).
    defenders = player_ratings(pg[["player_id", "team_id", "game_id", "date", "season", "min_l10", "pos_group"]], ratings)
    defenders = defenders.rename(columns={"player_id": "def_player_id", "team_id": "def_team_id", "pos_group": "def_pos"})
    defenders["w"] = defenders["min_l10"].fillna(DEFAULT_MINUTES).clip(lower=1.0)

    team = _weighted(defenders, ["game_id", "def_team_id"], "dbpm", "w").rename("opp_def_dbpm_team").reset_index()
    team = team.merge(defenders[["game_id", "def_team_id", "date"]].drop_duplicates(["game_id", "def_team_id"]), on=["game_id", "def_team_id"])
    team = team.sort_values(["def_team_id", "date", "game_id"])
    usual = team.groupby("def_team_id")["opp_def_dbpm_team"].transform(lambda s: s.shift(1).rolling(USUAL_WINDOW, min_periods=1).mean())
    team["opp_def_dbpm_vs_usual"] = team["opp_def_dbpm_team"] - usual
    pos = _weighted(defenders, ["game_id", "def_team_id", "def_pos"], "dbpm", "w").rename("opp_def_dbpm_pos").reset_index()
    starter = (defenders.sort_values("w", ascending=False, kind="stable").drop_duplicates(["game_id", "def_team_id", "def_pos"])
               [["game_id", "def_team_id", "def_pos", "dbpm"]].rename(columns={"dbpm": "opp_pos_starter_def_dbpm"}))

    keys = {"def_team_id": "opp_team_id", "def_pos": "pos_group"}
    out = pg.merge(team.drop(columns="date").rename(columns=keys), on=["game_id", "opp_team_id"], how="left")
    out = out.merge(pos.rename(columns=keys), on=["game_id", "opp_team_id", "pos_group"], how="left")
    return out.merge(starter.rename(columns=keys), on=["game_id", "opp_team_id", "pos_group"], how="left")


def _weighted(df: pd.DataFrame, keys: list[str], value: str, weight: str) -> pd.Series:
    """Weighted mean of `value` per group; NaN when every value in the group is NaN."""
    ok = df[value].notna()
    tmp = df.assign(_vw=df[value].where(ok, 0.0) * df[weight], _w=df[weight].where(ok, 0.0))
    g = tmp.groupby(keys)[["_vw", "_w"]].sum()
    return (g["_vw"] / g["_w"]).replace([np.inf, -np.inf], np.nan)
