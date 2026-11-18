"""Feature engineering.

Every rolling stat is computed from games strictly BEFORE the row's game (shift(1) then roll), so the same code
builds leak-free training rows for history and feature rows for upcoming games (whose stats are NaN).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.db import engine
from app.features.defenders import DEFENDER_FEATURES, RATING_COLS, empty_ratings, merge_defender_features
from app.models import DefenderRating, Game, GameLine, Player, PlayerGameStat, Team, TeamGameStat

TARGETS = ["pts", "reb", "ast"]
PLAYER_ROLL_COLS = ["pts", "reb", "ast", "min", "fga", "fta", "fg3a", "usg"]
TEAM_ROLL_COLS = [
    "pace", "off_rtg", "def_rtg", "pts_allowed", "reb_allowed", "ast_allowed", "reb_allowed_100", "ast_allowed_100",
    "ts_allowed", "fta_rate_allowed", "fg3a_rate_allowed", "tov_forced_100",
    "missed_fg", "oreb_rate", "efg", "net_rtg",
]
OWN_TEAM_COLS = ["pace", "off_rtg", "efg", "missed_fg", "oreb_rate", "net_rtg"]
H2H_COLS = ["pace", "pts", "pts_allowed"]  # team-level averages from the last few meetings of these two teams
H2H_WINDOW = 4
# Shooting efficiency and per-36 rates, computed as ratio-of-sums over the window (a 1-for-1 night shouldn't count like 12-for-20).
RATES = {
    "ts": (["pts"], ["fga", "fta"], lambda s: s["pts"] / (2 * (s["fga"] + 0.44 * s["fta"]))),
    "efg": (["fgm", "fg3m"], ["fga"], lambda s: (s["fgm"] + 0.5 * s["fg3m"]) / s["fga"]),
    "fg3_pct": (["fg3m"], ["fg3a"], lambda s: s["fg3m"] / s["fg3a"]),
    "ft_pct": (["ftm"], ["fta"], lambda s: s["ftm"] / s["fta"]),
    "fg3a_rate": (["fg3a"], ["fga"], lambda s: s["fg3a"] / s["fga"]),
    "fta_rate": (["fta"], ["fga"], lambda s: s["fta"] / s["fga"]),
    "pts_per36": (["pts"], ["min"], lambda s: 36 * s["pts"] / s["min"]),
    "fga_per36": (["fga"], ["min"], lambda s: 36 * s["fga"] / s["min"]),
    "reb_per36": (["reb"], ["min"], lambda s: 36 * s["reb"] / s["min"]),
    "oreb_per36": (["oreb"], ["min"], lambda s: 36 * s["oreb"] / s["min"]),
    "dreb_per36": (["dreb"], ["min"], lambda s: 36 * s["dreb"] / s["min"]),
    "ast_per36": (["ast"], ["min"], lambda s: 36 * s["ast"] / s["min"]),
    "tov_per36": (["tov"], ["min"], lambda s: 36 * s["tov"] / s["min"]),
    "pf_per36": (["pf"], ["min"], lambda s: 36 * s["pf"] / s["min"]),  # foul trouble caps minutes
}
RATES_WINDOWS = {"l5": 5, "l10": 10, "season": None}
OUT_STATUSES = {"Out", "Doubtful"}

# Missing-teammate detection: a teammate is "expected" if they played >= 3 of the team's previous 5 games
# averaging >= 15 minutes in those appearances.
ROTATION_WINDOW, ROTATION_MIN_GAMES, ROTATION_MIN_MINUTES = 5, 3, 15.0

# Sportsbook view of the game, from this team's side. Missing for games before lines were recorded.
LINE_FEATURES = ["game_total", "team_spread", "implied_team_pts", "implied_opp_pts"]

FEATURES = (
    [f"{c}_{w}" for c in PLAYER_ROLL_COLS for w in ("l5", "l10", "season", "ewm")]
    + [f"{k}_{w}" for k in RATES for w in RATES_WINDOWS]
    + ["ts_trend"]
    + [f"{t}_std_l10" for t in TARGETS]
    + [f"{t}_vs_opp" for t in TARGETS] + [f"{t}_last_vs_opp" for t in TARGETS] + ["games_vs_opp"]
    + [f"{t}_home_away" for t in TARGETS]
    + ["games_played_season", "days_since_last_game", "min_trend"]
    + [f"team_{c}_{w}" for c in OWN_TEAM_COLS for w in ("season", "l10")]
    + [f"h2h_{c}" for c in H2H_COLS] + ["h2h_games", "net_rtg_diff", "abs_net_rtg_diff"]
    + [f"opp_{c}_{w}" for c in TEAM_ROLL_COLS for w in ("season", "l10")]
    + ["projected_pace"]
    + [f"opp_pos_{t}_allowed" for t in TARGETS]
    + ["rest_days", "b2b", "three_in_four", "away_streak", "opp_rest_days", "opp_b2b", "rest_advantage"]
    + ["is_home", "same_conference", "same_division", "team_game_number"]
    + LINE_FEATURES
    + ["missing_teammates", "missing_teammate_pts", "missing_teammate_reb", "missing_teammate_ast", "missing_teammate_min"]
    + DEFENDER_FEATURES  # only used when the model's `defender_features` setting is on (see app.ml.train.usable_features)
)


@dataclass
class Frames:
    player_stats: pd.DataFrame  # one row per player-game (final games only)
    team_stats: pd.DataFrame  # one row per team-game (final games only)
    games: pd.DataFrame
    players: pd.DataFrame
    teams: pd.DataFrame
    lines: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["game_id", "home_spread", "total"]))
    defender_ratings: pd.DataFrame = field(default_factory=empty_ratings)  # DBPM snapshots (defender_ratings)


def load_frames() -> Frames:
    with engine.connect() as conn:
        games = pd.read_sql(select(Game), conn, parse_dates=["date"])
        ps = pd.read_sql(select(PlayerGameStat), conn)
        ts = pd.read_sql(select(TeamGameStat), conn)
        players = pd.read_sql(select(Player), conn)
        teams = pd.read_sql(select(Team), conn)
        lines = pd.read_sql(select(GameLine.game_id, GameLine.home_spread, GameLine.total), conn)
        ratings = pd.read_sql(select(*[getattr(DefenderRating, c) for c in RATING_COLS]), conn)
    return Frames(ps, ts, games, players, teams, lines, ratings)


@dataclass
class Upcoming:
    """Games to build feature rows for, and who is expected to play in them."""

    games: pd.DataFrame  # columns of Game
    roster: pd.DataFrame  # player_id, team_id, game_id
    out: set[int] = field(default_factory=set)  # players ruled out (injury report + user overrides)
    b2b_overrides: dict[int, bool] = field(default_factory=dict)  # team_id -> force B2B on/off


def build_features(fr: Frames, upcoming: Upcoming | None = None) -> pd.DataFrame:
    final_games = fr.games[fr.games["status"] == "final"]
    games = pd.concat([final_games, upcoming.games]) if upcoming is not None else final_games
    games = games.drop_duplicates("id", keep="last")

    tg = _team_frame(fr, games, upcoming)
    pg = _player_frame(fr, games, tg, upcoming)
    pg = _player_rolling(pg)
    pg = _merge_team_context(pg, tg, fr.teams)
    pg = _merge_positional_defense(pg, fr.players)
    pg = _merge_missing_teammates(pg, upcoming)
    pg = merge_defender_features(pg, fr.defender_ratings)
    return pg.copy()


# ---------------------------------------------------------------- team-level


def _team_frame(fr: Frames, games: pd.DataFrame, upcoming: Upcoming | None) -> pd.DataFrame:
    home = games.assign(team_id=games["home_team_id"], opp_team_id=games["away_team_id"], is_home=1)
    away = games.assign(team_id=games["away_team_id"], opp_team_id=games["home_team_id"], is_home=0)
    tg = pd.concat([home, away])[["id", "date", "season", "team_id", "opp_team_id", "is_home"]].rename(columns={"id": "game_id"})
    opp_box = fr.team_stats[["team_id", "game_id", "fga", "fta", "fg3a", "tov", "dreb"]].rename(
        columns={"team_id": "opp_team_id", "fga": "opp_fga", "fta": "opp_fta", "fg3a": "opp_fg3a", "tov": "opp_tov", "dreb": "opp_dreb"}
    )
    stats = fr.team_stats.merge(opp_box, on=["opp_team_id", "game_id"], how="left").drop(columns=["opp_team_id", "is_home"])
    stats["reb_allowed_100"] = 100 * stats["reb_allowed"] / stats["possessions"]
    stats["ast_allowed_100"] = 100 * stats["ast_allowed"] / stats["possessions"]
    # How efficiently opponents score against this defense, and how it gets scored on (FTs, threes) or forces turnovers.
    stats["ts_allowed"] = stats["pts_allowed"] / (2 * (stats["opp_fga"] + 0.44 * stats["opp_fta"]))
    stats["fta_rate_allowed"] = stats["opp_fta"] / stats["opp_fga"]
    stats["fg3a_rate_allowed"] = stats["opp_fg3a"] / stats["opp_fga"]
    stats["tov_forced_100"] = 100 * stats["opp_tov"] / stats["possessions"]
    # Rebound chances (own misses), glass-crashing, teammates' shot-making (feeds assists), overall strength.
    stats["missed_fg"] = stats["fga"] - stats["fgm"]
    stats["oreb_rate"] = stats["oreb"] / (stats["oreb"] + stats["opp_dreb"])
    stats["efg"] = (stats["fgm"] + 0.5 * stats["fg3m"]) / stats["fga"]
    stats["net_rtg"] = stats["off_rtg"] - stats["def_rtg"]
    stats = stats.drop(columns=["opp_fga", "opp_fta", "opp_fg3a", "opp_tov", "opp_dreb"])
    tg = tg.merge(stats, on=["team_id", "game_id"], how="left")
    tg = tg.merge(fr.lines[["game_id", "home_spread", "total"]], on="game_id", how="left")
    tg["team_spread"] = np.where(tg["is_home"] == 1, tg["home_spread"], -tg["home_spread"])  # negative = favored
    tg["game_total"] = tg["total"]
    tg["implied_team_pts"] = tg["game_total"] / 2 - tg["team_spread"] / 2
    tg["implied_opp_pts"] = tg["game_total"] / 2 + tg["team_spread"] / 2
    tg = tg.sort_values(["team_id", "date", "game_id"]).reset_index(drop=True)

    by_team = tg.groupby("team_id")
    prev = by_team[TEAM_ROLL_COLS].shift(1)
    tg[[f"{c}_l10" for c in TEAM_ROLL_COLS]] = prev.groupby(tg["team_id"]).rolling(10, min_periods=1).mean().reset_index(level=0, drop=True).to_numpy()
    prev_s = tg.groupby(["team_id", "season"])[TEAM_ROLL_COLS].shift(1)
    tg[[f"{c}_season" for c in TEAM_ROLL_COLS]] = (
        prev_s.groupby([tg["team_id"], tg["season"]]).expanding().mean().reset_index(level=[0, 1], drop=True).sort_index().to_numpy()
    )

    # Head-to-head: how these two teams' recent meetings played out (carries across seasons).
    pair = [tg["team_id"], tg["opp_team_id"]]
    h2h_prev = tg.groupby(["team_id", "opp_team_id"])[H2H_COLS].shift(1)
    tg[[f"h2h_{c}" for c in H2H_COLS]] = (
        h2h_prev.groupby(pair).rolling(H2H_WINDOW, min_periods=1).mean().reset_index(level=[0, 1], drop=True).sort_index().to_numpy()
    )
    tg["h2h_games"] = tg.groupby(["team_id", "opp_team_id"]).cumcount()

    prev_date = by_team["date"].shift(1)
    tg["rest_days"] = (tg["date"] - prev_date).dt.days.clip(upper=7)
    tg["b2b"] = (tg["rest_days"] == 1).astype(int)
    tg["three_in_four"] = ((tg["date"] - by_team["date"].shift(2)).dt.days <= 3).astype(int)
    block = (tg["is_home"] != by_team["is_home"].shift(1)).cumsum()
    tg["away_streak"] = np.where(tg["is_home"] == 0, tg.groupby(block).cumcount() + 1, 0)
    tg["team_game_number"] = tg.groupby(["team_id", "season"]).cumcount() + 1

    if upcoming is not None and upcoming.b2b_overrides:
        upcoming_ids = set(upcoming.games["id"])
        for team_id, is_b2b in upcoming.b2b_overrides.items():
            mask = (tg["team_id"] == team_id) & tg["game_id"].isin(upcoming_ids)
            tg.loc[mask, "b2b"] = int(is_b2b)
            tg.loc[mask, "rest_days"] = 1 if is_b2b else tg.loc[mask, "rest_days"].clip(lower=2)
    return tg


# ---------------------------------------------------------------- player-level


def _player_frame(fr: Frames, games: pd.DataFrame, tg: pd.DataFrame, upcoming: Upcoming | None) -> pd.DataFrame:
    ps = fr.player_stats[fr.player_stats["min"] > 0]
    if upcoming is not None:
        ps = pd.concat([ps, upcoming.roster[~upcoming.roster["player_id"].isin(upcoming.out)]])
    team_totals = tg[["team_id", "game_id", "opp_team_id", "is_home", "min", "fga", "fta", "tov"]].rename(
        columns={"min": "team_min", "fga": "team_fga", "fta": "team_fta", "tov": "team_tov"}
    )
    pg = ps.drop(columns=["opp_team_id"], errors="ignore").merge(team_totals, on=["team_id", "game_id"], how="inner")
    pg = pg.merge(games[["id", "date", "season"]].rename(columns={"id": "game_id"}), on="game_id")
    # Usage rate: share of team possessions a player uses while on the floor.
    pg["usg"] = 100 * (pg["fga"] + 0.44 * pg["fta"] + pg["tov"]) * (pg["team_min"] / 5) / (
        pg["min"] * (pg["team_fga"] + 0.44 * pg["team_fta"] + pg["team_tov"])
    )
    return pg.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)


def _player_rolling(pg: pd.DataFrame) -> pd.DataFrame:
    pid = pg["player_id"]
    prev = pg.groupby("player_id")[PLAYER_ROLL_COLS].shift(1)
    by_player = prev.groupby(pid)
    for w in (5, 10):
        pg[[f"{c}_l{w}" for c in PLAYER_ROLL_COLS]] = by_player.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).to_numpy()
    pg[[f"{c}_ewm" for c in PLAYER_ROLL_COLS]] = by_player.transform(lambda s: s.ewm(span=10, ignore_na=True).mean()).to_numpy()
    pg[[f"{t}_std_l10" for t in TARGETS]] = prev[TARGETS].groupby(pid).rolling(10, min_periods=3).std().reset_index(level=0, drop=True).to_numpy()

    prev_s = pg.groupby(["player_id", "season"])[PLAYER_ROLL_COLS].shift(1)
    season_avg = prev_s.groupby([pid, pg["season"]]).expanding().mean().reset_index(level=[0, 1], drop=True).sort_index()
    pg[[f"{c}_season" for c in PLAYER_ROLL_COLS]] = season_avg.to_numpy()
    pg["games_played_season"] = pg.groupby(["player_id", "season"]).cumcount()

    pg["days_since_last_game"] = pg.groupby("player_id")["date"].diff().dt.days.clip(upper=30)
    pg["min_trend"] = pg["min_l5"] - pg["min_season"]

    pg = _rates(pg.copy())
    pg["ts_trend"] = pg["ts_l5"] - pg["ts_season"]  # hot/cold shooting relative to their norm

    pg = _expanding_split(pg, ["player_id", "opp_team_id"], "vs_opp")
    pg["games_vs_opp"] = pg.groupby(["player_id", "opp_team_id"]).cumcount()
    pg[[f"{t}_last_vs_opp" for t in TARGETS]] = pg.groupby(["player_id", "opp_team_id"])[TARGETS].shift(1).to_numpy()
    pg = _expanding_split(pg, ["player_id", "is_home"], "home_away")
    return pg


def _rates(pg: pd.DataFrame) -> pd.DataFrame:
    raw = sorted({c for num, den, _ in RATES.values() for c in num + den})
    pid = pg["player_id"]
    new = {}
    for name, n in RATES_WINDOWS.items():
        if n is None:
            keys = [pid, pg["season"]]
            prev = pg.groupby(["player_id", "season"])[raw].shift(1)
            sums = prev.groupby(keys).cumsum()
        else:
            prev = pg.groupby("player_id")[raw].shift(1)
            sums = prev.groupby(pid).rolling(n, min_periods=1).sum().reset_index(level=0, drop=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            for key, (_, _, fn) in RATES.items():
                new[f"{key}_{name}"] = fn(sums).replace([np.inf, -np.inf], np.nan).to_numpy()
    return pd.concat([pg, pd.DataFrame(new, index=pg.index)], axis=1)


def _expanding_split(pg: pd.DataFrame, keys: list[str], suffix: str) -> pd.DataFrame:
    prev = pg.groupby(keys)[TARGETS].shift(1)
    means = prev.groupby([pg[k] for k in keys]).expanding().mean().reset_index(level=list(range(len(keys))), drop=True).sort_index()
    pg[[f"{t}_{suffix}" for t in TARGETS]] = means.to_numpy()
    return pg


# ---------------------------------------------------------------- context merges


def _merge_team_context(pg: pd.DataFrame, tg: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    own_cols = [f"{c}_{w}" for c in OWN_TEAM_COLS for w in ("season", "l10")]
    own = tg[["team_id", "game_id", *own_cols, *[f"h2h_{c}" for c in H2H_COLS], "h2h_games",
              "rest_days", "b2b", "three_in_four", "away_streak", "team_game_number", *LINE_FEATURES]]
    own = own.rename(columns={c: f"team_{c}" for c in own_cols})
    opp_cols = [f"{c}_{w}" for c in TEAM_ROLL_COLS for w in ("season", "l10")]
    opp = tg[["team_id", "game_id", *opp_cols, "rest_days", "b2b"]].rename(
        columns={"team_id": "opp_team_id", "rest_days": "opp_rest_days", "b2b": "opp_b2b", **{c: f"opp_{c}" for c in opp_cols}}
    )
    pg = pg.merge(own, on=["team_id", "game_id"], how="left").merge(opp, on=["opp_team_id", "game_id"], how="left")
    pg["projected_pace"] = pg[["team_pace_season", "opp_pace_season"]].mean(axis=1)
    # Big mismatches mean blowouts, and blowouts mean starters sit the 4th quarter.
    pg["net_rtg_diff"] = pg["team_net_rtg_season"] - pg["opp_net_rtg_season"]
    pg["abs_net_rtg_diff"] = pg["net_rtg_diff"].abs()
    pg["rest_advantage"] = pg["rest_days"] - pg["opp_rest_days"]

    conf = teams.set_index("id")["conference"]
    div = teams.set_index("id")["division"]
    pg["same_conference"] = (pg["team_id"].map(conf) == pg["opp_team_id"].map(conf)).astype(int)
    pg["same_division"] = (pg["team_id"].map(div) == pg["opp_team_id"].map(div)).astype(int)
    return pg


def _merge_positional_defense(pg: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date points/rebounds/assists the opponent allows per game to the player's position group."""
    pg = pg.merge(players[["id", "pos_group"]].rename(columns={"id": "player_id"}), on="player_id", how="left")
    pg["pos_group"] = pg["pos_group"].fillna("F")
    dvp = pg.groupby(["opp_team_id", "game_id", "date", "season", "pos_group"], as_index=False)[TARGETS].sum(min_count=1)
    dvp = dvp.sort_values(["opp_team_id", "pos_group", "date"]).reset_index(drop=True)
    keys = [dvp["opp_team_id"], dvp["pos_group"], dvp["season"]]
    prev = dvp.groupby(keys)[TARGETS].shift(1)
    means = prev.groupby(keys).expanding().mean().reset_index(level=[0, 1, 2], drop=True).sort_index()
    dvp[[f"opp_pos_{t}_allowed" for t in TARGETS]] = means.to_numpy()
    cols = ["opp_team_id", "game_id", "pos_group", *[f"opp_pos_{t}_allowed" for t in TARGETS]]
    return pg.merge(dvp[cols], on=["opp_team_id", "game_id", "pos_group"], how="left")


def _merge_missing_teammates(pg: pd.DataFrame, upcoming: Upcoming | None) -> pd.DataFrame:
    """For each team-game, how much production is missing from regular rotation players who are not playing.

    Rotations are per team-season, so offseason departures don't count as missing. Historical rows: a rotation player who didn't log minutes. Upcoming rows: a rotation player in `upcoming.out`.
    """
    cols = ["min", *TARGETS]
    rows = []
    for (team_id, _season), tgp in pg.groupby(["team_id", "season"]):
        order = tgp[["game_id", "date"]].drop_duplicates("game_id").sort_values("date")["game_id"]
        wide = {c: tgp.pivot_table(index="game_id", columns="player_id", values=c, aggfunc="first").reindex(order) for c in cols}
        mins = wide["min"]
        # Upcoming games have NaN minutes for the players expected to play; mark those as "playing".
        present = tgp.pivot_table(index="game_id", columns="player_id", values="team_id", aggfunc="size").reindex(order).reindex(columns=mins.columns).notna()
        prev_mins = mins.shift(1)
        games_played = prev_mins.notna().rolling(ROTATION_WINDOW, min_periods=1).sum()
        avg_min = prev_mins.rolling(ROTATION_WINDOW, min_periods=1).mean()
        rotation = (games_played >= ROTATION_MIN_GAMES) & (avg_min >= ROTATION_MIN_MINUTES)
        missing = rotation & ~present
        out = pd.DataFrame({"team_id": team_id, "game_id": order.to_numpy(), "missing_teammates": missing.sum(axis=1).to_numpy()})
        for c in cols:
            recent = wide[c].shift(1).rolling(ROTATION_WINDOW, min_periods=1).mean()
            out[f"missing_teammate_{c}"] = recent.where(missing).sum(axis=1).to_numpy()
        rows.append(out)
    missing_df = pd.concat(rows, ignore_index=True)
    return pg.merge(missing_df, on=["team_id", "game_id"], how="left")


def training_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Completed games where the player played and has enough history to say something."""
    return features[features["min"].notna() & (features["min"] > 0) & features["pts_l10"].notna()]
