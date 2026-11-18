import gzip
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import date, timedelta
from threading import Lock

import numpy as np
import pandas as pd
from scipy.special import ndtr
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, select
from sqlalchemy.orm import aliased

from app.clock import today_et
from app.config import CORS_ORIGINS
from app.config import TARGETS as STAT_TARGETS
from app.db import SessionLocal
from app.features.build import Frames, load_frames
from app.init_db import init_db
from app.ml.backtest import VERSION_PREFIX as BACKTEST_PREFIX
from app.ml import top5
from app.ml.predict import Overrides, predict_slate
from app.models import (
    DailyEvaluation, Game, LockedGame, ModelConfig, ModelRun, PickSnapshot, Player, PlayerGameStat, PlayerProp, Prediction, Team,
    Top5Lock,
)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()  # create any tables added since the last deploy (e.g. player_props) before serving
    threading.Thread(target=_refresh_loop, daemon=True).start()  # no visitor should pay for a rebuild
    yield


app = FastAPI(title="NBA Player Predictor", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)  # the Performance payload is mostly repetitive pick rows

log = logging.getLogger("api")

FRAMES_TTL_S = 600
_frames: tuple[float, Frames] | None = None
_slate_cache: dict[tuple, dict] = {}
_lock = Lock()


def frames() -> Frames:
    """History is only updated by the daily job, so keep it in memory for a few minutes."""
    global _frames
    with _lock:
        if _frames is None or time.monotonic() - _frames[0] > FRAMES_TTL_S:
            _frames = (time.monotonic(), load_frames())
            _slate_cache.clear()
        return _frames[1]


class SlateRequest(BaseModel):
    on: date | None = Field(None, alias="date")
    out: list[int] = []
    available: list[int] = []
    b2b: dict[int, bool] = {}


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/dates")
def upcoming_dates(days: int = 7):
    start = today_et()
    with SessionLocal() as s:
        rows = s.execute(
            select(Game.date, func.count()).where(Game.date >= start, Game.date < start + timedelta(days=days), Game.status == "scheduled")
            .group_by(Game.date).order_by(Game.date)
        ).all()
    return [{"date": d.isoformat(), "games": n} for d, n in rows]


@app.get("/api/slate")
def get_slate(on: date | None = Query(None, alias="date")):
    on = on or today_et()
    fr = frames()
    key = (on, None)
    if key not in _slate_cache:
        _slate_cache[key] = _run(fr, on, Overrides())
    return _slate_cache[key]


@app.post("/api/slate")
def post_slate(req: SlateRequest):
    """What-if: re-run the slate with players forced out/in and back-to-backs toggled."""
    return _run(frames(), req.on or today_et(), Overrides(out=set(req.out), available=set(req.available), b2b=req.b2b))


def _run(fr: Frames, on: date, overrides: Overrides) -> dict:
    try:
        return predict_slate(fr, on, overrides)
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.get("/api/players/{player_id}")
def player(player_id: int, games: int = 25):
    with SessionLocal() as s:
        p = s.get(Player, player_id)
        if p is None:
            raise HTTPException(404, "Player not found")
        team = s.get(Team, p.team_id) if p.team_id else None
        log = s.execute(
            select(Game.date, Game.id, PlayerGameStat.opp_team_id, PlayerGameStat.min, PlayerGameStat.pts, PlayerGameStat.reb,
                   PlayerGameStat.ast, PlayerGameStat.fga, PlayerGameStat.fta)
            .join(Game, Game.id == PlayerGameStat.game_id)
            .where(PlayerGameStat.player_id == player_id).order_by(Game.date.desc()).limit(games)
        ).all()
        preds = s.scalars(select(Prediction).where(Prediction.player_id == player_id).order_by(Prediction.game_date.desc()).limit(games * 3)).all()
        abbr = dict(s.execute(select(Team.id, Team.abbr)).all())

    by_game: dict[str, dict] = {}
    for pr in preds:
        by_game.setdefault(pr.game_id, {})[pr.target] = {"predicted": round(pr.predicted, 1), "p10": round(pr.p10, 1), "p90": round(pr.p90, 1)}
    return {
        "id": p.id, "name": p.name, "position": p.position, "team": team.abbr if team else None,
        "games": [
            {"date": r.date.isoformat(), "game_id": r.id, "opp": abbr.get(r.opp_team_id), "min": r.min, "pts": r.pts, "reb": r.reb,
             "ast": r.ast, "predicted": by_game.get(r.id)}
            for r in reversed(log)
        ],
    }


@app.get("/api/model/metrics")
def metrics():
    with SessionLocal() as s:
        runs = s.execute(
            select(ModelRun.version, ModelRun.target, ModelRun.trained_at, ModelRun.mae, ModelRun.baseline_mae, ModelRun.rmse,
                   ModelRun.interval_coverage, ModelRun.promoted, ModelRun.n_train, ModelRun.n_holdout, ModelRun.feature_importance)
            .order_by(ModelRun.trained_at.desc()).limit(90)
        ).all()
        live = s.execute(
            select(Prediction.target, func.count(), func.avg(func.abs(Prediction.predicted - Prediction.actual)),
                   func.avg(func.abs(Prediction.baseline - Prediction.actual)))
            .where(Prediction.actual.is_not(None), Prediction.game_date >= today_et() - timedelta(days=30))
            .group_by(Prediction.target)
        ).all()
    latest = {}
    for r in runs:
        latest.setdefault(r.target, r)
    return {
        "latest": [
            {"target": r.target, "version": r.version, "trained_at": r.trained_at.isoformat(), "mae": r.mae, "baseline_mae": r.baseline_mae,
             "rmse": r.rmse, "interval_coverage": r.interval_coverage, "promoted": r.promoted, "n_train": r.n_train,
             "n_holdout": r.n_holdout, "top_features": dict(list(r.feature_importance.items())[:15])}
            for r in latest.values()
        ],
        "history": [
            {"target": r.target, "version": r.version, "trained_at": r.trained_at.isoformat(), "mae": r.mae, "baseline_mae": r.baseline_mae,
             "promoted": r.promoted}
            for r in reversed(runs)
        ],
        "live_30d": [{"target": t, "n": n, "mae": mae, "baseline_mae": base} for t, n, mae, base in live],
        "config": _configs(),
    }


def _configs() -> list[dict]:
    with SessionLocal() as s:
        return [
            {"target": c.target, "params": c.params, "bias_correction": c.bias_correction, "note": c.note,
             "locked_params": c.locked_params, "pinned_version": c.pinned_version,
             "updated_at": c.updated_at.isoformat() if c.updated_at else None}
            for c in s.scalars(select(ModelConfig))
        ]


# ---------------------------------------------------------------- accuracy tracking


@app.get("/api/evaluation")
def evaluation(start: date | None = None, end: date | None = None):
    """Daily scorecards (one row per date x stat) plus totals weighted by number of players."""
    q = select(DailyEvaluation).order_by(DailyEvaluation.game_date)
    if start:
        q = q.where(DailyEvaluation.game_date >= start)
    if end:
        q = q.where(DailyEvaluation.game_date <= end)
    with SessionLocal() as s:
        rows = s.scalars(q).all()
    days = [
        {"date": r.game_date.isoformat(), "target": r.target, "model_version": r.model_version, "n": r.n_players,
         "mae": r.mae, "rmse": r.rmse, "bias": r.bias, "baseline_mae": r.baseline_mae, "beat_baseline_pct": r.beat_baseline_pct,
         "close_threshold": r.close_threshold, "within_close_pct": r.within_close_pct, "within_wide_pct": r.within_wide_pct,
         "interval_coverage": r.interval_coverage, "segments": r.segments,
         "kind": "backtest" if r.model_version.startswith(BACKTEST_PREFIX) else "live"}
        for r in rows
    ]
    # Live and backtest results are summarized separately so season totals never mix the two.
    totals: dict[str, dict] = {"live": {}, "backtest": {}}
    if days:
        df = pd.DataFrame(days)
        for (kind, target), g in df.groupby(["kind", "target"]):
            w = g["n"]
            avg = lambda col: float((g[col] * w).sum() / w.sum()) if g[col].notna().all() else None  # noqa: E731
            totals[kind][target] = {"days": len(g), "n": int(w.sum()), **{c: avg(c) for c in
                                    ("mae", "bias", "baseline_mae", "beat_baseline_pct", "within_close_pct", "within_wide_pct", "interval_coverage")},
                                    "close_threshold": float(g["close_threshold"].iat[0])}
    return {"days": days, "totals": totals}


def _graded(on: date | None = None):
    q = (
        select(Prediction.game_date, Prediction.player_id, Player.name, Team.abbr, Prediction.target, Prediction.predicted,
               Prediction.p10, Prediction.p90, Prediction.baseline, Prediction.actual, Prediction.model_version)
        .join(Player, Player.id == Prediction.player_id, isouter=True)
        .join(Team, Team.id == Player.team_id, isouter=True)
        .where(Prediction.actual.is_not(None))
        .order_by(Prediction.game_date, Prediction.player_id)
    )
    if on:
        q = q.where(Prediction.game_date == on)
    with SessionLocal() as s:
        return s.execute(q).all()


@app.get("/api/evaluation/{on}")
def evaluation_day(on: date):
    """Every player's prediction vs. actual for one date."""
    players: dict[int, dict] = {}
    for r in _graded(on):
        p = players.setdefault(r.player_id, {"player_id": r.player_id, "name": r.name, "team": r.abbr})
        p[r.target] = {"predicted": round(r.predicted, 1), "actual": r.actual, "error": round(r.predicted - r.actual, 1),
                       "p10": round(r.p10, 1), "p90": round(r.p90, 1), "baseline": None if r.baseline is None else round(r.baseline, 1)}
    return sorted(players.values(), key=lambda p: abs(p.get("pts", {}).get("error", 0)))


@app.get("/api/evaluation-export.csv")
def evaluation_export():
    """Every graded prediction, for end-of-season review in a spreadsheet."""
    df = pd.DataFrame(_graded(), columns=["date", "player_id", "player", "team", "stat", "predicted", "p10", "p90",
                                          "l10_baseline", "actual", "model_version"])
    df["error"] = (df["predicted"] - df["actual"]).round(2)
    df["abs_error"] = df["error"].abs()
    df["in_range"] = (df["actual"] >= df["p10"]) & (df["actual"] <= df["p90"])
    return Response(df.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="prediction-accuracy.csv"'})


# ---------------------------------------------------------------- performance vs. DraftKings lines

EDGE_BUCKETS = [(0, 1, "Under 1"), (1, 2, "1 to 2"), (2, 3, "2 to 3"), (3, float("inf"), "3 or more")]
Z80 = 1.2816  # p10..p90 is an 80% range: +/- 1.2816 standard deviations
DEFAULT_ODDS = -110  # when a line was stored without prices
TOP_N = 5
TOP_STAT = "pts"  # the nightly top-5 picks are player points props
DECIDED = ("win", "loss", "push")
PICK_COLUMNS = ["date", "game_id", "player_id", "player", "team", "opp", "stat", "line", "open_line", "over_odds", "under_odds",
                "predicted", "p10", "p90", "edge", "pick", "odds", "odds_assumed", "prob", "market_prob", "ev", "actual", "result",
                "units", "open_pick", "clv", "top5", "confidence_rank", "top5_gap", "gap_rank", "model_version", "kind"]


def _payout(odds: pd.Series) -> pd.Series:
    """Profit on a 1-unit stake at American odds: -110 pays 0.909, +120 pays 1.2."""
    return pd.Series(np.where(odds > 0, odds / 100, 100 / odds.abs()), index=odds.index)


def _implied(odds: pd.Series) -> pd.Series:
    return pd.Series(np.where(odds > 0, 100 / (odds + 100), -odds / (-odds + 100)), index=odds.index)


RAW_COLUMNS = ["date", "game_id", "player_id", "player", "team_id", "team", "home", "away", "home_team_id", "status", "stat", "line",
               "open_line", "over_odds", "under_odds", "predicted", "p10", "p90", "actual", "model_version"]


def _load(source, phase: str | None = None, on: date | None = None) -> pd.DataFrame:
    """Prediction + line rows. `source` is PlayerProp (the current/closing lines with the stored predictions) or
    PickSnapshot (both frozen before tip-off, for one `phase`)."""
    team, home, away = aliased(Team), aliased(Team), aliased(Team)
    keys = and_(source.player_id == Prediction.player_id, source.game_id == Prediction.game_id, source.target == Prediction.target)
    vals = Prediction if source is PlayerProp else source  # snapshots carry their own frozen prediction
    q = (
        select(Prediction.game_date, Prediction.game_id, Prediction.player_id, Player.name, Player.team_id, team.abbr, home.abbr, away.abbr,
               Game.home_team_id, Game.status, Prediction.target, source.line, source.open_line, source.over_odds,
               source.under_odds, vals.predicted, vals.p10, vals.p90, Prediction.actual, vals.model_version)
        .join(source, keys)
        .join(Game, Game.id == Prediction.game_id)
        .join(home, home.id == Game.home_team_id, isouter=True)
        .join(away, away.id == Game.away_team_id, isouter=True)
        .join(Player, Player.id == Prediction.player_id, isouter=True)
        .join(team, team.id == Player.team_id, isouter=True)
    )
    if phase is not None:
        q = q.where(source.phase == phase)
    if on is not None:
        q = q.where(Prediction.game_date == on)
    with SessionLocal() as s:
        return pd.DataFrame(s.execute(q).all(), columns=RAW_COLUMNS)


def _grade(df: pd.DataFrame) -> pd.DataFrame:
    """Pick, confidence, price, EV, result, units and CLV for each row (see _picks)."""
    df = df.copy()
    df["opp"] = np.where(df["team_id"] == df["home_team_id"], df["away"], df["home"])
    df["predicted"] = df["predicted"].round(1)  # pick off the shown prediction so the table can be checked by eye
    df["edge"] = (df["predicted"] - df["line"]).round(1)
    over = df["edge"] > 0
    df["pick"] = np.select([over, df["edge"] < 0], ["over", "under"], None)

    sigma = ((df["p90"] - df["p10"]) / (2 * Z80)).clip(lower=0.5)
    p_over = 1 - ndtr((np.floor(df["line"]) + 0.5 - df["predicted"]) / sigma)  # finishes at floor(line)+1 or more
    p_under = ndtr((np.ceil(df["line"]) - 0.5 - df["predicted"]) / sigma)  # finishes at ceil(line)-1 or less
    df["prob"] = np.where(df["pick"].isna(), np.nan, np.where(over, p_over, p_under))
    p_loss = np.where(over, p_under, p_over)
    pick_odds = pd.Series(np.where(over, df["over_odds"], df["under_odds"]), index=df.index, dtype="float")
    df["odds_assumed"] = pick_odds.isna() & df["pick"].notna()
    df["odds"] = pick_odds.fillna(DEFAULT_ODDS)
    pay = _payout(df["odds"])
    df["ev"] = np.where(df["pick"].isna(), np.nan, df["prob"] * pay - p_loss)
    other = pd.Series(np.where(over, df["under_odds"], df["over_odds"]), index=df.index, dtype="float")
    df["market_prob"] = _implied(pick_odds) / (_implied(pick_odds) + _implied(other))  # NaN without both prices

    final = df["status"] == "final"
    over_hit, under_hit = df["actual"] > df["line"], df["actual"] < df["line"]
    df["result"] = np.select(
        [~final, df["actual"].isna(), df["pick"].isna(), df["actual"] == df["line"],
         ((df["pick"] == "over") & over_hit) | ((df["pick"] == "under") & under_hit)],
        ["pending", "void", "no pick", "push", "win"], "loss")
    df["units"] = np.select([df["result"] == "win", df["result"] == "loss"], [pay, -1.0], 0.0)
    df.loc[~df["result"].isin(DECIDED), "units"] = np.nan

    # Closing line value: the side the model takes against the opening line, and how far the line then moved
    # toward it by tip-off (positive = the market moved the model's way, the best early sign of a real edge).
    open_edge = df["predicted"] - df["open_line"]
    df["open_pick"] = np.select([open_edge > 0, open_edge < 0], ["over", "under"], None)
    move = df["line"] - df["open_line"]
    df["clv"] = np.where(df["open_pick"] == "over", move, np.where(df["open_pick"] == "under", -move, np.nan))

    df["kind"] = np.where(df["model_version"].str.startswith(BACKTEST_PREFIX), "backtest", "live")
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    for c in ("prob", "ev", "market_prob", "units", "clv"):
        df[c] = df[c].round(4)
    return df


def _rank(df: pd.DataFrame, locks: top5.Locks | None = None) -> pd.DataFrame:
    """Each night's top 5 player-points picks, by confidence and by biggest gap from the line (confidence breaks ties).
    With `locks` (the final picks), nights whose early games have tipped keep the picks settled then (app.ml.top5)."""
    df = df.copy()
    points = df["pick"].notna() & (df["stat"] == TOP_STAT)
    df["confidence_rank"] = df[points].groupby(["kind", "date"])["prob"].rank(ascending=False, method="first")
    df["top5"] = df["confidence_rank"] <= TOP_N
    gap_key = df["edge"].abs() + df["prob"].fillna(0) / 1000
    df["gap_rank"] = gap_key[points].groupby([df["kind"], df["date"]]).rank(ascending=False, method="first")
    df["top5_gap"] = df["gap_rank"] <= TOP_N
    return top5.apply(df, locks, TOP_N) if locks and locks.games else df


# The grading below reads every prediction, line and snapshot and takes a couple of seconds, but the inputs only
# change when the daily / evening jobs write. So results are cached against a cheap fingerprint of those tables and
# recomputed only when it changes (or after DATA_TTL_S, as a backstop for writes the fingerprint can't see).
DATA_TTL_S = 900
_data_cache: dict[tuple, tuple[float, tuple, object]] = {}
_data_locks: dict[tuple, Lock] = {}
_data_locks_guard = Lock()


def _fingerprint() -> tuple:
    """One round trip whose result changes whenever a job adds or re-predicts predictions, fills actuals, finishes
    games, or stores lines / pick snapshots. Counts and value sums rather than timestamps, because the jobs upsert and
    an upsert doesn't touch updated_at."""
    parts = [
        func.count(), func.sum(Prediction.predicted), func.sum(Prediction.actual),
    ], [
        func.count(), func.sum(PlayerProp.line), func.sum(PlayerProp.over_odds), func.sum(PlayerProp.under_odds),
    ], [
        func.count(), func.sum(PickSnapshot.line), func.sum(PickSnapshot.predicted),
    ]
    sources = (Prediction, PlayerProp, PickSnapshot)
    cols = [select(c).select_from(src).scalar_subquery() for src, cs in zip(sources, parts) for c in cs]
    cols.append(select(func.count()).select_from(Game).where(Game.status == "final").scalar_subquery())
    cols += [select(func.count()).select_from(t).scalar_subquery() for t in (LockedGame, Top5Lock)]  # top 5 settled at tip-off
    with SessionLocal() as s:
        return tuple(s.execute(select(*cols)).one())


def _cached(key: tuple, compute, fp: tuple | None = None):
    """compute(), reused while the data fingerprint is unchanged. One computation per key at a time."""
    with _data_locks_guard:
        lock = _data_locks.setdefault(key, Lock())
    fp = _fingerprint() if fp is None else fp
    with lock:
        hit = _data_cache.get(key)
        if hit and hit[1] == fp and time.monotonic() - hit[0] < DATA_TTL_S:
            return hit[2]
        value = compute()
        _data_cache[key] = (time.monotonic(), fp, value)
        return value


def _picks() -> pd.DataFrame:
    return _cached(("picks",), _compute_picks)


def _compute_picks() -> pd.DataFrame:
    """Every prediction that has a DraftKings line, graded against the stored (latest / closing) line.

    - Pick: "over" when the prediction is above the line, "under" when below (no pick when exactly on it).
    - Result: win when the player finishes on the picked side, push on the line, void if he didn't play (the book
      refunds it), pending until the game is final.
    - Confidence (prob): the model's chance the pick wins, treating the outcome as normal around the prediction with
      the spread of its calibrated 80% range (p10..p90), with a continuity correction because stats are whole numbers.
    - Units: 1 unit risked per pick at the DraftKings price for that side. EV: the model's expected units per pick,
      prob x payout - P(loss). Market prob: the book's no-vig chance for the same side.
    """
    df = _grade(_load(PlayerProp))
    for col in ("top5", "top5_gap"):
        df[col] = False
    df["confidence_rank"] = df["gap_rank"] = np.nan
    return df[PICK_COLUMNS].sort_values(["date", "player", "stat"]).reset_index(drop=True) if not df.empty else pd.DataFrame(columns=PICK_COLUMNS)


PHASES = ("morning", "final")


def _view(allp: pd.DataFrame, kind: str, phase: str) -> pd.DataFrame:
    """The picks a Performance page grades, with each night's top 5 ranked.

    Live nights: only picks frozen before tip-off (pick_snapshots) for the page's phase: "morning" (the morning run,
    before the injury updates) or "final" (the last injury update before each game's tip-off). Nothing about them changes
    once games start, and lines first seen after the games never count. On the final page a night's top 5 is settled
    game by game as they tip off (app.ml.top5), so a later game can't push out a pick that was in it at tip-off. Backtest nights are historical and graded on the
    stored lines (the same on both pages).
    """
    if kind == "backtest":
        df = allp[allp["kind"] == "backtest"]
    else:
        snap = _load(PickSnapshot, phase)
        df = _grade(snap) if not snap.empty else allp.iloc[0:0]
        df = df[df["kind"] == "live"]
    if df.empty:
        return pd.DataFrame(columns=PICK_COLUMNS)
    locks = top5.load() if kind == "live" and phase == "final" else None
    return _rank(df, locks)[PICK_COLUMNS].sort_values(["date", "player", "stat"]).reset_index(drop=True)


def _record(g: pd.DataFrame) -> dict:
    """Win-loss record, units won at the DraftKings price, and expected units (the model's EV) for decided picks."""
    g = g[g["result"].isin(DECIDED)]
    w, l, p = (int((g["result"] == r).sum()) for r in DECIDED)
    n = w + l + p
    units, ev = float(g["units"].sum()), float(g["ev"].sum())
    return {"wins": w, "losses": l, "pushes": p, "picks": n, "win_pct": w / (w + l) if w + l else None,
            "units": round(units, 2), "roi": units / n if n else None, "expected_units": round(ev, 2),
            "avg_prob": float(g["prob"].mean()) if n else None, "odds_assumed": int(g["odds_assumed"].sum()), **_clv(g)}


def _clv(g: pd.DataFrame) -> dict:
    """How the line moved after opening, measured from the side the model took against the opening line."""
    c = g["clv"].dropna()
    return {"clv_n": len(c), "clv_avg": float(c.mean()) if len(c) else None,
            "clv_toward": float((c > 0).mean()) if len(c) else None, "clv_away": float((c < 0).mean()) if len(c) else None}


def _running(g: pd.DataFrame, by_stat: bool) -> list[dict]:
    out, cum = [], {}
    for d, day_df in g[g["result"].isin(DECIDED)].groupby("date"):
        groups = [("all", day_df), *day_df.groupby("stat")] if by_stat else [("all", day_df)]
        for stat, gs in groups:
            day = _record(gs)
            c = cum.setdefault(stat, {"wins": 0, "losses": 0, "pushes": 0, "units": 0.0, "expected_units": 0.0})
            for k in c:
                c[k] += day[k]
            out.append({"date": d, "stat": stat, "day_wins": day["wins"], "day_losses": day["losses"], "day_pushes": day["pushes"],
                        "day_units": day["units"], "wins": c["wins"], "losses": c["losses"], "pushes": c["pushes"],
                        "units": round(c["units"], 2), "expected_units": round(c["expected_units"], 2),
                        "win_pct": c["wins"] / (c["wins"] + c["losses"]) if c["wins"] + c["losses"] else None})
    return out


def _top_performers(kind: str, picks: pd.DataFrame) -> list[dict]:
    """Each night's top 5 by actual points / rebounds / assists, with what the model predicted and where it ranked them."""
    with SessionLocal() as s:
        preds = pd.DataFrame(s.execute(select(Prediction.game_date, Prediction.player_id, Prediction.game_id, Prediction.target,
                                              Prediction.predicted, Prediction.model_version)).all(),
                             columns=["date", "player_id", "game_id", "stat", "predicted", "model_version"])
        if preds.empty:
            return []
        preds = preds[preds["model_version"].str.startswith(BACKTEST_PREFIX) == (kind == "backtest")]
        dates = sorted(set(preds["date"]))
        if not dates:
            return []
        box = pd.DataFrame(s.execute(
            select(Game.date, PlayerGameStat.player_id, PlayerGameStat.game_id, Player.name, Team.abbr, PlayerGameStat.pts, PlayerGameStat.reb,
                   PlayerGameStat.ast)
            .join(Game, Game.id == PlayerGameStat.game_id).join(Player, Player.id == PlayerGameStat.player_id, isouter=True)
            .join(Team, Team.id == PlayerGameStat.team_id, isouter=True).where(Game.date.in_(dates))
        ).all(), columns=["date", "player_id", "game_id", "player", "team", "pts", "reb", "ast"])
    preds["model_rank"] = preds.groupby(["date", "stat"])["predicted"].rank(ascending=False, method="min").astype(int)
    keys = ["player_id", "game_id", "stat"]
    top = (box.melt(id_vars=["date", "player_id", "game_id", "player", "team"], value_vars=["pts", "reb", "ast"],
                    var_name="stat", value_name="actual")
           .sort_values(["date", "stat", "actual", "player"], ascending=[True, True, False, True]))
    top["rank"] = top.groupby(["date", "stat"]).cumcount() + 1
    top = top[top["rank"] <= 5]
    top = top.merge(preds[keys + ["predicted", "model_rank"]].drop_duplicates(keys), on=keys, how="left")
    top = top.merge(picks[keys + ["line", "pick", "result"]].drop_duplicates(keys), on=keys, how="left")
    return [{
        "date": str(r.date), "stat": r.stat, "rank": int(r.rank), "player_id": int(r.player_id), "player": r.player,
        "team": r.team, "actual": int(r.actual),
        "predicted": None if pd.isna(r.predicted) else round(float(r.predicted), 1),
        "model_rank": None if pd.isna(r.model_rank) else int(r.model_rank),
        "line": None if pd.isna(r.line) else float(r.line),
        "pick": None if pd.isna(r.pick) else r.pick, "result": None if pd.isna(r.result) else r.result,
    } for r in top.itertuples()]


def _json_rows(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(df.notna(), None).to_dict("records")


@app.get("/api/performance")
def performance(request: Request, kind: str = Query("live", pattern="^(live|backtest)$"),
                phase: str = Query("morning", pattern="^(morning|final)$")):
    """Over/under record, units and EV of the model's picks against DraftKings player props: all picks, each night's
    top 5 player-points picks (by biggest gap from the line and by confidence), and each night's top 5 performers."""
    key = ("performance", kind, phase)
    hit = _data_cache.get(key)  # kept fresh by _refresh_loop, so a visitor never waits on a rebuild
    raw, packed = hit[2] if hit else _cached(key, lambda: _encode(_performance(kind, phase)))
    if "gzip" in request.headers.get("accept-encoding", ""):
        return Response(packed, media_type="application/json", headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"})
    return Response(raw, media_type="application/json", headers={"Vary": "Accept-Encoding"})


def _encode(payload: dict) -> tuple[bytes, bytes]:
    """The response body, encoded and gzipped once per data change instead of on every request (it's several MB)."""
    raw = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    return raw, gzip.compress(raw, compresslevel=6)


REFRESH_EVERY_S = 60


def _refresh_performance():
    """Rebuild any Performance view whose data changed (or that is older than DATA_TTL_S). Visitors keep getting the
    previous copy until the new one is ready."""
    fp = _fingerprint()
    for phase in PHASES:
        for kind in ("live", "backtest"):
            _cached(("performance", kind, phase), lambda k=kind, p=phase: _encode(_performance(k, p)), fp)


def _refresh_loop():
    """Fill the Performance cache at startup, then check for new data from the daily / 6pm jobs every minute."""
    while True:
        try:
            _refresh_performance()
        except Exception:  # a failed refresh only means a slower or older page, never a wrong one
            log.exception("refreshing the performance cache failed")
        time.sleep(REFRESH_EVERY_S)


def _performance(kind: str, phase: str) -> dict:
    allp = _picks()
    views = {k: _view(allp, k, phase) for k in ("live", "backtest")}
    available = {k: int(v["result"].isin(DECIDED).sum()) for k, v in views.items()}
    df = views[kind]
    decided = df[df["result"].isin(DECIDED)]

    abs_edge = decided["edge"].abs()
    by_edge = [{"label": label, "min": lo, **_record(decided[(abs_edge >= lo) & (abs_edge < hi)])} for lo, hi, label in EDGE_BUCKETS]
    prob = decided["prob"]
    by_confidence = [{"label": label, **_record(decided[(prob >= lo) & (prob < hi)])}
                     for lo, hi, label in [(0, 0.55, "Under 55%"), (0.55, 0.6, "55–60%"), (0.6, 0.65, "60–65%"), (0.65, 0.7, "65–70%"),
                                           (0.7, 1.01, "70%+")]]
    top = _top_performers(kind, df)
    top_df = pd.DataFrame(top)
    top_summary = None
    if not top_df.empty:
        nights = top_df.groupby(["date", "stat"])["model_rank"].apply(lambda r: int((r <= 5).sum()))
        top_summary = {
            "nights": int(top_df["date"].nunique()),
            "in_model_top5": {stat: float(v.mean()) for stat, v in nights.groupby(level="stat")},
            "record": {stat: _record_simple(g) for stat, g in [("all", top_df), *top_df.groupby("stat")]},
        }
    return _json_safe({
        "kind": kind,
        "available": available,
        "totals": {stat: _record(g) for stat, g in [("all", decided), *decided.groupby("stat")]},
        "no_pick": int((df["result"] == "no pick").sum()),
        "void": int((df["result"] == "void").sum()),
        "running": _running(decided, by_stat=True),
        "by_edge": by_edge,
        "by_confidence": by_confidence,
        "phase": phase,
        "top5": {name: _strategy(df, flag, rank) for name, flag, rank in
                 (("gap", "top5_gap", "gap_rank"), ("confidence", "top5", "confidence_rank"))},
        "top_performers": top,
        "top_summary": top_summary,
        "picks": _json_rows(df[df["result"] != "pending"]),
    })


def _json_safe(v):
    """NaN/inf aren't valid JSON (the response would fail to serialize): turn them into null, everywhere, and numpy
    scalars into plain Python numbers."""
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def _strategy(df: pd.DataFrame, flag: str, rank: str) -> dict:
    """One way of picking each night's top 5: its record, night-by-night tally and the picks themselves."""
    picked = df[df[flag].fillna(False).astype(bool)].copy()
    picked["rank"] = picked[rank]
    return {"totals": _record(picked), "running": _running(picked, by_stat=False),
            "picks": _json_rows(picked.sort_values(["date", "rank"], ascending=[False, True]))}


def _record_simple(g: pd.DataFrame) -> dict:
    w, l, p = (int((g["result"] == r).sum()) for r in DECIDED)
    return {"wins": w, "losses": l, "pushes": p, "picks": w + l + p, "win_pct": w / (w + l) if w + l else None}


@app.get("/api/accuracy-vs-line")
def evaluation_vs_line():
    """Accuracy on the players DraftKings posts a line for: the model's average miss next to the line's own average
    miss, per stat, overall and by date (live and backtest separately)."""
    df = _picks()
    df = df[~df["result"].isin(["pending", "void"]) & df["actual"].notna()].copy()
    df["model_err"] = (df["predicted"] - df["actual"]).abs()
    df["line_err"] = (df["line"] - df["actual"]).abs()

    def summary(g: pd.DataFrame) -> dict:
        n = len(g)
        return {"n": n, "model_mae": float(g["model_err"].mean()), "line_mae": float(g["line_err"].mean()),
                "model_bias": float((g["predicted"] - g["actual"]).mean()), "line_bias": float((g["line"] - g["actual"]).mean()),
                "model_closer": float((g["model_err"] < g["line_err"]).mean()), "line_closer": float((g["line_err"] < g["model_err"]).mean())}

    totals = {k: {stat: summary(g) for stat, g in gk.groupby("stat")} for k, gk in df.groupby("kind")}
    days = [{"date": d, "stat": stat, "kind": k, **summary(g)} for (d, stat, k), g in df.groupby(["date", "stat", "kind"])]
    return _json_safe({"totals": totals, "days": days})


@app.get("/api/picks")
def top_picks(on: date | None = Query(None, alias="date"), n: int = Query(TOP_N, ge=1, le=20)):
    """Each stat's top picks for one date: the props where the stored pre-game prediction is furthest from the
    DraftKings line (confidence breaks ties), the same way the Performance page ranks its top 5."""
    on = on or today_et()
    df = _picks()
    df = df[(df["date"] == on.isoformat()) & df["pick"].notna() & (df["result"] != "void")]
    out = {}
    for stat in STAT_TARGETS:
        g = df[df["stat"] == stat]
        g = g.assign(gap=g["edge"].abs()).sort_values(["gap", "prob"], ascending=[False, False]).head(n)
        out[stat] = _json_rows(g.drop(columns="gap"))
    return _json_safe({"date": on.isoformat(), "picks": out})


@app.get("/api/performance-export.csv")
def performance_export():
    """Every over/under pick against the DraftKings line (live and backtest), with confidence, price, EV and units."""
    return Response(_picks().to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="over-under-picks.csv"'})
