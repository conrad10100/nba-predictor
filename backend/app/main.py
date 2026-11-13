import time
from contextlib import asynccontextmanager
from datetime import date, timedelta
from threading import Lock

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, select
from sqlalchemy.orm import aliased

from app.clock import today_et
from app.config import CORS_ORIGINS
from app.db import SessionLocal
from app.features.build import Frames, load_frames
from app.init_db import init_db
from app.ml.backtest import VERSION_PREFIX as BACKTEST_PREFIX
from app.ml.predict import Overrides, predict_slate
from app.models import DailyEvaluation, Game, ModelConfig, ModelRun, Player, PlayerGameStat, PlayerProp, Prediction, Team

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()  # create any tables added since the last deploy (e.g. player_props) before serving
    yield


app = FastAPI(title="NBA Player Predictor", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

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
PICK_COLUMNS = ["date", "game_id", "player_id", "player", "team", "opp", "stat", "line", "open_line", "predicted", "edge", "pick",
                "actual", "result", "model_version", "kind"]


def _picks() -> pd.DataFrame:
    """Every graded prediction that has a DraftKings line, with the model's over/under pick and how it turned out.

    The pick is "over" when the prediction is above the line and "under" when below (no pick when exactly on it).
    A pick wins when the actual stat lands on that side of the line; landing exactly on the line is a push.
    """
    team, opp = aliased(Team), aliased(Team)
    q = (
        select(Prediction.game_date, Prediction.game_id, Prediction.player_id, Player.name, team.abbr, opp.abbr, Prediction.target,
               PlayerProp.line, PlayerProp.open_line, Prediction.predicted, Prediction.actual, Prediction.model_version)
        .join(PlayerProp, and_(PlayerProp.player_id == Prediction.player_id, PlayerProp.game_id == Prediction.game_id,
                               PlayerProp.target == Prediction.target))
        .join(Player, Player.id == Prediction.player_id, isouter=True)
        .join(PlayerGameStat, and_(PlayerGameStat.player_id == Prediction.player_id, PlayerGameStat.game_id == Prediction.game_id), isouter=True)
        .join(team, team.id == PlayerGameStat.team_id, isouter=True)
        .join(opp, opp.id == PlayerGameStat.opp_team_id, isouter=True)
        .where(Prediction.actual.is_not(None))
    )
    with SessionLocal() as s:
        df = pd.DataFrame(s.execute(q).all(), columns=["date", "game_id", "player_id", "player", "team", "opp", "stat", "line", "open_line",
                                                        "predicted", "actual", "model_version"])
    if df.empty:
        return pd.DataFrame(columns=PICK_COLUMNS)
    df["predicted"] = df["predicted"].round(1)  # pick off the shown prediction so the table can be checked by eye
    df["edge"] = (df["predicted"] - df["line"]).round(1)
    df["pick"] = np.select([df["edge"] > 0, df["edge"] < 0], ["over", "under"], None)
    over_hit, under_hit = df["actual"] > df["line"], df["actual"] < df["line"]
    df["result"] = np.select(
        [df["pick"].isna(), df["actual"] == df["line"], (df["pick"] == "over") & over_hit, (df["pick"] == "under") & under_hit],
        ["no pick", "push", "win", "win"], "loss")
    df["kind"] = np.where(df["model_version"].str.startswith(BACKTEST_PREFIX), "backtest", "live")
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    return df[PICK_COLUMNS].sort_values(["date", "player", "stat"]).reset_index(drop=True)


def _record(g: pd.DataFrame) -> dict:
    w, l, p = (int((g["result"] == r).sum()) for r in ("win", "loss", "push"))
    return {"wins": w, "losses": l, "pushes": p, "picks": w + l + p, "win_pct": w / (w + l) if w + l else None}


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
    picks = picks.set_index(["player_id", "game_id", "stat"])
    preds["model_rank"] = preds.groupby(["date", "stat"])["predicted"].rank(ascending=False, method="min").astype(int)
    preds = preds.set_index(["player_id", "game_id", "stat"])
    out = []
    for (d, stat), g in box.melt(id_vars=["date", "player_id", "game_id", "player", "team"], value_vars=["pts", "reb", "ast"],
                                 var_name="stat", value_name="actual").groupby(["date", "stat"]):
        top = g.sort_values(["actual", "player"], ascending=[False, True]).head(5)
        for rank, r in enumerate(top.itertuples(), 1):
            key = (r.player_id, r.game_id, stat)
            pr = preds.loc[key] if key in preds.index else None
            pk = picks.loc[key] if key in picks.index else None
            out.append({
                "date": str(d), "stat": stat, "rank": rank, "player_id": int(r.player_id), "player": r.player, "team": r.team,
                "actual": int(r.actual), "predicted": None if pr is None else round(float(pr["predicted"]), 1),
                "model_rank": None if pr is None else int(pr["model_rank"]),
                "line": None if pk is None else float(pk["line"]), "pick": None if pk is None else pk["pick"],
                "result": None if pk is None else pk["result"],
            })
    return out


@app.get("/api/performance")
def performance(kind: str = Query("live", pattern="^(live|backtest)$")):
    """Over/under record of the model's predictions against DraftKings player prop lines, plus each night's top 5 performers."""
    allp = _picks()
    available = {k: int((allp["kind"] == k).sum()) for k in ("live", "backtest")}
    df = allp[allp["kind"] == kind]
    graded = df[df["result"] != "no pick"]

    running, cum = [], {}
    for d, g in graded.groupby("date"):
        for stat, gs in [("all", g), *g.groupby("stat")]:
            day = _record(gs)
            c = cum.setdefault(stat, {"wins": 0, "losses": 0, "pushes": 0})
            for k in c:
                c[k] += day[k]
            running.append({"date": d, "stat": stat, "day_wins": day["wins"], "day_losses": day["losses"], "day_pushes": day["pushes"],
                            "wins": c["wins"], "losses": c["losses"], "pushes": c["pushes"],
                            "win_pct": c["wins"] / (c["wins"] + c["losses"]) if c["wins"] + c["losses"] else None})

    abs_edge = graded["edge"].abs()
    by_edge = [{"label": label, "min": lo, **_record(graded[(abs_edge >= lo) & (abs_edge < hi)])} for lo, hi, label in EDGE_BUCKETS]
    top = _top_performers(kind, allp)
    top_df = pd.DataFrame(top)
    top_summary = None
    if not top_df.empty:
        nights = top_df.groupby(["date", "stat"])["model_rank"].apply(lambda r: int((r <= 5).sum()))
        top_summary = {
            "nights": int(top_df["date"].nunique()),
            "in_model_top5": {stat: float(v.mean()) for stat, v in nights.groupby(level="stat")},  # avg of the 5 the model also had top 5
            "record": {stat: _record(g[g["result"].isin(["win", "loss", "push"])]) for stat, g in [("all", top_df), *top_df.groupby("stat")]},
        }
    return {
        "kind": kind,
        "available": available,
        "totals": {stat: _record(g) for stat, g in [("all", graded), *graded.groupby("stat")]},
        "no_pick": int((df["result"] == "no pick").sum()),
        "running": running,
        "by_edge": by_edge,
        "top_performers": top,
        "top_summary": top_summary,
        "picks": df.replace({np.nan: None}).to_dict("records"),
    }


@app.get("/api/performance-export.csv")
def performance_export():
    """Every graded over/under pick against the DraftKings line, live and backtest."""
    return Response(_picks().to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="over-under-picks.csv"'})
