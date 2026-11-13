import time
from contextlib import asynccontextmanager
from datetime import date, timedelta
from threading import Lock

import numpy as np
import pandas as pd
from scipy.special import ndtr
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
Z80 = 1.2816  # p10..p90 is an 80% range: +/- 1.2816 standard deviations
DEFAULT_ODDS = -110  # when a line was stored without prices
TOP_N = 5
TOP_STAT = "pts"  # the nightly top-5 picks are player points props
DECIDED = ("win", "loss", "push")
PICK_COLUMNS = ["date", "game_id", "player_id", "player", "team", "opp", "stat", "line", "open_line", "over_odds", "under_odds",
                "predicted", "p10", "p90", "edge", "pick", "odds", "odds_assumed", "prob", "market_prob", "ev", "actual", "result",
                "units", "top5", "confidence_rank", "top5_gap", "gap_rank", "model_version", "kind"]


def _payout(odds: pd.Series) -> pd.Series:
    """Profit on a 1-unit stake at American odds: -110 pays 0.909, +120 pays 1.2."""
    return pd.Series(np.where(odds > 0, odds / 100, 100 / odds.abs()), index=odds.index)


def _implied(odds: pd.Series) -> pd.Series:
    return pd.Series(np.where(odds > 0, 100 / (odds + 100), -odds / (-odds + 100)), index=odds.index)


def _picks() -> pd.DataFrame:
    """Every prediction that has a DraftKings line, with the model's over/under pick, how confident it is and how it did.

    - Pick: "over" when the prediction is above the line, "under" when below (no pick when exactly on it).
    - Result: win when the player finishes on the picked side, push on the line, void if he didn't play (the book
      refunds it), pending until the game is final.
    - Confidence (prob): the model's chance the pick wins, treating the outcome as normal around the prediction with
      the spread of its calibrated 80% range (p10..p90), with a continuity correction because stats are whole numbers.
    - Units: 1 unit risked per pick at the DraftKings price for that side. EV: the model's expected units per pick,
      prob x payout - P(loss). Market prob: the book's no-vig chance for the same side.
    - Top 5, two ways, each night among player-points picks and from everything known before tip-off: the five with the
      highest confidence (top5 / confidence_rank), and the five where the prediction is furthest from the line
      (top5_gap / gap_rank; ties broken by confidence).
    """
    team, home, away = aliased(Team), aliased(Team), aliased(Team)
    q = (
        select(Prediction.game_date, Prediction.game_id, Prediction.player_id, Player.name, Player.team_id, team.abbr, home.abbr, away.abbr,
               Game.home_team_id, Game.status, Prediction.target, PlayerProp.line, PlayerProp.open_line, PlayerProp.over_odds,
               PlayerProp.under_odds, Prediction.predicted, Prediction.p10, Prediction.p90, Prediction.actual, Prediction.model_version)
        .join(PlayerProp, and_(PlayerProp.player_id == Prediction.player_id, PlayerProp.game_id == Prediction.game_id,
                               PlayerProp.target == Prediction.target))
        .join(Game, Game.id == Prediction.game_id)
        .join(home, home.id == Game.home_team_id, isouter=True)
        .join(away, away.id == Game.away_team_id, isouter=True)
        .join(Player, Player.id == Prediction.player_id, isouter=True)
        .join(team, team.id == Player.team_id, isouter=True)
    )
    with SessionLocal() as s:
        df = pd.DataFrame(s.execute(q).all(), columns=[
            "date", "game_id", "player_id", "player", "team_id", "team", "home", "away", "home_team_id", "status", "stat", "line",
            "open_line", "over_odds", "under_odds", "predicted", "p10", "p90", "actual", "model_version"])
    if df.empty:
        return pd.DataFrame(columns=PICK_COLUMNS)
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

    df["kind"] = np.where(df["model_version"].str.startswith(BACKTEST_PREFIX), "backtest", "live")
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    points = df["pick"].notna() & (df["stat"] == TOP_STAT)
    df["confidence_rank"] = df[points].groupby(["kind", "date"])["prob"].rank(ascending=False, method="first")
    df["top5"] = df["confidence_rank"] <= TOP_N
    gap_key = df["edge"].abs() + df["prob"].fillna(0) / 1000  # bigger gap first; confidence breaks ties
    df["gap_rank"] = gap_key[points].groupby([df["kind"], df["date"]]).rank(ascending=False, method="first")
    df["top5_gap"] = df["gap_rank"] <= TOP_N
    for c in ("prob", "ev", "market_prob", "units"):
        df[c] = df[c].round(4)
    return df[PICK_COLUMNS].sort_values(["date", "player", "stat"]).reset_index(drop=True)


def _record(g: pd.DataFrame) -> dict:
    """Win-loss record, units won at the DraftKings price, and expected units (the model's EV) for decided picks."""
    g = g[g["result"].isin(DECIDED)]
    w, l, p = (int((g["result"] == r).sum()) for r in DECIDED)
    n = w + l + p
    units, ev = float(g["units"].sum()), float(g["ev"].sum())
    return {"wins": w, "losses": l, "pushes": p, "picks": n, "win_pct": w / (w + l) if w + l else None,
            "units": round(units, 2), "roi": units / n if n else None, "expected_units": round(ev, 2),
            "avg_prob": float(g["prob"].mean()) if n else None, "odds_assumed": int(g["odds_assumed"].sum())}


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


def _json_rows(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(df.notna(), None).to_dict("records")


@app.get("/api/performance")
def performance(kind: str = Query("live", pattern="^(live|backtest)$")):
    """Over/under record, units and EV of the model's picks against DraftKings player props: all picks, each night's
    top 5 player-points picks (by biggest gap from the line and by confidence), and each night's top 5 performers."""
    allp = _picks()
    available = {k: int(((allp["kind"] == k) & allp["result"].isin(DECIDED)).sum()) for k in ("live", "backtest")}
    df = allp[allp["kind"] == kind]
    decided = df[df["result"].isin(DECIDED)]

    abs_edge = decided["edge"].abs()
    by_edge = [{"label": label, "min": lo, **_record(decided[(abs_edge >= lo) & (abs_edge < hi)])} for lo, hi, label in EDGE_BUCKETS]
    prob = decided["prob"]
    by_confidence = [{"label": label, **_record(decided[(prob >= lo) & (prob < hi)])}
                     for lo, hi, label in [(0, 0.55, "Under 55%"), (0.55, 0.6, "55–60%"), (0.6, 0.65, "60–65%"), (0.65, 0.7, "65–70%"),
                                           (0.7, 1.01, "70%+")]]
    top = _top_performers(kind, allp)
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


@app.get("/api/performance-export.csv")
def performance_export():
    """Every over/under pick against the DraftKings line (live and backtest), with confidence, price, EV and units."""
    return Response(_picks().to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="over-under-picks.csv"'})
