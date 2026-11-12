"""Score a slate of upcoming games, with optional what-if overrides (injuries, back-to-backs)."""

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.config import TARGETS as ACTIVE_TARGETS
from app.db import SessionLocal, upsert
from app.features.build import OUT_STATUSES, TARGETS, Frames, Upcoming, build_features
from app.ml.registry import load_active
from app.ml import intervals
from app.ml.tuning import bias_correction
from app.models import Injury, Prediction

ROSTER_LOOKBACK_DAYS = 21  # players who haven't played in 3 weeks are treated as unavailable
INJURY_LOOKBACK_DAYS = 2


@dataclass
class Overrides:
    out: set[int] = field(default_factory=set)  # force OUT
    available: set[int] = field(default_factory=set)  # force IN, even if the injury report says out
    b2b: dict[int, bool] = field(default_factory=dict)  # team_id -> force back-to-back on/off


def injury_report(on: date) -> dict[int, dict]:
    """Latest injury status per player from snapshots taken within the last couple of days."""
    with SessionLocal() as s:
        rows = s.scalars(
            select(Injury).where(Injury.date <= on, Injury.date >= on - timedelta(days=INJURY_LOOKBACK_DAYS)).order_by(Injury.date)
        ).all()
    return {r.player_id: {"status": r.status, "detail": r.detail} for r in rows}


def expected_roster(fr: Frames, games: pd.DataFrame, on: date) -> pd.DataFrame:
    """Players on each team (current roster) who have played recently."""
    team_ids = set(games["home_team_id"]) | set(games["away_team_id"])
    recent_cutoff = pd.Timestamp(on - timedelta(days=ROSTER_LOOKBACK_DAYS))
    ps = fr.player_stats.merge(fr.games[["id", "date"]], left_on="game_id", right_on="id")
    recent = set(ps.loc[(ps["date"] >= recent_cutoff) & (ps["min"] > 0), "player_id"])
    roster = fr.players[fr.players["team_id"].isin(team_ids) & fr.players["id"].isin(recent)][["id", "team_id"]]
    roster = roster.rename(columns={"id": "player_id"})
    by_team = pd.concat([
        games[["id", "home_team_id"]].rename(columns={"id": "game_id", "home_team_id": "team_id"}),
        games[["id", "away_team_id"]].rename(columns={"id": "game_id", "away_team_id": "team_id"}),
    ])
    return roster.merge(by_team, on="team_id")


def predict_slate(fr: Frames, on: date, overrides: Overrides | None = None, persist: bool = False) -> dict:
    overrides = overrides or Overrides()
    games = fr.games[(fr.games["date"] == pd.Timestamp(on)) & (fr.games["status"] == "scheduled")]
    if games.empty:
        return {"date": on.isoformat(), "games": [], "predictions": [], "model_version": None}

    roster = expected_roster(fr, games, on)
    injuries = injury_report(on)
    reported_out = {pid for pid, inj in injuries.items() if inj["status"] in OUT_STATUSES}
    out = (reported_out | overrides.out) - overrides.available
    out &= set(roster["player_id"])

    features = build_features(fr, Upcoming(games, roster, out=out, b2b_overrides=overrides.b2b))
    rows = features[features["game_id"].isin(games["id"]) & features["pts"].isna()].reset_index(drop=True)

    version = None
    predicted = []  # stats with a live model; the others are left blank rather than failing the whole slate
    for target in TARGETS:
        active = load_active(target) if target in ACTIVE_TARGETS else None
        if active is None:
            rows[[f"{target}_{k}" for k in ("raw", "pred", "p10", "p90")]] = np.nan
            continue
        predicted.append(target)
        version, bundle = active
        X = rows[bundle["features"]]
        raw = bundle["point"].predict(X)
        rows[f"{target}_raw"] = raw
        point = np.clip(raw + bias_correction(target), 0, None)
        lo, hi = intervals.apply(rows, target, point, bundle.get("interval", intervals.FALLBACK))
        rows[f"{target}_pred"] = point
        rows[f"{target}_p10"] = lo
        rows[f"{target}_p90"] = hi
    if not predicted:
        raise RuntimeError("No promoted models yet; run `python -m app.ml.train` first")
    rows["pra_pred"] = rows[[f"{t}_pred" for t in TARGETS]].sum(axis=1, min_count=len(TARGETS))

    if persist:
        _save(rows, version, on, injuries, predicted)
    return _response(fr, games, rows, out, injuries, overrides, version, on)


CONTEXT_COLS = ["b2b", "rest_days", "same_conference", "is_home", "min_l10", "missing_teammate_pts", "projected_pace", "opp_def_rtg_season"]


def _save(rows: pd.DataFrame, version: str, on: date, injuries: dict[int, dict], targets: list[str]) -> None:
    """Store pre-game predictions plus the context the daily evaluation slices accuracy by."""
    records = []
    for r in rows.itertuples():
        ctx = {c: _num(getattr(r, c)) for c in CONTEXT_COLS}
        ctx["injury_status"] = injuries.get(r.player_id, {}).get("status")
        for t in targets:
            records.append({
                "player_id": int(r.player_id), "game_id": r.game_id, "target": t, "game_date": on, "model_version": version,
                "predicted": float(getattr(r, f"{t}_pred")), "p10": float(getattr(r, f"{t}_p10")), "p90": float(getattr(r, f"{t}_p90")),
                "baseline": _num(getattr(r, f"{t}_l10")), "context": {**ctx, "raw": round(float(getattr(r, f"{t}_raw")), 2)},
            })
    upsert(Prediction.__table__, records)


def _num(v) -> float | None:
    return None if v is None or pd.isna(v) else round(float(v), 1)


def _response(fr, games, rows, out, injuries, overrides, version, on) -> dict:
    teams = fr.teams.set_index("id")
    names = fr.players.set_index("id")["name"]
    positions = fr.players.set_index("id")["position"]
    team_ctx = rows.drop_duplicates(["game_id", "team_id"]).set_index(["game_id", "team_id"])

    def team_info(game_id, team_id):
        t = teams.loc[team_id]
        ctx = team_ctx.loc[(game_id, team_id)] if (game_id, team_id) in team_ctx.index else None
        return {
            "id": int(team_id), "abbr": t["abbr"], "name": t["name"], "conference": t["conference"],
            "b2b": bool(ctx["b2b"]) if ctx is not None else None,
            "b2b_overridden": int(team_id) in overrides.b2b,
            "rest_days": _num(ctx["rest_days"]) if ctx is not None else None,
            "pace": _num(ctx["team_pace_season"]) if ctx is not None else None,
            "def_rtg": _num(ctx["opp_def_rtg_season"]) if ctx is not None else None,  # filled from the opponent's row below
        }

    games_out = []
    for g in games.itertuples():
        home, away = team_info(g.id, g.home_team_id), team_info(g.id, g.away_team_id)
        home["def_rtg"], away["def_rtg"] = away["def_rtg"], home["def_rtg"]  # opp_def_rtg on a team's row is its opponent's
        games_out.append({
            "game_id": g.id, "home": home, "away": away,
            "same_conference": home["conference"] == away["conference"],
            "out": [
                {"player_id": int(pid), "name": names.get(pid, str(pid)), "team_id": int(tid),
                 "status": injuries.get(pid, {}).get("status", "Out"), "manual": pid in overrides.out}
                for pid, tid in _out_players(fr, g, out)
            ],
        })

    preds = []
    for r in rows.itertuples():
        inj = injuries.get(r.player_id)
        preds.append({
            "player_id": int(r.player_id), "name": names.get(r.player_id), "position": positions.get(r.player_id),
            "team_id": int(r.team_id), "opp_team_id": int(r.opp_team_id), "game_id": r.game_id,
            "injury_status": inj["status"] if inj else None, "injury_detail": inj["detail"] if inj else None,
            "min_l10": _num(r.min_l10), "usg_season": _num(r.usg_season), "ts_season": _num(r.ts_season),
            "missing_teammate_pts": _num(r.missing_teammate_pts),
            **{f"{t}_{k}": _num(getattr(r, f"{t}_{k}")) for t in TARGETS for k in ("pred", "p10", "p90", "l5", "l10", "season", "vs_opp")},
            "pra_pred": _num(r.pra_pred),
        })
    preds.sort(key=lambda p: -(p["pts_pred"] or 0))
    return {"date": on.isoformat(), "model_version": version, "games": games_out, "predictions": preds}


def _out_players(fr: Frames, game, out: set[int]):
    teams = {game.home_team_id, game.away_team_id}
    current = fr.players[fr.players["id"].isin(out) & fr.players["team_id"].isin(teams)]
    return list(zip(current["id"], current["team_id"]))
