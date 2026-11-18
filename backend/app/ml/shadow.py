"""Week-long shadow test of a feature set (e.g. the defender ratings) on live slates.

Every daily retrain also trains a shadow model: the champion with the feature switch flipped (app.ml.train). Each
pre-game prediction run stores its predictions in `shadow_predictions`, apart from the live ones, so picks, the
Performance page and grading never see them. Once a week (the first morning run 7+ days after the test started),
both are compared on the games played since, using the same players and the same final pre-game run:

- average miss on players with a DraftKings line (decides), and on everyone
- over/under hit rate against the DraftKings line (shown for reference)

The shadow takes over only if its average miss on DraftKings-line players is at least 0.5% lower, over at least
MIN_LINE_PREDICTIONS predictions; otherwise the live setup stays. Either way a new week starts with the champion's
setting flipped again as the shadow, so the comparison keeps running.

python -m app.ml.shadow [target]    show the current week's comparison so far
"""

import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.config import TARGETS
from app.db import SessionLocal, engine, upsert
from app.ml import tuning
from app.ml.registry import _load_bytes
from app.models import ModelConfig, PlayerProp, Prediction, ShadowModel, ShadowPrediction

WEEK_DAYS = 7
MIN_LINE_PREDICTIONS = 150  # need this many graded predictions with DraftKings lines before deciding


def save_predictions(rows: pd.DataFrame, on: date, targets: list[str]) -> int:
    """Predict the slate with each stat's shadow model and store it. Never raises: the shadow can't break a run."""
    n = 0
    for target in targets:
        try:
            with SessionLocal() as s:
                sm = s.get(ShadowModel, target)
                cfg = s.get(ModelConfig, target)
                if sm is None:
                    continue
                version, params, artifact = sm.version, sm.params, sm.artifact
                if cfg is not None and cfg.shadow_since is None:
                    cfg.shadow_since = on  # first shadow slate starts the week
                    s.commit()
            bundle = _load_bytes(artifact)
            raw = bundle["point"].predict(rows[bundle["features"]])
            point = np.clip(raw + tuning.bias_correction(target), 0, None)
            diff = {k: v for k, v in params.items() if k in tuning.FEATURE_SWITCHES}
            records = [
                {"player_id": int(pid), "game_id": gid, "target": target, "game_date": on, "model_version": version,
                 "params": diff, "predicted": float(p)}
                for pid, gid, p in zip(rows["player_id"], rows["game_id"], point)
            ]
            n += upsert(ShadowPrediction.__table__, records)
        except Exception as e:  # noqa: BLE001
            print(f"{target}: shadow predictions failed: {e!r}", flush=True)
    return n


def compare(target: str, start: date, end: date) -> dict:
    """Live vs shadow on graded games with start <= date < end (only player-games both predicted)."""
    q = (
        select(Prediction.predicted.label("live"), ShadowPrediction.predicted.label("shadow"), Prediction.actual,
               PlayerProp.line, ShadowPrediction.params)
        .join(ShadowPrediction, (ShadowPrediction.player_id == Prediction.player_id) & (ShadowPrediction.game_id == Prediction.game_id)
              & (ShadowPrediction.target == Prediction.target))
        .outerjoin(PlayerProp, (PlayerProp.player_id == Prediction.player_id) & (PlayerProp.game_id == Prediction.game_id)
                   & (PlayerProp.target == Prediction.target))
        .where(Prediction.target == target, Prediction.actual.is_not(None), Prediction.game_date >= start, Prediction.game_date < end,
               Prediction.model_version.not_like("backtest-%"))
    )
    with engine.connect() as conn:
        df = pd.read_sql(q, conn)
    out = {"target": target, "start": start.isoformat(), "end": (end - timedelta(days=1)).isoformat(), "n": len(df)}
    if df.empty:
        return {**out, "n_lines": 0}
    out["shadow_params"] = df["params"].iloc[-1] if isinstance(df["params"].iloc[-1], dict) else {}
    lines = df[df["line"].notna()]
    out["n_lines"] = len(lines)
    for who in ("live", "shadow"):
        out[f"{who}_mae"] = float((df[who] - df["actual"]).abs().mean())
        if len(lines):
            out[f"{who}_mae_lines"] = float((lines[who] - lines["actual"]).abs().mean())
            out[f"{who}_hit_rate"] = _hit_rate(lines[who], lines["line"], lines["actual"])
    return out


def _hit_rate(pred: pd.Series, line: pd.Series, actual: pd.Series) -> float | None:
    """Share of over/under picks (over when the prediction is above the line) that won; pushes and no-picks excluded."""
    pick = np.sign(pred - line)
    result = np.sign(actual - line)
    graded = (pick != 0) & (result != 0)
    return float((pick[graded] == result[graded]).mean()) if graded.any() else None


def describe(r: dict) -> str:
    if not r.get("n_lines"):
        return f"{r['target']} {r['start']}..{r['end']}: no graded predictions with lines yet"
    switch = ", ".join(f"{k}={v}" for k, v in r.get("shadow_params", {}).items()) or "shadow"

    def pct(v):
        return "n/a" if v is None else f"{v:.1%}"
    return (f"{r['target']} {r['start']}..{r['end']}, shadow {switch}: avg miss on {r['n_lines']} DraftKings-line predictions "
            f"live {r['live_mae_lines']:.3f} vs shadow {r['shadow_mae_lines']:.3f}; over/under hit rate live "
            f"{pct(r['live_hit_rate'])} vs shadow {pct(r['shadow_hit_rate'])}; all {r['n']}: {r['live_mae']:.3f} vs {r['shadow_mae']:.3f}")


def weekly_decision(today: date) -> list[str]:
    """For each stat whose shadow week is up: compare, switch if the shadow won, and start a new week."""
    notes = []
    for target in TARGETS:
        with SessionLocal() as s:
            cfg = s.get(ModelConfig, target)
            since = cfg.shadow_since if cfg else None
        if since is None or (today - since).days < WEEK_DAYS:
            continue
        r = compare(target, since, today)
        if r.get("n_lines", 0) < MIN_LINE_PREDICTIONS:
            notes.append(f"{describe(r)} -> not enough graded predictions yet, extending the test")
            continue
        won = r["shadow_mae_lines"] < r["live_mae_lines"] * (1 - tuning.MIN_IMPROVEMENT)
        if won:
            for key, value in r.get("shadow_params", {}).items():
                tuning.set_switch(target, key, int(value), "shadow test")
        verdict = "shadow wins, switching to it" if won else "live model kept"
        note = f"{describe(r)} -> {verdict}"
        with SessionLocal() as s:
            cfg = s.get(ModelConfig, target)
            cfg.shadow_since, cfg.shadow_note = None, note[:512]  # the next shadow slate starts a new week
            if won:  # the old shadow is now the live setup; today's retrain makes the new (flipped) shadow
                s.query(ShadowModel).filter(ShadowModel.target == target).delete()
            s.commit()
        notes.append(note)
    return notes


def main(targets: list[str]) -> None:
    from app.clock import today_et

    with SessionLocal() as s:
        for target in targets:
            cfg = s.get(ModelConfig, target)
            if cfg and cfg.shadow_note:
                print(f"last verdict: {cfg.shadow_note}")
            since = cfg.shadow_since if cfg else None
            if since is None:
                print(f"{target}: no shadow week running (it starts with the next prediction run after a retrain)")
                continue
            print(f"this week so far: {describe(compare(target, since, today_et() + timedelta(days=1)))}")


if __name__ == "__main__":
    main(sys.argv[1:2] or TARGETS)
