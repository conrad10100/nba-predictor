"""Settle each night's top 5 as games tip off, so a pick that was in the top 5 when its game started stays there.

The final picks are frozen per game shortly before its own tip-off (jobs.daily --refresh --before-tip). Ranked
afresh over the whole night, a late game's pick refreshed at 9:45 could push out an early game's pick that was in
the top 5 when that game tipped at 7:00. Instead, at each tip-off the board is settled for the games that just
started: their picks in the top 5 at that moment are locked in (`top5_locks`), their other picks drop out of
contention (`locked_games`), and later games only compete for the slots still open. Nights with no locked games
(before this existed, and the morning page) are ranked over the whole night as before.
"""

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.db import SessionLocal, upsert
from app.models import LockedGame, Top5Lock

TOP_N = 5
TOP_STAT = "pts"
# method -> (top-5 flag column, rank column) in the graded picks
METHODS = {"confidence": ("top5", "confidence_rank"), "gap": ("top5_gap", "gap_rank")}


@dataclass
class Locks:
    games: dict[str, set[str]] = field(default_factory=dict)  # ISO date -> game ids already settled
    picks: set[tuple] = field(default_factory=set)  # (ISO date, method, player_id, game_id, stat)


def key(df: pd.DataFrame, method: str) -> pd.Series:
    """What each method ranks on (higher is better); confidence breaks ties for the gap."""
    if method == "confidence":
        return df["prob"]
    return df["edge"].abs() + df["prob"].fillna(0) / 1000


def apply(df: pd.DataFrame, locks: Locks, n: int = TOP_N) -> pd.DataFrame:
    """Re-rank the nights that have settled games: locked picks keep their slots, picks from other settled games
    are out, and the open slots go to the best picks from games not settled yet. Expects graded live picks with the
    columns set by the plain ranking (app.main._rank); returns a copy."""
    df = df.copy()
    points = df["pick"].notna() & (df["stat"] == TOP_STAT)
    for d, done in locks.games.items():
        night = df[points & (df["date"] == d)]
        if night.empty:
            continue
        for method, (flag, rank_col) in METHODS.items():
            k = key(night, method)
            locked = pd.Series([(d, method, p, g, s) in locks.picks for p, g, s in
                                zip(night["player_id"], night["game_id"], night["stat"])], index=night.index)
            open_ = ~night["game_id"].isin(done)
            top = locked.copy()
            slots = n - int(locked.sum())
            if slots > 0:
                top[k[open_].sort_values(ascending=False, kind="stable").index[:slots]] = True
            # top 5 first (best first), then everyone else by the same key
            order = pd.concat([k[top].sort_values(ascending=False, kind="stable"),
                               k[~top].sort_values(ascending=False, kind="stable")]).index
            df.loc[night.index, flag] = top
            df.loc[order, rank_col] = np.arange(1, len(order) + 1)
    return df


def load() -> Locks:
    locks = Locks()
    with SessionLocal() as s:
        for g in s.execute(select(LockedGame.game_date, LockedGame.game_id)):
            locks.games.setdefault(g.game_date.isoformat(), set()).add(g.game_id)
        for r in s.execute(select(Top5Lock.game_date, Top5Lock.method, Top5Lock.player_id, Top5Lock.game_id, Top5Lock.target)):
            locks.picks.add((r.game_date.isoformat(), r.method, r.player_id, r.game_id, r.target))
    return locks


def lock_started(on: date, started: set[str]) -> tuple[int, int]:
    """Settle the top 5 for games on `on` that have started since the last call. Run before a refresh re-freezes
    anything, so the board is the one that stood at their tip-off. Returns (games settled, picks locked)."""
    locks = load()
    new = started - locks.games.get(on.isoformat(), set())
    if not new:
        return 0, 0
    from app.main import _grade, _load, _rank  # the Performance page's own grading and ranking
    from app.models import PickSnapshot

    snap = _load(PickSnapshot, "final", on=on)
    rows = []
    if not snap.empty:
        df = _grade(snap)
        df = _rank(df[df["kind"] == "live"], locks)
        for method, (flag, _) in METHODS.items():
            hit = df[df[flag].astype(bool) & df["game_id"].isin(new)]
            rows += [{"game_date": on, "method": method, "player_id": int(p), "game_id": g, "target": s}
                     for p, g, s in zip(hit["player_id"], hit["game_id"], hit["stat"])]
    upsert(Top5Lock.__table__, rows)
    upsert(LockedGame.__table__, [{"game_id": g, "game_date": on} for g in sorted(new)])
    return len(new), len(rows)
