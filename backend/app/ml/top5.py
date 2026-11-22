"""Settle each night's top 5 game by game, so a pick shown as locked stays in the top 5 at the line shown.

A game's picks are re-frozen by every refresh before its tip-off (jobs.daily --refresh --before-tip). At its last one
(within LOCK_WITHIN of tip-off, or at tip-off if that refresh didn't run) the board is settled for that game: its
picks in the top 5 at that moment are locked in (`top5_locks`), its other picks drop out of contention
(`locked_games`), its snapshots are no longer rewritten, and later games only compete for the slots still open.
Until then a game's picks in the top 5 are provisional: a later refresh can change them. Nights with no settled games
(before this existed, and the morning page) are ranked over the whole night as before.

Every board a run leaves behind is also logged (`top5_seen`), so a provisional pick that later dropped out is still on
record, with the line it was shown at.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.db import SessionLocal, engine, upsert
from app.models import LockedGame, Top5Lock, Top5Seen

TOP_N = 5
TOP_STAT = "pts"
# method -> (top-5 flag column, rank column) in the graded picks
METHODS = {"confidence": ("top5", "confidence_rank"), "gap": ("top5_gap", "gap_rank")}
# A refresh this close to a game's tip-off is its last one (the refresh cron runs every 15 minutes), so it settles it.
LOCK_WITHIN = timedelta(minutes=15)


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


def board(on: date, locks: Locks | None = None) -> pd.DataFrame:
    """The night's live final-phase picks, graded and ranked the way the Performance page shows them."""
    from app.main import _grade, _load, _rank  # the Performance page's own grading and ranking
    from app.models import PickSnapshot

    snap = _load(PickSnapshot, "final", on=on)
    if snap.empty:
        return snap
    df = _grade(snap)
    return _rank(df[df["kind"] == "live"], locks if locks is not None else load())


def settle(on: date, game_ids: set[str]) -> tuple[int, int]:
    """Settle the top 5 for games on `on` not settled yet: lock their picks that are in the top 5 on the current board.
    Returns (games settled, picks locked)."""
    locks = load()
    new = set(game_ids) - locks.games.get(on.isoformat(), set())
    if not new:
        return 0, 0
    df = board(on, locks)
    rows = []
    if not df.empty:
        for method, (flag, _) in METHODS.items():
            hit = df[df[flag].astype(bool) & df["game_id"].isin(new)]
            rows += [{"game_date": on, "method": method, "player_id": int(p), "game_id": g, "target": s}
                     for p, g, s in zip(hit["player_id"], hit["game_id"], hit["stat"])]
    upsert(Top5Lock.__table__, rows)
    upsert(LockedGame.__table__, [{"game_id": g, "game_date": on} for g in sorted(new)])
    return len(new), len(rows)


def settled_games(on: date) -> set[str]:
    return load().games.get(on.isoformat(), set())


def record_seen(on: date, df: pd.DataFrame | None = None, now: datetime | None = None) -> int:
    """Log every pick in the night's top 5 right now (both methods). A pick seen before keeps its first sighting (and
    the line it was shown at); only its last sighting and best rank move."""
    df = board(on) if df is None else df
    if df is None or df.empty:
        return 0
    now = (now or datetime.now(timezone.utc)).replace(tzinfo=None)
    rows = []
    for method, (flag, rank_col) in METHODS.items():
        hit = df[df[flag].fillna(False).astype(bool)]
        rows += [{"game_date": on, "method": method, "player_id": int(r.player_id), "game_id": r.game_id, "target": r.stat,
                  "first_seen_at": now, "last_seen_at": now, "best_rank": int(getattr(r, rank_col)), "pick": r.pick,
                  "line": float(r.line), "odds": _num(r.odds), "prob": _num(r.prob), "predicted": _num(r.predicted)}
                 for r in hit.itertuples()]
    if not rows:
        return 0
    t = Top5Seen.__table__
    stmt = (pg_insert if engine.dialect.name == "postgresql" else sqlite_insert)(t).values(rows)
    least = func.least if engine.dialect.name == "postgresql" else func.min
    stmt = stmt.on_conflict_do_update(index_elements=[c.name for c in t.primary_key.columns],
                                      set_={"last_seen_at": stmt.excluded.last_seen_at,
                                            "best_rank": least(t.c.best_rank, stmt.excluded.best_rank)})
    with engine.begin() as conn:
        conn.execute(stmt)
    return len(rows)


def _num(v) -> float | None:
    return None if v is None or pd.isna(v) else float(v)
