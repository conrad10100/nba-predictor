"""Freeze each pick before tip-off so the nightly top 5 can't change after the fact."""

from datetime import date, datetime, time

from sqlalchemy import and_, select

from app.db import SessionLocal, upsert
from app.models import PickSnapshot, PlayerProp, Prediction

SNAPSHOT_COLUMNS = ("line", "open_line", "over_odds", "under_odds")
# Rebuilding a night from stored lines: a line written by this time (UTC) on game day came from a pre-game run
# (morning or 5pm ET refresh); anything later is a closing line filled in after the games.
REBUILD_CUTOFF_UTC = time(23, 0)
FIRST_REBUILD_DATE = date(2026, 11, 13)  # first live night whose lines were stored before tip-off


def _rows(where) -> list[dict]:
    q = (
        select(Prediction.player_id, Prediction.game_id, Prediction.target, Prediction.game_date, Prediction.predicted,
               Prediction.p10, Prediction.p90, Prediction.model_version, *[getattr(PlayerProp, c) for c in SNAPSHOT_COLUMNS])
        .join(PlayerProp, and_(PlayerProp.player_id == Prediction.player_id, PlayerProp.game_id == Prediction.game_id,
                               PlayerProp.target == Prediction.target))
        .where(where)
    )
    with SessionLocal() as s:
        return [dict(r._mapping) for r in s.execute(q)]


def snapshot_pregame(game_ids: set[str]) -> int:
    """Freeze the current line + prediction for games that haven't started (as seen by this run's props load).
    Later pre-game runs overwrite earlier ones; nothing writes a game's snapshot once it has started."""
    if not game_ids:
        return 0
    rows = [{**r, "source": "pregame"} for r in _rows(Prediction.game_id.in_(game_ids))]
    return upsert(PickSnapshot.__table__, rows)


def rebuild_missing(today: date) -> list[date]:
    """For live nights before snapshots existed, rebuild them from lines written before that night's games
    (by REBUILD_CUTOFF_UTC on game day). Only fills nights that have no snapshots at all."""
    with SessionLocal() as s:
        done = set(s.scalars(select(PickSnapshot.game_date).distinct()))
        dates = [d for d in s.scalars(select(Prediction.game_date).distinct().where(
            Prediction.game_date >= FIRST_REBUILD_DATE, Prediction.game_date < today, ~Prediction.model_version.like("backtest-%")))
            if d not in done]
    rebuilt = []
    for d in sorted(dates):
        cutoff = datetime.combine(d, REBUILD_CUTOFF_UTC)
        rows = [{**r, "source": "rebuilt"} for r in _rows(and_(Prediction.game_date == d, PlayerProp.updated_at <= cutoff))]
        if rows:
            upsert(PickSnapshot.__table__, rows)
            rebuilt.append(d)
    return rebuilt
