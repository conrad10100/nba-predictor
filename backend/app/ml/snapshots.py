"""Freeze each pick before tip-off so the nightly top 5 can't change after the fact."""

from sqlalchemy import and_, select

from app.db import SessionLocal, upsert
from app.models import PickSnapshot, PlayerProp, Prediction

SNAPSHOT_COLUMNS = ("line", "open_line", "over_odds", "under_odds")
PHASES = ("morning", "final")


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


def snapshot_pregame(game_ids: set[str], phases: tuple[str, ...]) -> int:
    """Freeze the current line + prediction for games that haven't started (as seen by this run's props load) under
    each of `phases`. The morning run writes both phases; the evening injury update overwrites "final". Nothing writes
    a game's snapshot once it has started."""
    if not game_ids:
        return 0
    rows = _rows(Prediction.game_id.in_(game_ids))
    return upsert(PickSnapshot.__table__, [{**r, "phase": ph, "source": "pregame"} for ph in phases for r in rows])
