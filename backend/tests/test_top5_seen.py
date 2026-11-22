"""Every pick that was ever in the top 5 is logged with the line it was first shown at, and graded at that line."""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import app.main as m
from app.db import Base
from app.ml import top5
from app.models import Top5Seen
from jobs import daily

D = date(2026, 11, 22)


@pytest.fixture
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(top5, "engine", engine)
    monkeypatch.setattr(m, "SessionLocal", sessionmaker(engine))
    return engine


def _board(rows):
    """rows: (player_id, game_id, line, confidence rank, in confidence top 5)"""
    return pd.DataFrame([{"player_id": p, "game_id": g, "stat": "pts", "pick": "over", "line": line, "odds": -110,
                          "prob": 0.6, "predicted": line + 3, "top5": top, "confidence_rank": rank, "top5_gap": False,
                          "gap_rank": 9} for p, g, line, rank, top in rows])


def _seen(engine):
    with sessionmaker(engine)() as s:
        return {r.player_id: r for r in s.execute(select(Top5Seen)).scalars()}


def test_a_pick_that_drops_out_stays_on_record_with_its_first_line(db):
    t1 = datetime(2026, 11, 22, 23, 30, tzinfo=timezone.utc)  # 6:30pm ET
    top5.record_seen(D, _board([(1, "late", 18.5, 3, True), (2, "late", 10.5, 6, False)]), now=t1)
    # 9:30: the line moved and player 1 dropped out; player 2 moved up.
    top5.record_seen(D, _board([(1, "late", 20.5, 7, False), (2, "late", 10.5, 2, True)]), now=t1 + timedelta(hours=3))
    top5.record_seen(D, _board([(1, "late", 19.5, 1, True)]), now=t1 + timedelta(hours=3, minutes=15))

    seen = _seen(db)
    assert set(seen) == {1, 2}
    assert seen[1].line == 18.5 and seen[1].first_seen_at == t1.replace(tzinfo=None)  # as first shown
    assert seen[1].last_seen_at == (t1 + timedelta(hours=3, minutes=15)).replace(tzinfo=None)
    assert seen[1].best_rank == 1
    assert seen[2].first_seen_at == (t1 + timedelta(hours=3)).replace(tzinfo=None)


def test_graded_at_the_line_first_shown(db):
    top5.record_seen(D, _board([(1, "g", 18.5, 1, True), (2, "g", 10.5, 2, True)]), now=datetime(2026, 11, 22, 23, 0))
    view = pd.DataFrame([
        # player 1 scored 19: over 18.5 (first shown) wins, though the final line was 20.5
        {"date": D.isoformat(), "player_id": 1, "game_id": "g", "stat": "pts", "player": "A", "team": "X", "opp": "Y",
         "actual": 19, "result": "loss", "top5": False, "locked": False},
        {"date": D.isoformat(), "player_id": 2, "game_id": "g", "stat": "pts", "player": "B", "team": "X", "opp": "Y",
         "actual": None, "result": "pending", "top5": True, "locked": True},
    ])
    m._data_cache.pop(("top5_seen",), None)
    out = m._top5_seen(view, fp=(("t",), ("b",)))["confidence"]
    by = {r["player_id"]: r for r in out}
    assert by[1]["result"] == "win" and by[1]["line"] == 18.5 and not by[1]["in_top5"]
    assert by[1]["units"] == pytest.approx(100 / 110, abs=1e-3)
    assert by[2]["result"] == "pending" and by[2]["in_top5"] and by[2]["locked"]


def test_only_the_last_refresh_before_tip_settles_a_game():
    now = datetime(2026, 11, 22, 23, 30, tzinfo=timezone.utc)
    soon = [{"home": "A", "state": "pre", "start": now + timedelta(minutes=25)},  # another refresh at :45 first
            {"home": "B", "state": "pre", "start": now + timedelta(minutes=10)},  # next tick is after tip: settle now
            {"home": "C", "state": "pre", "start": now - timedelta(minutes=5)}]   # running late
    assert [g["home"] for g in daily._last_refresh(soon, now)] == ["B", "C"]
