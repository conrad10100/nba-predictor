"""A refresh after an early tip-off (a 1pm weekend game) must not rewrite that game's stored predictions."""

import numpy as np
import pandas as pd

from app.ml import predict

DAY = pd.Timestamp("2026-11-22")


class _Model:
    def predict(self, X):
        return np.full(len(X), 20.0)


def test_started_games_keep_their_stored_predictions(monkeypatch):
    games = pd.DataFrame([{"id": gid, "date": DAY, "home_team_id": h, "away_team_id": a, "status": "scheduled"}
                          for gid, h, a in (("early", 1, 2), ("late", 3, 4))])
    fr = predict.Frames(pd.DataFrame(), pd.DataFrame(), games, pd.DataFrame(), pd.DataFrame())
    rows = pd.DataFrame({"player_id": [10, 11, 30], "game_id": ["early", "early", "late"], "pts": np.nan, "x": 1.0})
    saved, shadow_saved = [], []

    monkeypatch.setattr(predict, "expected_roster", lambda *a: pd.DataFrame({"player_id": [10, 11, 30]}))
    monkeypatch.setattr(predict, "injury_report", lambda on: {})
    monkeypatch.setattr(predict, "build_features", lambda *a: rows.copy())
    monkeypatch.setattr(predict, "load_active", lambda t: ("v1", {"features": ["x"], "point": _Model()}))
    monkeypatch.setattr(predict, "bias_correction", lambda t: 0.0)
    monkeypatch.setattr(predict.intervals, "apply", lambda df, t, p, cal: (p - 5, p + 5))
    monkeypatch.setattr(predict, "_save", lambda r, *a: saved.append(set(r["game_id"])))
    monkeypatch.setattr(predict.shadow, "save_predictions", lambda r, *a: shadow_saved.append(set(r["game_id"])))
    monkeypatch.setattr(predict, "_response", lambda fr, games, rows, *a: {"players": len(rows)})

    out = predict.predict_slate(fr, DAY.date(), persist=True, started={"early"})
    assert saved == [{"late"}] and shadow_saved == [{"late"}]
    assert out == {"players": 3}  # the Slate still shows every game

    saved.clear()
    predict.predict_slate(fr, DAY.date(), persist=True)  # morning run: nothing has started
    assert saved == [{"early", "late"}]
