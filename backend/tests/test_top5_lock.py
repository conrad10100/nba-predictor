import pandas as pd

import app.main as m
from app.ml import top5

D = "2026-11-18"


def _night(probs: dict[str, list[float]]) -> pd.DataFrame:
    """Points picks for one night: game id -> confidence of each of its picks (player ids numbered in order)."""
    rows, pid = [], 0
    for game, ps in probs.items():
        for p in ps:
            pid += 1
            rows.append({"kind": "live", "date": D, "stat": "pts", "pick": "over", "player_id": pid, "game_id": game,
                         "prob": p, "edge": round(p * 10, 1)})
    return pd.DataFrame(rows)


def _top(df: pd.DataFrame, flag: str = "top5") -> set[int]:
    return set(df.loc[df[flag], "player_id"])


def _settle(df: pd.DataFrame, locks: top5.Locks, game: str) -> None:
    """What lock_started does when `game` tips: lock its picks that are in the top 5 right now."""
    ranked = m._rank(df, locks)
    for method, (flag, _) in top5.METHODS.items():
        hit = ranked[ranked[flag] & (ranked["game_id"] == game)]
        locks.picks |= {(D, method, p, g, s) for p, g, s in zip(hit["player_id"], hit["game_id"], hit["stat"])}
    locks.games.setdefault(D, set()).add(game)


def test_a_late_injury_cant_push_out_picks_locked_at_an_earlier_tip():
    # 7:00 game: players 1-4; 9:00 game: players 5-8. At 7:00 the top 5 is 1, 2, 3 (early) and 5, 6 (late).
    early = _night({"early": [0.70, 0.68, 0.66, 0.50], "late": [0.69, 0.67, 0.55, 0.54]})
    assert _top(m._rank(early)) == {1, 2, 3, 5, 6}
    locks = top5.Locks()
    _settle(early, locks, "early")

    # 9:45 refresh: an injury makes every late pick look great. Ranked afresh they'd fill the top 5 ...
    late = _night({"early": [0.70, 0.68, 0.66, 0.50], "late": [0.90, 0.89, 0.88, 0.87]})
    assert _top(m._rank(late)) == {5, 6, 7, 8, 1}
    # ... but the early picks locked at 7:00 keep their slots and the late game only gets the 2 open ones.
    ranked = m._rank(late, locks)
    assert _top(ranked) == {1, 2, 3, 5, 6}
    assert _top(ranked, "top5_gap") == {1, 2, 3, 5, 6}
    assert sorted(ranked.loc[ranked["top5"], "confidence_rank"]) == [1, 2, 3, 4, 5]


def test_an_early_pick_outside_the_top_5_at_tip_off_cant_enter_later():
    early = _night({"early": [0.70, 0.40], "late": [0.69, 0.68, 0.67, 0.66, 0.65]})
    locks = top5.Locks()
    _settle(early, locks, "early")  # player 2 (0.40) missed the cut at 7:00
    late = _night({"early": [0.70, 0.40], "late": [0.69, 0.30, 0.30, 0.30, 0.30]})  # late picks collapse
    assert 2 not in _top(m._rank(late, locks))
    assert len(_top(m._rank(late, locks))) == 5


def test_nights_without_settled_games_rank_as_before():
    df = _night({"a": [0.9, 0.8, 0.7], "b": [0.6, 0.5, 0.4]})
    assert m._rank(df, top5.Locks()).equals(m._rank(df))
